"""Select the best checkpoint on a frozen 5+5 suite; stop at N train episodes."""
import copy
import csv
import json
import numpy as np
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from env_utils import episode


class ValidateAndStop(StopTrainingOnMaxEpisodes):
    def __init__(self, run, validation_cases, make_validation_env, max_episodes=500, every=25):
        super().__init__(max_episodes=max_episodes, verbose=0)
        self.run = run
        self.validation_cases = copy.deepcopy(validation_cases)
        self.make_validation_env = make_validation_env
        self.every = every
        self.training_returns = []
        self.simulator_frames = 0
        self.best_mean_return = -float("inf")
        self.validation_rows = []
        self.summaries = []
        self.group_summaries = []
        (run/"validation_cases.json").write_text(json.dumps(
            dict(split="validation", cases=self.validation_cases), indent=2)+"\n")

    @staticmethod
    def write_csv(path, rows):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def validate(self):
        count = len(self.training_returns)
        results = []
        for case in self.validation_cases:
            env = self.make_validation_env(case)
            try:
                result = episode(self.model, env, case["seed"])
                results.append(dict(training_episodes=count, case_id=case["id"],
                                    group=case["group"], **result))
            finally:
                env.close()

        def summarize(group, rows):
            values = [row["episode_return"] for row in rows]
            return dict(training_episodes=count, policy_decisions=self.num_timesteps,
                        group=group, episodes=len(rows), mean_return=float(np.mean(values)),
                        std_return=float(np.std(values)),
                        success_rate=float(np.mean([row["success"] for row in rows])))

        summary = summarize("overall", results)
        groups = {group: summarize(group, [row for row in results if row["group"] == group])
                  for group in sorted({row["group"] for row in results})}
        self.validation_rows.extend(results)
        self.summaries.append(summary)
        self.group_summaries.extend(groups.values())
        self.write_csv(self.run/"validation.csv", self.validation_rows)
        rows = sorted(self.summaries+self.group_summaries,
                      key=lambda row: (row["training_episodes"], row["group"]))
        self.write_csv(self.run/"validation_summary.csv", rows)
        if summary["mean_return"] > self.best_mean_return:
            self.best_mean_return = summary["mean_return"]
            self.model.save(str(self.run/"best_model"))
            selection = dict(**summary, groups=groups,
                             validation_case_ids=[case["id"] for case in self.validation_cases],
                             validation_seeds=[case["seed"] for case in self.validation_cases],
                             selection="Highest mean raw return across all 10 validation cases; final tests excluded")
            (self.run/"best_model.json").write_text(json.dumps(selection, indent=2)+"\n")
        group_text = "; ".join(f"{name}={value['mean_return']:.2f}" for name,value in groups.items())
        print(f"Train episodes {count}/{self.max_episodes}; "
              f"train last-100 mean={np.mean(self.training_returns[-100:]):.2f}; "
              f"validation {group_text}; overall={summary['mean_return']:.2f}; "
              f"best={self.best_mean_return:.2f}", flush=True)

    def _on_step(self):
        keep_training = super()._on_step()
        for info in self.locals["infos"]:
            self.simulator_frames += int(info.get("simulator_frames", 1))
            if "episode" in info:
                self.training_returns.append(float(info["episode"]["r"]))
                if len(self.training_returns) % self.every == 0 or not keep_training:
                    self.validate()
        return keep_training
