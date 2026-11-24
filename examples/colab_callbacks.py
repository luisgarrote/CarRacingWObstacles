"""Select the best checkpoint on fixed validation seeds; stop at N train episodes."""
import csv
import json
import numpy as np
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from env_utils import episode


class ValidateAndStop(StopTrainingOnMaxEpisodes):
    def __init__(self, run, validation_env, seeds, max_episodes=500, every=25):
        super().__init__(max_episodes=max_episodes, verbose=0)
        self.run = run
        self.validation_env = validation_env
        self.seeds = list(seeds)
        self.every = every
        self.training_returns = []
        self.simulator_frames = 0
        self.best_mean_return = -float("inf")
        self.validation_rows = []
        self.summaries = []

    @staticmethod
    def write_csv(path, rows):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def validate(self):
        count = len(self.training_returns)
        results = [episode(self.model, self.validation_env, seed) for seed in self.seeds]
        values = [result["episode_return"] for result in results]
        summary = dict(training_episodes=count, policy_decisions=self.num_timesteps,
                       mean_return=float(np.mean(values)), std_return=float(np.std(values)),
                       success_rate=float(np.mean([r["success"] for r in results])))
        self.validation_rows.extend(dict(training_episodes=count, **result) for result in results)
        self.summaries.append(summary)
        self.write_csv(self.run/"validation.csv", self.validation_rows)
        self.write_csv(self.run/"validation_summary.csv", self.summaries)
        if summary["mean_return"] > self.best_mean_return:
            self.best_mean_return = summary["mean_return"]
            self.model.save(str(self.run/"best_model"))
            selection = dict(**summary, validation_seeds=self.seeds,
                             selection="Highest mean raw validation return; final test cases excluded")
            (self.run/"best_model.json").write_text(json.dumps(selection, indent=2)+"\n")
        print(f"Train episodes {count}/{self.max_episodes}; "
              f"train last-100 mean={np.mean(self.training_returns[-100:]):.2f}; "
              f"validation mean={summary['mean_return']:.2f}; best={self.best_mean_return:.2f}", flush=True)

    def _on_step(self):
        keep_training = super()._on_step()
        for info in self.locals["infos"]:
            self.simulator_frames += int(info.get("simulator_frames", 1))
            if "episode" in info:
                self.training_returns.append(float(info["episode"]["r"]))
                if len(self.training_returns) % self.every == 0 or not keep_training:
                    self.validate()
        return keep_training
