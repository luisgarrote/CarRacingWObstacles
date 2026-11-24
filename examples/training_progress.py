"""Episode-based stopping, live statistics and occasional SB3 checkpoints."""
import json
import time
import numpy as np
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from plot_rewards import plot_rewards


class EpisodeProgress(StopTrainingOnMaxEpisodes):
    def __init__(self, output, max_episodes=None):
        # The example uses one environment, so this counts total episodes.
        super().__init__(max_episodes=max_episodes or 10**12, verbose=1)
        self.output = output
        self.target = max_episodes
        self.returns = []
        self.successes = []
        self.frames = 0
        self.started = time.monotonic()

    def _on_step(self):
        keep_training = super()._on_step()
        for info in self.locals["infos"]:
            self.frames += info.get("simulator_frames", 1)
            if "episode" not in info:
                continue
            self.returns.append(float(info["episode"]["r"]))
            self.successes.append(bool(info.get("is_success", False)))
            count = len(self.returns)
            progress = dict(completed_episodes=count, target_episodes=self.target,
                            policy_decisions=self.num_timesteps, simulator_frames=self.frames,
                            last_return=self.returns[-1],
                            mean_last_100=float(np.mean(self.returns[-100:])),
                            success_rate_last_100=float(np.mean(self.successes[-100:])),
                            elapsed_seconds=time.monotonic()-self.started)
            self.output.joinpath("progress.json").write_text(json.dumps(progress, indent=2)+"\n")
            if count % 25 == 0 or not keep_training:
                plot_rewards(self.output/"training.monitor.csv", self.output/"rewards.png")
                print(f"Episodes {count}/{self.target or '-'}; last-100 mean "
                      f"{progress['mean_last_100']:.2f}; physics frames {self.frames}", flush=True)
            if count % 50 == 0:
                checkpoints = self.output/"checkpoints"
                checkpoints.mkdir(exist_ok=True)
                self.model.save(str(checkpoints/f"episode_{count:04d}"))
        return keep_training
