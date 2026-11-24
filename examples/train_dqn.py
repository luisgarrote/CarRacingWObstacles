"""Direct SB3 DQN with discrete controls, RGB history and optional persistence."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from env_utils import make_env, prepare_env
from training_progress import EpisodeProgress
from plot_rewards import plot_rewards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000, help="physics frames per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/dqn")
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--traffic", type=int, default=0)
    parser.add_argument("--obstacles", type=int, default=0)
    args = parser.parse_args()
    if min(args.steps,args.history,args.action_repeat,args.max_steps,
           args.buffer_size,args.batch_size,args.train_freq) < 1:
        parser.error("counts must be positive")
    if min(args.learning_starts,args.traffic,args.obstacles) < 0:
        parser.error("learning-starts, traffic and obstacles must be nonnegative")
    if args.episodes is not None and args.episodes < 1:
        parser.error("--episodes must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("choose an empty output directory")
    torch.set_num_threads(1)
    base = make_env(continuous=False, max_steps=args.max_steps, action_repeat=args.action_repeat,
                    randomize=args.randomize, traffic=args.traffic, obstacles=args.obstacles)
    env = prepare_env(Monitor(base, filename=str(output/"training"),
                              info_keywords=("episode_simulator_frames", "is_success")), args.history)
    budget = (args.episodes * math.ceil(args.max_steps/args.action_repeat)
              + args.train_freq) if args.episodes else args.steps
    config = vars(args) | dict(algorithm="DQN", policy="CnnPolicy", continuous=False,
                              decision_budget=budget, learning_rate=1e-4,
                              exploration_fraction=.3, exploration_final_eps=.05,
                              target_update_interval=1000, gradient_steps=1,
                              versions={p:importlib.metadata.version(p) for p in
                                        ("stable-baselines3", "torch", "gymnasium", "numpy")})
    (output/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    callback = EpisodeProgress(output, args.episodes)
    try:
        env.seed(args.seed)
        model = DQN("CnnPolicy", env, seed=args.seed, device="cpu", verbose=1,
                    learning_rate=1e-4, buffer_size=args.buffer_size, batch_size=args.batch_size,
                    learning_starts=args.learning_starts, train_freq=args.train_freq,
                    gradient_steps=1, target_update_interval=1000,
                    exploration_fraction=.3, exploration_final_eps=.05)
        model.learn(total_timesteps=budget, callback=callback)
        model.save(str(output/"model"))
        config.update(actual_timesteps=model.num_timesteps, completed_episodes=len(callback.returns),
                      simulator_frames=callback.frames)
        (output/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    finally:
        env.close()
    plot_rewards(output/"training.monitor.csv", output/"rewards.png")
    print(f"Saved DQN model, statistics and graph in {output}")


if __name__ == "__main__":
    main()
