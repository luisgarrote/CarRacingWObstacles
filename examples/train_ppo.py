"""A direct SB3 PPO example: Monitor -> PPO.learn -> model.save -> reward plot."""
import argparse
import json
import math
import importlib.metadata
import subprocess
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env_utils import make_env, prepare_env, DEFAULT_HISTORY, POLICY
from plot_rewards import plot_rewards
from training_progress import EpisodeProgress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--episodes", type=int, default=None, help="stop after this many completed episodes instead of --steps")
    parser.add_argument("--action-repeat", type=int, default=1, help="physics frames per policy decision")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/ppo")
    parser.add_argument("--history", type=int, default=DEFAULT_HISTORY)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--rollout", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--traffic", type=int, default=0)
    parser.add_argument("--obstacles", type=int, default=0)
    args = parser.parse_args()
    if min(args.steps,args.history,args.rollout,args.batch_size,args.epochs,args.action_repeat)<1 or min(args.traffic,args.obstacles)<0:
        parser.error("counts must be positive (traffic/obstacles may be zero)")
    if args.episodes is not None and args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.rollout < 2 or args.batch_size < 2 or args.rollout % args.batch_size:
        parser.error("rollout must be >=2 and divisible by batch-size (>=2)")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("choose an empty output directory for this run")
    torch.set_num_threads(1)
    base = make_env(max_steps=args.max_steps, traffic=args.traffic,
                    obstacles=args.obstacles, randomize=args.randomize, action_repeat=args.action_repeat)
    base = Monitor(base, filename=str(output/"training"),
                   info_keywords=("episode_simulator_frames", "is_success"))
    env = prepare_env(base, history=args.history)
    # The time limit remains in PHYSICS frames, outside the policy budget.
    budget = (args.episodes * math.ceil((args.max_steps or 1000)/args.action_repeat)
              + args.rollout) if args.episodes else args.steps
    config = vars(args) | {"algorithm":"PPO", "policy":POLICY, "decision_budget":budget,
                          "versions":{p:importlib.metadata.version(p) for p in
                                      ("stable-baselines3", "torch", "gymnasium", "numpy", "box2d-py")}}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    config["source_revision"] = revision.stdout.strip()
    (output/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    callback = EpisodeProgress(output, args.episodes)
    try:
        env.seed(args.seed)
        # This is the model students change when comparing SB3 algorithms.
        model = PPO(POLICY, env, seed=args.seed, device="cpu", verbose=1,
                    n_steps=args.rollout, batch_size=args.batch_size, n_epochs=args.epochs,
                    learning_rate=3e-4)
        model.learn(total_timesteps=budget, callback=callback)
        model.save(str(output/"model"))
        config.update(actual_timesteps=model.num_timesteps, completed_episodes=len(callback.returns),
                      simulator_frames=callback.frames)
        (output/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    finally:
        env.close()
    plot_rewards(output/"training.monitor.csv", output/"rewards.png")
    print(f"Saved model, configuration, Monitor log and graph in {output}")


if __name__ == "__main__":
    main()
