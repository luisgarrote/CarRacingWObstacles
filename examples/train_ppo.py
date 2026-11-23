"""A direct SB3 PPO example: Monitor -> PPO.learn -> model.save -> reward plot."""
import argparse
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env_utils import make_env, prepare_env, DEFAULT_HISTORY, POLICY
from plot_rewards import plot_rewards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100000)
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
    if min(args.steps,args.history,args.rollout,args.batch_size,args.epochs)<1 or min(args.traffic,args.obstacles)<0:
        parser.error("counts must be positive (traffic/obstacles may be zero)")
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
                    obstacles=args.obstacles, randomize=args.randomize)
    base = Monitor(base, filename=str(output/"training"))  # saves raw episode returns
    env = prepare_env(base, history=args.history)
    try:
        env.seed(args.seed)
        # This is the model students change when comparing SB3 algorithms.
        model = PPO(POLICY, env, seed=args.seed, device="cpu", verbose=1,
                    n_steps=args.rollout, batch_size=args.batch_size, n_epochs=args.epochs,
                    learning_rate=3e-4)
        model.learn(total_timesteps=args.steps)
        model.save(str(output/"model"))
        config = vars(args) | {"algorithm":"PPO", "policy":POLICY,
                              "actual_timesteps":model.num_timesteps}
        (output/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    finally:
        env.close()
    plot_rewards(output/"training.monitor.csv", output/"rewards.png")
    print(f"Saved model, configuration, Monitor log and graph in {output}")


if __name__ == "__main__":
    main()
