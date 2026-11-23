"""Evaluate all ten frozen cases (5 + 5), saving every episode and group means."""
import argparse
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from env_utils import make_env, prepare_env, load_cases, episode, save_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--max-steps",type=int,default=None,help="short integration checks only; changes benchmark")
    args = parser.parse_args()
    run = Path(args.run); config = json.loads((run/"config.json").read_text())
    torch.set_num_threads(1)
    model = PPO.load(str(run/"model.zip"),device="cpu")
    rows = []
    for case in load_cases():
        env = prepare_env(make_env(case,max_steps=args.max_steps),config["history"])
        try:
            result = episode(model,env,case["seed"])
            rows.append(dict(case_id=case["id"],group=case["group"],**result))
            print(case["id"], "return:", round(result["episode_return"],2), "success:",result["success"])
        finally:
            env.close()
    if args.max_steps:
        print("SHORT CHECK: results use a modified horizon and are not benchmark scores.")
    name = f"evaluation_short_{args.max_steps}" if args.max_steps else "evaluation"
    save_results(rows,run/name)


if __name__ == "__main__":
    main()
