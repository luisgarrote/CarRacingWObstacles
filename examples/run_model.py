"""Load a saved PPO model and run one named held-out scenario."""
import argparse
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from env_utils import make_env, prepare_env, get_case, episode, save_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--case",default="normal_1")
    parser.add_argument("--max-steps",type=int,default=None,help="short integration checks only")
    args = parser.parse_args()
    run = Path(args.run); config = json.loads((run/"config.json").read_text())
    case = get_case(args.case); torch.set_num_threads(1)
    env = prepare_env(make_env(case,max_steps=args.max_steps),config["history"])
    try:
        model = PPO.load(str(run/"model.zip"),env=env,device="cpu")
        result = episode(model,env,case["seed"]) | {"case_id":case["id"],"group":case["group"]}
        save_results([result],run/f"run_{case['id']}")
        print(json.dumps(result,indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
