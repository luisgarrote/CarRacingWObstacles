"""Record one fixed scenario as MP4; stream frames instead of keeping them in RAM."""
import argparse
import json
from pathlib import Path
import gymnasium as gym
import imageio.v2 as imageio
import torch
from PIL import Image
from stable_baselines3 import PPO
from env_utils import make_env, prepare_env, get_case, episode, VIDEO_FPS


class VideoFrames(gym.Wrapper):
    def __init__(self, env, writer):
        super().__init__(env)
        self.writer = writer
        self.active = True

    def frame(self):
        frame = Image.fromarray(self.env.render())
        frame.thumbnail((640,640))
        import numpy as np
        self.writer.append_data(np.asarray(frame))

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if self.active:
            self.frame()
        return result

    def step(self, action):
        result = self.env.step(action)
        if self.active:
            self.frame()  # capture the terminal image before VecEnv auto-reset
        if result[2] or result[3]:
            self.active = False
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--case",default="normal_1")
    parser.add_argument("--output",default="outputs/model_demo.mp4")
    parser.add_argument("--max-steps",type=int,default=None,help="short integration checks only")
    args = parser.parse_args()
    run = Path(args.run); config = json.loads((run/"config.json").read_text())
    case = get_case(args.case); torch.set_num_threads(1)
    output = Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    with imageio.get_writer(str(output),fps=VIDEO_FPS,codec="libx264",macro_block_size=1) as writer:
        raw = make_env(case,max_steps=args.max_steps)
        env = prepare_env(VideoFrames(raw,writer),config["history"])
        try:
            model = PPO.load(str(run/"model.zip"),env=env,device="cpu")
            metrics = episode(model,env,case["seed"])
        finally:
            env.close()
    output.with_suffix(".json").write_text(json.dumps(dict(case=case["id"],fps=VIDEO_FPS,**metrics),indent=2)+"\n")
    print(output)


if __name__ == "__main__":
    main()
