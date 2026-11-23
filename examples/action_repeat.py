"""Hold each policy action for N physics frames and sum their raw rewards."""
import gymnasium as gym


class ActionRepeat(gym.Wrapper):
    def __init__(self, env, repeat=1):
        super().__init__(env)
        if repeat < 1:
            raise ValueError("action repeat must be positive")
        self.repeat = repeat
        self.episode_frames = 0

    def reset(self, **kwargs):
        self.episode_frames = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        reward_sum = 0.0
        frames = 0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            reward_sum += float(reward)
            frames += 1
            if terminated or truncated:
                break  # never repeat an action into a new episode
        self.episode_frames += frames
        info = dict(info, simulator_frames=frames,
                    episode_simulator_frames=self.episode_frames)
        return obs, reward_sum, terminated, truncated, info
