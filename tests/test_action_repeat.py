"""Action persistence must preserve rewards and stop at episode boundaries."""
import unittest
import importlib.util
from pathlib import Path
import gymnasium as gym
import numpy as np
source = Path(__file__).resolve().parents[1] / "examples/action_repeat.py"
spec = importlib.util.spec_from_file_location("student_action_repeat", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ActionRepeat = module.ActionRepeat


class CounterEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 4, (1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(1, np.float32), {}

    def step(self, action):
        if self.steps >= 4:
            raise AssertionError('step called after terminal frame')
        self.steps += 1
        return np.array([self.steps], np.float32), float(self.steps), self.steps==4, False, {}


class PersistenceTests(unittest.TestCase):
    def test_repeat_sums_rewards_and_stops_on_terminal(self):
        env = ActionRepeat(CounterEnv(), 3)
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step([.5])
        self.assertEqual(reward, 6.)
        self.assertEqual(obs[0], 3)
        self.assertFalse(terminated)
        self.assertEqual(info['simulator_frames'], 3)
        obs, reward, terminated, truncated, info = env.step([.5])
        self.assertEqual(reward, 4.)
        self.assertTrue(terminated)
        self.assertEqual(info['simulator_frames'], 1)
        self.assertEqual(info['episode_simulator_frames'], 4)
        env.reset()
        self.assertEqual(env.step([.5])[-1]['episode_simulator_frames'], 3)
        env.close()
