"""Regression checks for the classroom/SB3 contract (python -m unittest discover)."""
import math
import os
import pickle
import random
import unittest
from types import SimpleNamespace

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import gymnasium as gym
import numpy as np
from car_racing_obstacles import CarRacingObstacles, register


class EnvironmentRegressionTests(unittest.TestCase):
    def make(self, **kwargs):
        env = CarRacingObstacles(render_mode='rgb_array', **kwargs)
        self.addCleanup(env.close)
        return env

    def test_registration_is_idempotent(self):
        register(); register()
        env = gym.make('CarRacingObstacles-v3', render_mode='rgb_array')
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=123)
        self.assertEqual(obs.shape, (96, 96, 3))

    def test_both_action_modes_and_final_rgb_with_all_extras(self):
        for continuous in (False, True):
            with self.subTest(continuous=continuous):
                env = self.make(continuous=continuous, n_static_obstacles=2,
                                n_dynamic_obstacles=2, n_mountains=2, use_ghost=True)
                obs, _ = env.reset(seed=123)
                self.assertEqual(env.currentIters, 0)
                self.assertEqual(env.still_time, 0.0)
                np.testing.assert_array_equal(obs, env._render('state_pixels'))
                positions = [tuple(d['body'].position) for d in env.dynamic_obstacles]
                for _ in range(8):
                    action = np.array([0, .3, 0], np.float32) if continuous else 3
                    obs, reward, terminated, truncated, _ = env.step(action)
                    self.assertTrue(env.observation_space.contains(obs))
                    self.assertTrue(np.isfinite(reward))
                    np.testing.assert_array_equal(obs, env._render('state_pixels'))
                    self.assertFalse(terminated or truncated)
                self.assertTrue(any(a != tuple(d['body'].position)
                                    for a, d in zip(positions, env.dynamic_obstacles)))

    def test_no_false_laps_after_six_tiles(self):
        env = self.make()
        env.reset(seed=123)
        maximum = 0
        for _ in range(200):
            _, _, terminated, truncated, info = env.step(np.array([0, .5, 0], np.float32))
            maximum = max(maximum, env.tile_visited_count)
            self.assertFalse(info.get('lap_finished', False))
            if terminated or truncated:
                break
        self.assertGreater(maximum, 5)
        self.assertNotEqual(env.tile_idx, 0)

    @staticmethod
    def contact(tile, wheel):
        return SimpleNamespace(fixtureA=SimpleNamespace(body=SimpleNamespace(userData=tile)),
                               fixtureB=SimpleNamespace(body=SimpleNamespace(userData=wheel)))

    def test_finish_crossing_checks_coverage_even_on_revisited_tile(self):
        env = self.make()
        env.reset(seed=123)
        tile = env.road[0]
        tile.road_visited = True
        env._left_start = True
        listener = env.world.contactListener_bug_workaround
        env.tile_visited_count = 6
        listener._contact(self.contact(tile, env.car.wheels[0]), True)
        self.assertFalse(env.new_lap)
        env.tile_visited_count = math.ceil(len(env.track)*env.lap_complete_percent)
        listener._contact(self.contact(tile, env.car.wheels[0]), True)
        self.assertTrue(env.new_lap)
        _, _, terminated, truncated, info = env.step(np.zeros(3, np.float32))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info['is_success'])

    def test_lap_cannot_cancel_collision(self):
        env = self.make()
        env.reset(seed=123)
        env.new_lap = True
        x, y = env.car.hull.position
        env._create_static_obstacle(float(x), float(y), 0.3)
        _, _, terminated, truncated, info = env.step(np.zeros(3, np.float32))
        self.assertTrue(info['collision_static'])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info['is_success'])

    def test_lap_cannot_cancel_stuck_or_offroad_failures(self):
        for cause in ('stuck', 'offroad'):
            with self.subTest(cause=cause):
                env = self.make()
                env.reset(seed=123)
                env.new_lap = True
                if cause == 'stuck':
                    env.still_time = env.still_terminate_time
                else:
                    env.offroad_time = env.offroad_terminate_time
                    env._on_track = lambda *args: False
                _, _, terminated, _, info = env.step(np.zeros(3, np.float32))
                self.assertTrue(terminated)
                self.assertFalse(info['is_success'])

    def test_internal_horizon_is_a_truncation(self):
        env = self.make(max_episode_steps=1)
        env.reset(seed=123)
        _, _, terminated, truncated, _ = env.step(np.zeros(3, np.float32))
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_unseeded_resets_use_only_environment_rng(self):
        first = self.make(n_dynamic_obstacles=1, domain_randomize=True)
        second = self.make(n_dynamic_obstacles=1, domain_randomize=True)
        first.reset(seed=123); second.reset(seed=123)
        for _ in range(2):
            random.seed(42); a, _ = first.reset()
            random.seed(999); b, _ = second.reset()
            np.testing.assert_array_equal(a, b)
            np.testing.assert_allclose(first.track, second.track)

    def test_visual_ghost_cannot_collect_tiles_or_collide(self):
        env = self.make(use_ghost=True)
        env.reset(seed=123)
        tile = env.road[len(env.road)//2]
        before = env.reward
        listener = env.world.contactListener_bug_workaround
        listener._contact(self.contact(tile, env.ghost_car.wheels[0]), True)
        self.assertFalse(tile.road_visited)
        self.assertEqual(before, env.reward)
        for body in env.ghost_car.drawlist:
            for fixture in body.fixtures:
                self.assertEqual(fixture.filterData.maskBits, 0)
                self.assertTrue(fixture.sensor)

    def test_ghost_does_not_change_main_car_trajectory_or_rewards(self):
        first = self.make(use_ghost=False)
        second = self.make(use_ghost=True)
        first.reset(seed=123); second.reset(seed=123)
        for _ in range(40):
            a = np.array([.1, .4, 0], np.float32)
            _, r1, t1, x1, _ = first.step(a)
            _, r2, t2, x2, _ = second.step(a)
            self.assertAlmostEqual(r1, r2)
            self.assertEqual((t1, x1), (t2, x2))
            np.testing.assert_allclose(first.car.hull.position, second.car.hull.position, atol=1e-6)

    def test_pickle_keeps_task_configuration(self):
        env = self.make(n_dynamic_obstacles=2, use_ghost=True, max_episode_steps=37)
        restored = pickle.loads(pickle.dumps(env))
        self.addCleanup(restored.close)
        self.assertEqual(restored.n_dynamic_obstacles, 2)
        self.assertTrue(restored.use_ghost)
        self.assertEqual(restored.maxIters, 37)

    def test_mountain_hazard_matches_rendered_world_location(self):
        env = self.make()
        env.reset(seed=123)
        mountain = env._create_mountain_body(40.0, 50.0, 3.0)
        np.testing.assert_allclose(mountain.position, (40.0, 50.0))
        polygon = np.asarray(env._world_poly_from_body(mountain))
        self.assertTrue(np.all(np.linalg.norm(polygon - [40.0, 50.0], axis=1) < 3.6))
        self.assertTrue(env._on_hazard(40.0, 50.0))
        self.assertFalse(env._on_hazard(0.0, 0.0))


if __name__ == '__main__':
    unittest.main()
