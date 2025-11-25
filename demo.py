import gymnasium as gym
#import car_racing_obstacles  # or your package that does the registration
from car_racing_obstacles import CarRacingObstacles
gym.register(
    id="CarRacingObstacles-v3",
    entry_point="car_racing_obstacles:CarRacingObstacles",
)

import numpy as np
import pygame
a = np.array([0.0, 0.0, 0.0], dtype=np.float32)

def register_input():
    global quit_game, restart
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                a[0] = -1.0
            if event.key == pygame.K_RIGHT:
                a[0] = +1.0
            if event.key == pygame.K_UP:
                a[1] = +1.0
            if event.key == pygame.K_DOWN:
                a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
            if event.key == pygame.K_RETURN:
                restart = True
            if event.key == pygame.K_ESCAPE:
                quit_game = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                a[0] = 0.0
            if event.key == pygame.K_RIGHT:
                a[0] = 0.0
            if event.key == pygame.K_UP:
                a[1] = 0.0
            if event.key == pygame.K_DOWN:
                a[2] = 0.0

        if event.type == pygame.QUIT:
            quit_game = True



env = gym.make("CarRacingObstacles-v3", render_mode="human")  # or "human" rgb_array
obs, info = env.reset()
#obs, info = env.reset(options={"ghost_file": "ghost_data/ghost1.npz"})
while True:
    #action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    register_input()
    if term or trunc:
        break
env.close()