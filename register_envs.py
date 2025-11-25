import gymnasium as gym
from car_racing_obstacles import CarRacingObstacles

gym.register(
    id="CarRacingObstacles-v3",
    entry_point="car_racing_obstacles:CarRacingObstacles",
)

 