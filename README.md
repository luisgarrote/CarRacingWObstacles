# CarRacingObstacles-v3

An extended version of Gymnasium’s **CarRacing-v3** environment, adding:

- Static obstacles  
- Dynamic obstacles  
- Mountains  
- A ghost car that follows the racing line  


---

## Installation

You need Python 3.9+.

### 1. Clone the repository

```
git clone https://github.com/luisgarrote/CarRacingWObstacles.git
cd CarRacingWObstacles
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Install the package locally

```
pip install -e .
```

This registers the environment `CarRacingObstacles-v3`.

---

## Using the Environment

```python
import gymnasium as gym
import car_racing_obstacles   # imports registration

env = gym.make("CarRacingObstacles-v3", render_mode="human")
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

---

## Repository Structure

```
car_racing_obstacles/
│
├── car_racing_obstacles.py      # The main environment extension
├── car_dynamics.py              # Car physics (from Gymnasium)
├── car_racing.py                # Base environment (from Gymnasium)
├── register_envs.py             # Registers env into Gymnasium
│
├── demo.py
├── README.md
├── requirements.txt
├── setup.py

```

## Note

This environment is intentionally kept simple to install and run on any computer that supports Python, Gymnasium, and Box2D. Students can quickly begin training RL agents with the added complexity of obstacles and a ghost competitor.

---

## License

MIT License.
