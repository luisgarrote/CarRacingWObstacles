# CarRacingObstacles-v3

An extended version of Gymnasium’s **CarRacing-v3** environment, adding:

- Static obstacles  
- Dynamic obstacles  
- Mountain / elevation zones  
- A ghost car that follows the racing line  
- Improved aesthetics  

This environment is designed for teaching **Deep Reinforcement Learning**, providing more realistic scenarios while remaining lightweight and deterministic.

---

## 🚀 Installation

You need Python 3.9+.

### 1. Clone the repository

```
git clone https://github.com/YOUR_USERNAME/car-racing-obstacles.git
cd car-racing-obstacles
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

## 🏎️ Using the Environment

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

## 📂 Repository Structure

```
car_racing_obstacles/
│
├── car_racing_obstacles.py      # The main environment extension
├── car_dynamics.py              # Car physics (from Gymnasium)
├── register_envs.py             # Registers env into Gymnasium
│
├── README.md
├── requirements.txt
├── setup.py
└── examples/
      └── play_keyboard.py
```

---

## 🛠️ Creating Your Own Obstacles

You can modify `car_racing_obstacles.py` to change the number, size, or behavior of obstacles.

---

## 👩‍🏫 For Students

This environment is intentionally kept simple to install and run on any computer that supports Python, Gymnasium, and Box2D. Students can quickly begin training RL agents with the added complexity of obstacles and a ghost competitor.

---

## 📄 License

MIT License.
