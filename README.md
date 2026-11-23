# CarRacingObstacles-v3

A Gymnasium CarRacing environment with static obstacles, moving obstacles,
mountains/trees, and an optional **visual-only ghost car**. The learner controls
one car. Observations are 96 x 96 RGB images (`uint8`).

## Install

Use Python 3.10+ in an activated virtual environment. Python 3.11 and 3.12 are
included in CI. Box2D needs a C++ compiler if a compatible wheel is unavailable.
Install SWIG before building its bindings:

```sh
git clone https://github.com/luisgarrote/CarRacingWObstacles.git
cd CarRacingWObstacles
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel swig
python -m pip install --no-build-isolation -r requirements.txt
python -m pip install -e ".[sb3]"
```

Omit `[sb3]` for the environment alone. `pygame-ce` provides the `pygame` import;
avoid installing it alongside the separate `pygame` distribution.

## Use

Import the module once to register the environment in the current Python process.
Registration is idempotent, so repeated imports/calls in notebooks are safe.

```python
import gymnasium as gym
import car_racing_obstacles

env = gym.make(
    "CarRacingObstacles-v3",
    render_mode="rgb_array",  # use "human" for an interactive window
    continuous=True,
    n_static_obstacles=5,
    n_dynamic_obstacles=5,
    n_mountains=0,
    use_ghost=True,
)
obs, info = env.reset(seed=123)
for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        break
env.close()
```

Continuous actions are `[steering, gas, brake]`, bounded by `[-1, 1]`, `[0, 1]`
and `[0, 1]`. Set `continuous=False` for the five discrete actions (no-op, left,
right, gas, brake). DQN requires discrete mode. PPO and A2C can use either mode;
DDPG and TD3 require continuous mode.

For Stable-Baselines3, use `CnnPolicy`; stack recent image frames when temporal
information is needed. SB3 handles image intensity normalization. Import
`car_racing_obstacles` in each process that constructs an environment, including
Zoo workers. The installed wheel includes the modules, so importing does not
require running from this source directory.

## Task semantics and reproducibility

- Only the learner's wheels earn new-tile rewards. The ghost is visible but
  cannot collide, change the learner's trajectory, or collect rewards.
- This is a **single-lap task**: visiting all tiles, or crossing the start tile
  after leaving it and meeting `lap_complete_percent` (default 0.95), ends the
  task. Tile visit flags are not cleared during an episode.
- A static-obstacle collision, prolonged stopping/off-road driving, or leaving
  the playfield ends the episode. Successful lap handling never cancels a failure.
- The default 1000-action horizon is a **truncation**. `CarRacingObstacles`
  accepts `max_episode_steps` directly; `gym.make(..., max_episode_steps=...)`
  also supports Gymnasium's external TimeLimit wrapper.
- `info` includes `lap_finished`, `is_success`, and cause-specific flags such as
  `collision_static`, `stuck`, `offroad_too_long`, and `out_of_bounds` when set.
  `collision_static` is not a metric for every type of physical contact.
- `reset(seed=...)` seeds the environment RNG; subsequent unseeded resets
  continue that RNG, independently of Python's process-global random state.
- RGB observations reflect all object creation/movement for that reset or step.
- `domain_randomize=True` changes appearance at reset. Both appearance and
  track generation draw from the environment RNG. A fixed-appearance run can
  still generate different tracks.

### Version 0.2.0 migration

This release fixes empty wheels, automatic registration, false lap events,
repeated tile rewards, ghost reward attribution/collisions, cancelled failure
terminations, moving-obstacle NumPy/Box2D conversion errors, time-limit flags,
reset RNG handling and stale RGB observations. Ghost wheel transforms and motion
rate now follow the declared heading and simulation timestep. Pickling preserves
obstacle and horizon settings.

**Retrain policies and regenerate benchmark results** after updating. Returns and
lap counts from the earlier task implementation are not directly comparable.
The legacy DQN notebook remains an exercise with TODO cells, rather than an SB3
implementation. Its replay buffer now keeps a monotonic insertion counter.

## Checks

```sh
python -m unittest discover -s tests -v
python -m pip wheel . --no-deps --no-build-isolation -w dist
python -m pip install --no-deps --force-reinstall dist/*.whl
# From a DIFFERENT directory, test that the installed wheel works:
python -c "import car_racing_obstacles, gymnasium as gym; e=gym.make('CarRacingObstacles-v3', render_mode='rgb_array'); e.reset(seed=123); e.close()"
```

CI runs the environment regressions and wheel installation check on Linux and
macOS. The `demo.py` script provides keyboard controls. See the source constructor
for obstacle, appearance and ghost settings.

## License

MIT License.
