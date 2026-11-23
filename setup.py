from setuptools import setup

setup(
    name="car_racing_obstacles",
    version="0.2.0",
    py_modules=["car_racing_obstacles", "car_racing", "car_dynamics", "register_envs"],
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=1.0,<2", "box2d-py==2.3.8",
        "pygame-ce>=2.5,<3", "numpy>=1.23,<3",
    ],
    extras_require={"sb3": ["stable-baselines3>=2.7,<3"]},
    description="Extended CarRacing-v3 environment with obstacles and a visual ghost.",
    author="APA2025 - LG",
)
