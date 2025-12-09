from setuptools import setup, find_packages

setup(
    name="car_racing_obstacles",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium","box2d",
        "pygame",
        "numpy"
    ],
    description="Extended CarRacing-v3 environment with obstacles, mountains, and ghost car.",
    author="APA2025 - LG",
)
