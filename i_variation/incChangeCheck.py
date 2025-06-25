import signal
import sys
import os # To ensure the log directory exists before saving

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
import numpy as np
from orbital_envInc import OrbitalEnvInc # Make sure this file exists and contains OrbitalEnvInc

env = OrbitalEnvInc(timepenalty=0.01, masspenalty=5)
model = PPO.load("ppo_inc_01t_05m_interrupted_works_up")
done = False
obs, _ = env.reset()


print("Initial state:")
print(f"a={env.state[0]:.2f} km, e={env.state[1]:.4f}, i={np.degrees(env.state[2]):.2f}°, RAAN={np.degrees(env.state[3]):.2f}°, argp={np.degrees(env.state[4]):.2f}°, v={np.degrees(env.state[5]):.2f}°, m={env.state[6]:.2f} kg")

print("Goal state:")
print(f"a={env.goal[0]:.2f} km, e={env.goal[1]:.4f}, i={np.degrees(env.goal[2]):.2f}°, RAAN={np.degrees(env.goal[3]):.2f}°, argp={np.degrees(env.goal[4]):.2f}°")


while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)


print("\nFinal state:")
print(f"a={env.state[0]:.2f} km, e={env.state[1]:.4f}, i={np.degrees(env.state[2]):.2f}°, RAAN={np.degrees(env.state[3]):.2f}°, argp={np.degrees(env.state[4]):.2f}°, v={np.degrees(env.state[5]):.2f}°, m={env.state[6]:.2f} kg")


env.plot_trajectory()
env.plot_orbit_3d(title='Full 3D Orbit')
env.plot_actions()
