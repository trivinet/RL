from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from orbital_envGTO import OrbitalEnvGTO  # adjust import if needed
import numpy as np



env = OrbitalEnvGTO(timepenalty=0.01,masspenalty=5)
model = PPO.load("ppo_gto_05t_05m_working")
done = False
obs, _ = env.reset()


print("Initial state:")
print(f"a={env.state[0]:.2f} km, e={env.state[1]:.4f}, i={np.degrees(env.state[2]):.2f}°, RAAN={np.degrees(env.state[3]):.2f}°, argp={np.degrees(env.state[4]):.2f}°, v={np.degrees(env.state[5]):.2f}°, m={env.state[6]:.2f} kg")

print(f"final_t= {env.final_t}, T_max = {env.T_max}")

print("Goal state:")
print(f"a={env.goal[0]:.2f} km, e={env.goal[1]:.4f}, i={np.degrees(env.goal[2]):.2f}°, RAAN={np.degrees(env.goal[3]):.2f}°, argp={np.degrees(env.goal[4]):.2f}°")



# Weights for prioritization
w_a = 1.0  # prioritize a increase
w_e = 0.625625  # prioritize e decrease
w_i = 80.0  # prioritize i decrease

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)


print("\nFinal state:")
print(f"a={env.state[0]:.2f} km, e={env.state[1]:.4f}, i={np.degrees(env.state[2]):.2f}°, RAAN={np.degrees(env.state[3]):.2f}°, argp={np.degrees(env.state[4]):.2f}°, v={np.degrees(env.state[5]):.2f}°, m={env.state[6]:.2f} kg")


env.plot_trajectory()
env.plot_orbit_3d(title='Full 3D Orbit')
env.plot_actions()
