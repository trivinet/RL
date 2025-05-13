from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from orbital_env import OrbitalEnv
""" from orbital_env2D import OrbitalEnv2D """
import pandas as pd
import numpy as np

env = OrbitalEnv()
""" env = OrbitalEnv2D() """


""" vec_env = make_vec_env(lambda: env, n_envs=4)

model = PPO(
    "MlpPolicy",
    env=vec_env,
    learning_rate=2.5e-4,
    n_steps=1024,
    batch_size=64,
    ent_coef=0.2,
    clip_range=0.1,
    verbose=1,
    tensorboard_log="./ppo_orbit/"
) """
""" model = PPO.load("model_3D_J2_continued", env=vec_env) """
""" model.learn(total_timesteps=500_000, reset_num_timesteps=False, tb_log_name="3D_J2_continued")
model.save("model_3D_J2_continued") """

model = PPO.load("model_3D_J2_continued")

obs = env.reset()
done = False; total_reward = 0; step = 0
max_steps = 43200

while not done and step < max_steps:
    action, _ = model.predict(obs)
    """ action = np.array([1.0, 0.0, 1.0])  # full thrust in +R """
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    step += 1

print("🛰️ Episode finished")
print(f"Total steps: {step}")
print(f"Total reward: {total_reward:.4f}")
print(f"Final mass: {env.state[-1]:.2f} kg")

env.plot_trajectory()
env.plot_xyz_trajectory()
env.plot_actions()

""" columns = ['a (km)', 'e', 'i (rad)', 'RAAN (rad)', 'argp (rad)', 'nu (rad)', 'mass (kg)']
trajectory_df = pd.DataFrame(env.trajectory, columns=columns)
print(trajectory_df.head())
trajectory_df.to_csv("ppo_orbit_trajectory.csv", index=False) """
