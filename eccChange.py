from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
""" from orbital_env import OrbitalEnv """
from orbital_env2D import OrbitalEnv2D
import pandas as pd
import numpy as np

""" env = OrbitalEnv() """
env = OrbitalEnv2D()

vec_env = make_vec_env(lambda: env, n_envs=4)

model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    verbose=1,
    learning_rate=1e-3,
    n_steps=1024,
    seed=42,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./ppo_orbit_tensorboard/"
)

model.learn(total_timesteps=2_000_000, tb_log_name="low_thrust_2D_e")
model.save("ppo_orbit_agent_J2_2D_e")

""" model = PPO.load("ppo_orbit_agent_J2_2D_e") """

env.goal = np.array([7000, 0.2, 51*np.pi/180, 120*np.pi/180, 45.0*np.pi/180]) # [a, e, i, raan, argp]
env.state_0 = np.array([7000, 0.02, 51*np.pi/180, 120*np.pi/180, 45.0*np.pi/180, 0.0, env.mass_initial]) # [a, e, i, raan, argp, v, m]


# K parameters
env.k_parameters = np.array([0.5,2,0.5,0.5,0.5])

obs = env.reset()
done = False; total_reward = 0; step = 0
max_steps = 43200

while not done and step < max_steps:
    action, _ = model.predict(obs)
    """ action = [0,1,0,1] """
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
