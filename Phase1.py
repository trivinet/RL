from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from orbital_env import OrbitalEnv
import pandas as pd

env = OrbitalEnv()
env.use_rk = True  # or True

vec_env = make_vec_env(lambda: env, n_envs=1)

""" model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./ppo_orbit_tensorboard/"
)

model.learn(total_timesteps=50_000)
model.save("ppo_orbit_agent") """

model = PPO.load("ppo_orbit_agent")

obs = env.reset()
done = False
total_reward = 0
step = 0
max_steps = 10000

while not done and step < max_steps:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    step += 1

print("🛰️ Episode finished")
print(f"Total steps: {step}")
print(f"Total reward: {total_reward:.4f}")
print(f"Final mass: {env.state[-1]:.2f} kg")

env.plot_trajectory()
env.plot_xyz_trajectory()


columns = ['a (km)', 'e', 'i (rad)', 'RAAN (rad)', 'argp (rad)', 'nu (rad)', 'mass (kg)']
trajectory_df = pd.DataFrame(env.trajectory, columns=columns)
print(trajectory_df.head())
trajectory_df.to_csv("ppo_orbit_trajectory.csv", index=False)
