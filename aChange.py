from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from orbital_envCont import OrbitalEnvCont  # adjust import if needed
import numpy as np

""" log_dir = "./ppo_cont"  # or any other path
raw_env = OrbitalEnvCont()
check_env(raw_env)  # ✅ Check raw, unwrapped env

env = DummyVecEnv([lambda: Monitor(raw_env)])  # Wrap after check

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=3e-4,
    n_steps=2048,
    policy_kwargs=dict(net_arch=[32, 32]),  # instead of 64, 64
    batch_size=32,
    gae_lambda=0.95,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.01
)

model.learn(total_timesteps=2_000_000, tb_log_name="ppo_cont_15t_0m")
model.save("ppo_cont_15t_0m")
env.close()
 """
env = OrbitalEnvCont(timepenalty=0.01,masspenalty=5)
model = PPO.load("ppo_01t_05m")
done = False
env.episode_counter += 1
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

print(f"final state achieved in {len(env.trajectory)*env.dt}s or {len(env.trajectory)*env.dt/(24*60*60)}days")


env.plot_trajectory()
env.plot_orbit_3d(title='Full 3D Orbit')
env.plot_actions()
