from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from orbital_envDiscr import OrbitalEnvDiscr
import numpy as np
import torch

def make_env():
    def _init():
        env = OrbitalEnvDiscr()
        return Monitor(env)
    return _init

def main():
    torch.set_num_threads(1)  # avoid multithreading slowdown

    n_envs = 4  # or 6–8 if your CPU allows
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,  # ✅ disables console output
        tensorboard_log="ppo_logs_discr",  # ✅ remove logging if not needed
        learning_rate=3e-4,
        n_steps=2048 // n_envs,
        batch_size=32,
        policy_kwargs=dict(net_arch=[32, 32]),
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )

    # ✅ Save exactly at 500k steps
    model.learn(total_timesteps=500_000)
    model.save("ppo_discr_best")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
