from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from orbital_envCont import OrbitalEnvCont
import os
import torch
import itertools

def make_env(time_penalty, mass_penalty):
    def _init():
        env = OrbitalEnvCont(time_penalty, mass_penalty)
        return Monitor(env)
    return _init

def train_single_config(time_penalty, mass_penalty):
    log_name = f"{(int(time_penalty*100)):02}t_{mass_penalty:02}m"
    print(f"🚀 Training config: {log_name}")

    n_envs = 4
    env_fns = [make_env(time_penalty, mass_penalty) for _ in range(n_envs)]
    env = SubprocVecEnv(env_fns)
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log="ppo_tradeoff",
        learning_rate=3e-4,
        n_steps=2048 // n_envs,
        batch_size=32,
        policy_kwargs=dict(net_arch=[32, 32]),
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=500_000, tb_log_name=f"ppo_{log_name}")
    model.save(f"ppo_{log_name}")
    env.close()

def main():
    torch.set_num_threads(1)
    t_values = [0.01, 0.03, 0.05]
    mass_values = [0, 5, 10, 15]

    for time_penalty, mass_penalty in itertools.product(t_values, mass_values):
        log_dir = f"{(int(time_penalty*100)):02}t_{mass_penalty:02}m"
        os.makedirs(log_dir)
        train_single_config(time_penalty, mass_penalty)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
