import signal
import sys
import os # To ensure the log directory exists before saving

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from orbital_envEcc import OrbitalEnvEcc  # adjust import if needed
import numpy as np

# Define a global variable to hold the model instance so the signal handler can access it
# This is a common pattern for signal handlers in scripts
global global_model
SAVE_PATH = "ppo_ecc_01t_05m_interrupted" # Define a specific path for interrupted saves

# --- Signal Handler Functions ---
def save_model_on_interrupt(model_to_save, path):
    """
    Saves the Stable Baselines3 model to the specified path.
    """
    if model_to_save is not None:
        print(f"\nCaught Ctrl+C. Saving model to {path}...")
        try:
            # Stable Baselines3 model.save() handles directory creation.
            model_to_save.save(path)
            print("Model saved successfully.")
        except Exception as e:
            print(f"Error saving model: {e}")
    else:
        print("\nCaught Ctrl+C. Model not yet initialized or accessible for saving.")
    sys.exit(0) # Exit cleanly after attempting to save

def signal_handler(sig, frame):
    """
    Handler for SIGINT (Ctrl+C). Calls the model saving function.
    """
    print("\nInterrupt signal received. Initiating model save...")
    # Access the global model instance
    save_model_on_interrupt(global_model, SAVE_PATH)

# Register the signal handler BEFORE your training starts
signal.signal(signal.SIGINT, signal_handler)

# --- Your original training script with modifications ---

log_dir = "./ppo_ecc"  # or any other path
raw_env = OrbitalEnvEcc(timepenalty=0.01,masspenalty=5)
check_env(raw_env)  # ✅ Check raw, unwrapped env

env = DummyVecEnv([lambda: Monitor(raw_env)])  # Wrap after check

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=3e-4,
    n_steps=4096,
    policy_kwargs=dict(net_arch=[64, 64]),
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.015
)

# Assign the created model to the global variable so the signal handler can access it
global_model = model

print(f"Training PPO model for {500_000} timesteps. Press Ctrl+C to interrupt and save.")

try:
    model.learn(total_timesteps=500_000, tb_log_name="ppo_ecc_01t_05m")
    # If training completes normally, save the final model
    model.save("ppo_ecc_01t_05m_final")
    print("Training completed. Model saved as 'ppo_ecc_01t_05m_final'.")
except KeyboardInterrupt:
    # This block will be executed if Ctrl+C is pressed, but the signal handler
    # should already have saved the model and exited. This is mostly for completeness.
    print("Training interrupted by user. Model save handled by signal handler.")
except Exception as e:
    print(f"An unexpected error occurred during training: {e}")
    # Optionally, save model even on other errors
    # save_model_on_interrupt(global_model, f"{SAVE_PATH}_error")
finally:
    env.close()
    print("Environment closed.")

# The testing part of your script (commented out in your original code)
# remains separate and can be run after a model has been saved.
"""
# env = OrbitalEnvEcc(timepenalty=0.01,masspenalty=5)
# model = PPO.load("ppo_ecc_01t_05m")
# done = False
# obs, _ = env.reset()

# print("Initial state:")
# print(f"a={env.state[0]:.2f} km, e={env.state[1]:.4f}, i={np.degrees(env.state[2]):.2f}°, RAAN={np.degrees(env.state[3]):.2f}°, argp={np.degrees(env.state[4]):.2f}°, v={np.degrees(env.state[5]):.2f}°, m={env.state[6]:.2f} kg")

# print("Goal state:")
# print(f"a={env.goal[0]:.2f} km, e={env.goal[1]:.4f}, i={np.degrees(env.goal[2]):.2f}°, RAAN={np.degrees(env.goal[3]):.2f}°, argp={np.degrees(env.goal[4]):.2f}°")

# while not done:
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, _, _ = env.step(action)

# print("\nFinal state:")
# print(f"a={env.state[0]:.2f} km, e={env.state[1]:.4f}, i={np.degrees(env.state[2]):.2f}°, RAAN={np.degrees(env.state[3]):.2f}°, argp={np.degrees(env.state[4]):.2f}°, v={np.degrees(env.state[5]):.2f}°, m={env.state[6]:.2f} kg")

# env.plot_trajectory()
# env.plot_orbit_3d(title='Full 3D Orbit')
# env.plot_actions()
"""