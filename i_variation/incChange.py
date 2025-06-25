import signal
import sys
import os # To ensure the log directory exists before saving

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
import numpy as np

# Define a global variable to hold the model instance so the signal handler can access it
global global_model
# Define a specific path for interrupted saves, updated for inclination
SAVE_PATH = "ppo_inc_01t_05m_interrupted" 

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

# --- Main Training Script Modifications ---

# Updated log directory for inclination
log_dir = "./ppo_inc"
# Import the new OrbitalEnvInc class
# Make sure to save the OrbitalEnvInc class in a file named orbital_env_inc.py
# For this example, I'm assuming it's in a file that can be imported directly
# Or, if this script contains both classes, you might just use OrbitalEnvInc directly
from orbital_envInc import OrbitalEnvInc # Make sure this file exists and contains OrbitalEnvInc

raw_env = OrbitalEnvInc(timepenalty=0.01, masspenalty=5)
check_env(raw_env)  # ✅ Check raw, unwrapped env

env = DummyVecEnv([lambda: Monitor(raw_env)])  # Wrap after check

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=5e-3,
    n_steps=4096,
    policy_kwargs=dict(net_arch=[64, 64]),
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.999,
    clip_range=0.3,
    ent_coef=0.1
)

# Assign the created model to the global variable so the signal handler can access it
global_model = model

print(f"Training PPO model for {500_000} timesteps. Press Ctrl+C to interrupt and save.")

try:
    model.learn(total_timesteps=500_000, tb_log_name="ppo_inc_01t_05m")
    # If training completes normally, save the final model
    model.save("ppo_inc_01t_05m_final")
    print("Training completed. Model saved as 'ppo_inc_01t_05m_final'.")
except KeyboardInterrupt:
    print("Training interrupted by user. Model save handled by signal handler.")
except Exception as e:
    print(f"An unexpected error occurred during training: {e}")
finally:
    env.close()
    print("Environment closed.")