import signal
import sys
import os # To ensure the log directory exists before saving
import torch.nn

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv # Added SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env # Helper for vec env creation
from stable_baselines3.common.env_checker import check_env


from orbital_envGTO import OrbitalEnvGTO # adjust import if needed
import numpy as np

# Define a global variable to hold the model instance so the signal handler can access it
global global_model
MODEL_SAVE_BASE_DIR = "./saved_models" 

# Define a specific path for interrupted saves within that directory
SAVE_PATH = os.path.join(MODEL_SAVE_BASE_DIR, "ppo_gto_01t_05m_interrupted")

# Ensure the base save directory exists BEFORE attempting to save
os.makedirs(MODEL_SAVE_BASE_DIR, exist_ok=True)

# --- Signal Handler Functions ---
def save_model_on_interrupt(model_to_save, path):
    """
    Saves the Stable Baselines3 model to the specified path.
    """
    if model_to_save is not None:
        print(f"\nCaught Ctrl+C. Saving model to {path}...")
        try:
            model_to_save.save(path)
            print("Model saved successfully.")
        except Exception as e:
            print(f"Error saving model: {e}")
    else:
        print("\nCaught Ctrl+C. Model not yet initialized or accessible for saving.")
    sys.exit(0)

def signal_handler(sig, frame):
    """
    Handler for SIGINT (Ctrl+C). Calls the model saving function.
    """
    print("\nInterrupt signal received. Initiating model save...")
    save_model_on_interrupt(global_model, SAVE_PATH)

# Register the signal handler BEFORE your training starts
signal.signal(signal.SIGINT, signal_handler)

# --- Your original training script with modifications ---

log_dir = "./ppo_gto_logs" # Changed for clarity, usually a separate dir for logs
os.makedirs(log_dir, exist_ok=True) # Ensure log directory exists

# Environment setup
# You might want to experiment with differ  ent penalty values
# Consider creating a function to make env creation more modular if using multiple envs
def make_env(timepenalty, masspenalty, log_env=True):
    # Ensure log directory for the environment is created if logging within the env
    if log_env:
        # Construct the full path for the environment's specific log directory
        # It should be a subdirectory of the main log_dir
        env_specific_log_subdir_name = f"{(int(timepenalty*100)):02}t_{masspenalty:02}m"
        full_env_log_dir = os.path.join(log_dir, "env_specific_logs", env_specific_log_subdir_name) # <-- MODIFIED
        os.makedirs(full_env_log_dir, exist_ok=True) # Create this specific directory
    else:
        full_env_log_dir = None # Or some default, or ensure OrbitalEnvGTO handles None

    # Pass the correctly created directory path to OrbitalEnvGTO
    # Assuming OrbitalEnvGTO takes a 'log_dir' argument for its internal logging
    return OrbitalEnvGTO(timepenalty=timepenalty, masspenalty=masspenalty, log=log_env, log_dir=full_env_log_dir)


if __name__ == '__main__':

    raw_env_for_check = make_env(timepenalty=0.01, masspenalty=0.5, log_env=False) # No env logging during check
    check_env(raw_env_for_check)
    del raw_env_for_check # Clean up

    # --- Vectorized Environment ---
    # For performance, especially with a complex environment, SubprocVecEnv is usually preferred over DummyVecEnv
    # num_envs_per_worker is implicitly 1 for make_vec_env when vec_env_cls is not SubprocVecEnv
    # Set num_envs to a reasonable number, often 4-8 is a good start for local training
    num_envs = 4
    env_kwargs = dict(timepenalty=0.01, masspenalty=0.5, log_env=True) # Pass args to env constructor
    env = make_vec_env(
        lambda: make_env(**env_kwargs),
        n_envs=num_envs,
        seed=0, # Fixed seed for reproducibility
        vec_env_cls=SubprocVecEnv, # Use SubprocVecEnv for parallel execution
        monitor_dir=os.path.join(log_dir, "monitor_logs") # Monitor logs for each env
    )
    # Ensure monitor log directory exists
    os.makedirs(os.path.join(log_dir, "monitor_logs"), exist_ok=True)


    # --- PPO Model Configuration ---
    # Hyperparameters for PPO often need extensive tuning.
    # These are adjusted values based on common practices for continuous control.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        n_epochs=10,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            activation_fn=torch.nn.ReLU
        ),
        seed=0,
        device="auto"
    )

    # Assign the created model to the global variable so the signal handler can access it
    global_model = model

    print(f"Training PPO model for {1_000_000} timesteps. Press Ctrl+C to interrupt and save.") # Increased total_timesteps

    try:
        model.learn(total_timesteps=1_000_000, tb_log_name="ppo_gto_01t_05m")
        # If training completes normally, save the final model
        final_model_path = os.path.join(MODEL_SAVE_BASE_DIR, "ppo_gto_01t_05m_final")
        model.save(final_model_path) # Save to the specified base directory
        print(f"Training completed. Model saved as '{final_model_path}'.")
    except KeyboardInterrupt:
        print("Training interrupted by user. Model save handled by signal handler.")
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
        # Optionally, save model even on other errors
        # save_model_on_interrupt(global_model, f"{SAVE_PATH}_error")
    finally:
        # Ensure environment is closed even if an error occurs
        env.close()
        print("Environment closed.")


