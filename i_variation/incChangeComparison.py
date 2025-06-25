from stable_baselines3 import PPO
from orbital_envInc import OrbitalEnvInc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Use elegant matplotlib settings
mpl.rcParams.update({
    "text.usetex": False,  # Set to True if LaTeX is configured
    "font.family": "serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "legend.frameon": True,
    "figure.dpi": 100
})

def run_case(initial_i, goal_i, label, model, max_steps=1450):
    env = OrbitalEnvInc(timepenalty=0.01, masspenalty=5)
    env.reset()
    env.state = np.array([8500, 0.01, initial_i, 120*np.pi/180, 45*np.pi/180, 0.0, 500.0])
    env.goal = np.array([8500, 0.01, goal_i, 120*np.pi/180, 45*np.pi/180])

    obs = env._get_obs()
    done = False
    fw_list, time_list = [], []
    step_count = 0
    correct = -np.sign(env.state[2]-env.goal[2])

    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        action = action * correct * 4
        obs, reward, done, _, _ = env.step(action)
        time_list.append(len(env.trajectory) * env.dt / 86400)  # Time in days
        fw_list.append(env.actions[-1][2])
        step_count += 1

    return {
        "label": label,
        "time": np.array(time_list),
        "fw": np.array(fw_list)
    }

# Load trained model
model = PPO.load("ppo_inc_01t_05m_interrupted_works_up")

# Run both cases (limit to 1 day)
lower_case = run_case(initial_i=np.radians(47.5), goal_i=np.radians(42.5), label="Lower", model=model)
raise_case = run_case(initial_i=np.radians(42.5), goal_i=np.radians(47.5), label="Raise", model=model)

# ---- Plot F_S
plt.figure(figsize=(6, 4))
plt.plot(raise_case["time"][99:], raise_case["fw"][99:], label=r"$F_W$ (Raise)", linestyle='-', color='black')
plt.plot(lower_case["time"][99:], lower_case["fw"][99:], label=r"$F_W$ (Lower)", linestyle='--', color='black')
plt.xlabel("Time [days]")
plt.ylabel(r"$F_W$")
plt.title("Cross-Track Thrust ($F_W$) Comparison ")
plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="black", framealpha=1.0)
plt.tight_layout()
plt.savefig("inc_fw_1day.png")
plt.show()

