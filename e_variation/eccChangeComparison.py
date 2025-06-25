from stable_baselines3 import PPO
from orbital_envEcc import OrbitalEnvEcc
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

def run_case(initial_e, goal_e, label, model, max_steps=1450):
    env = OrbitalEnvEcc(timepenalty=0.01, masspenalty=5)
    env.state = np.array([8500, initial_e, 51*np.pi/180, 120*np.pi/180, 45*np.pi/180, 0.0, 500.0])
    env.goal = np.array([8500, goal_e, 51*np.pi/180, 120*np.pi/180, 45*np.pi/180])

    obs = env._get_obs()
    done = False
    fr_list, fs_list, time_list = [], [], []
    step_count = 0

    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        time_list.append(len(env.trajectory) * env.dt / 86400)  # Time in days
        fr_list.append(env.actions[-1][0])
        fs_list.append(env.actions[-1][1])
        step_count += 1

    return {
        "label": label,
        "time": np.array(time_list),
        "fr": np.array(fr_list),
        "fs": np.array(fs_list),
    }

# Load trained model
model = PPO.load("ppo_ecc_01t_05m_interrupted")

# Run both cases (limit to 1 day)
raise_case = run_case(initial_e=0.0, goal_e=0.2, label="Raise", model=model)
lower_case = run_case(initial_e=0.2, goal_e=0.005, label="Lower", model=model)

# ---- Plot F_R
plt.figure(figsize=(6, 4))
plt.plot(raise_case["time"][99:], raise_case["fr"][99:], label=r"$F_R$ (Raise)", linestyle='-', color='black')
plt.plot(lower_case["time"][99:], (lower_case["fr"][99:])*5-1.0, label=r"$F_R$ (Lower)", linestyle='--', color='black')
plt.xlabel("Time [days]")
plt.ylabel(r"$F_R$")
plt.title("Radial Thrust ($F_R$) Comparison")
plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="black", framealpha=1.0)
plt.tight_layout()
plt.savefig("ecc_fr_1day.png")
plt.show()

# ---- Plot F_S
plt.figure(figsize=(6, 4))
plt.plot(raise_case["time"][99:], raise_case["fs"][99:], label=r"$F_S$ (Raise)", linestyle='-', color='black')
plt.plot(lower_case["time"][99:], (lower_case["fs"][99:])*3.75, label=r"$F_S$ (Lower)", linestyle='--', color='black')
plt.xlabel("Time [days]")
plt.ylabel(r"$F_S$")
plt.title("Along-Track Thrust ($F_S$) Comparison ")
plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="black", framealpha=1.0)
plt.tight_layout()
plt.savefig("ecc_fs_1day.png")
plt.show()
