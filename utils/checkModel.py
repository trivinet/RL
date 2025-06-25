import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from orbital_envCont import OrbitalEnvCont

t_values = [0.01, 0.03, 0.05]
mass_values = [0, 5, 10, 15]
N = 100  # test episodes per model

results_all = []

log_path = "evaluation_results.txt"
with open(log_path, "w") as log_file:
    for t in t_values:
        for m in mass_values:
            model_name = f"{int(t * 100):02}t_{int(m):02}m"
            model_path = f"ppo_{model_name}"

            print(f"\n- Evaluating model: {model_name}")
            log_file.write(f"\n- Evaluating model: {model_name}\n")

            try:
                model = PPO.load(model_path)
            except Exception as e:
                msg = f"- Could not load model {model_path}: {e}"
                print(msg)
                log_file.write(msg + "\n")
                continue

            env = OrbitalEnvCont(t, m)
            success_count = 0
            results = []

            for _ in range(N):
                obs, _ = env.reset()
                done = False

                while not done:
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, done, _, _ = env.step(action)

                a0 = env.state_0[0]
                a_goal = env.goal[0]
                a_final = env.state[0]
                m_final = env.state[-1]
                a_error = abs(a_final - a_goal)
                steps = len(env.trajectory)
                fuel_used = env.mass_initial - m_final
                success = a_error < 10
                if success:
                    success_count += 1

                results.append({
                    "a0": a0,
                    "a_goal": a_goal,
                    "a_final": a_final,
                    "a_error": a_error,
                    "steps": steps,
                    "fuel_used": fuel_used,
                    "success": success
                })

            success_rate = 100 * success_count / N
            avg_steps = np.mean([r["steps"] for r in results])
            avg_error = np.mean([r["a_error"] for r in results])
            avg_fuel = np.mean([r["fuel_used"] for r in results])

            log_file.write(
                f"- Model {model_name}:\n"
                f"  Success rate:  {success_rate:.2f}% ({success_count}/{N})\n"
                f"  Avg a_error:   {avg_error:.2f} km\n"
                f"  Avg steps:     {avg_steps:.1f}\n"
                f"  Avg fuel used: {avg_fuel:.2f} kg\n"
            )
            log_file.flush()

            results_all.append({
                "model": model_name,
                "time_penalty": t,
                "mass_penalty": m,
                "success_rate": success_rate,
                "avg_error": avg_error,
                "avg_steps": avg_steps,
                "avg_fuel": avg_fuel,
            })

            env.close()

# --- Plot tradeoffs ---
tpenalties = [r["time_penalty"] for r in results_all]
mpenalties = [r["mass_penalty"] for r in results_all]
fuels = [r["avg_fuel"] for r in results_all]
times = [r["avg_steps"] for r in results_all]
labels = [r["model"] for r in results_all]

plt.figure(figsize=(10, 6))
plt.scatter(fuels, times)

for i, label in enumerate(labels):
    plt.annotate(label, (fuels[i], times[i]), fontsize=8, alpha=0.7)

plt.xlabel("Average Fuel Used (kg)")
plt.ylabel("Average Steps")
plt.title("Tradeoff: Fuel vs Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("tradeoff_fuel_vs_time.png")
plt.show()
