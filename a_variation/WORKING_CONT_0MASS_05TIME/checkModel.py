import numpy as np
from stable_baselines3 import PPO
from orbital_envCont import OrbitalEnvCont

model_path = "ppo_cont"  # adjust if needed
model = PPO.load(model_path)

N = 100  # number of episodes to test
success_count = 0

results = []
env = OrbitalEnvCont()

for i in range(N):
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
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

# Summary
success_rate = 100 * success_count / N
avg_steps = np.mean([r["steps"] for r in results])
avg_error = np.mean([r["a_error"] for r in results])
avg_fuel = np.mean([r["fuel_used"] for r in results])

print(f"\n✅ Evaluation complete:")
print(f"Success rate: {success_rate:.2f}% ({success_count}/{N})")
print(f"Average a_error: {avg_error:.2f} km")
print(f"Average steps: {avg_steps:.1f}")
print(f"Average fuel used: {avg_fuel:.2f} kg")
