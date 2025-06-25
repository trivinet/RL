from orbital_envCont import OrbitalEnvCont
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Initialize environment with specific penalty config
env = OrbitalEnvCont(timepenalty=0.01, masspenalty=5)

gve_trajectory = []

done = False
env.state = np.array([7000, 0.02, 51*np.pi/180, 120*np.pi/180, 45*np.pi/180, 0.0, 500])

# Simulate with zero thrust
while not done:
    action = [1]
    obs, reward, done, _, _ = env.step(action)
    gve_trajectory.append(env.state[:6])  # a, e, i, RAAN, argp, nu

gve_trajectory = np.array(gve_trajectory)
time_array = np.arange(len(gve_trajectory)) * env.dt / 3600  # Convert to hours

# Export Python trajectory
df = pd.DataFrame(
    np.column_stack((time_array, gve_trajectory)),
    columns=['time', 'a', 'e', 'i', 'RAAN', 'argp', 'nu']
)
df.to_csv('python_orbit_gve.csv', index=False)

# Load datasets
matlab_df = pd.read_csv('matlab_orbit_j2.csv', header=None)
matlab_df.columns = ['time', 'a', 'e', 'i', 'RAAN', 'argp', 'nu']
matlab_df['time'] = matlab_df['time'] / 3600  # Convert to hours

python_df = pd.read_csv('python_orbit_gve.csv')
python_df[['i', 'RAAN', 'argp', 'nu']] = np.degrees(python_df[['i', 'RAAN', 'argp', 'nu']])

# Plotting
elements = ['a', 'e', 'i', 'RAAN', 'argp']
labels = ['Semi-major axis (km)', 'Eccentricity', 'Inclination (°)', 'RAAN (°)', 'Argument of Periapsis (°)']

for el, label in zip(elements, labels):
    plt.figure(figsize=(8, 4))
    plt.plot(matlab_df['time'], matlab_df[el], label='MATLAB (Cartesian)',
             linestyle=':', color='black', linewidth=1.2)
    plt.plot(python_df['time'], python_df[el], label='Python (GVE)',
             linestyle='-', color='black', linewidth=1)
    plt.xlabel('Time (hours)')
    plt.ylabel(label)
    plt.title(f'Evolution of {label}', fontsize=13)
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.5, color='gray')
    plt.tight_layout()
    plt.show()
