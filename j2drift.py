import numpy as np
import matplotlib.pyplot as plt

# Constants
J2 = 1.08263e-3
R_E = 6378.0  # km
mu = 398600.0  # km^3/s^2
deg_per_rad = 180 / np.pi
sec_per_day = 86400.0

# Altitudes to simulate (km)
altitudes = [100, 200, 400, 600, 1000, 2000]

# Grayscale line styles
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (1, 1))]

# Inclination range (deg)
incl_deg = np.linspace(0, 180, 500)
incl_rad = np.radians(incl_deg)

# --- Function to plot RAAN or Argument of Perigee drift ---
def plot_j2_drift(drift_type='RAAN'):
    plt.figure(figsize=(12, 7))
    for h, style in zip(altitudes, line_styles):
        a = R_E + h
        n = np.sqrt(mu / a**3)
        
        if drift_type == 'RAAN':
            drift = -1.5 * J2 * (R_E / a)**2 * n * np.cos(incl_rad)
            ylabel = r"$\dot{\Omega}$ (deg/day)"
            title = r"RAAN Drift ($\dot{\Omega}$) vs Inclination"
        elif drift_type == 'w':
            drift = 0.75 * J2 * (R_E / a)**2 * n * (5 * np.cos(incl_rad)**2 - 1)
            ylabel = r"$\dot{\omega}$ (deg/day)"
            title = r"Argument of Perigee Drift ($\dot{\omega}$) vs Inclination"
        
        drift_deg_day = drift * deg_per_rad * sec_per_day
        plt.plot(incl_deg, drift_deg_day, label=f"{h} km", linestyle=style, color='black', linewidth=2)

    plt.axhline(0, color='gray', linewidth=1.2, linestyle='--')
    plt.title(title, fontsize=20)
    plt.xlabel("Inclination (deg)", fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(title="Altitude", fontsize=14, title_fontsize=15, loc='upper right')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0, 180)
    plt.ylim(-10, 10)
    plt.tight_layout()
    plt.show()

# Plot RAAN drift
plot_j2_drift('RAAN')

# Plot Argument of Perigee drift
plot_j2_drift('w')