import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import solve_ivp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from astropy.coordinates import CartesianRepresentation
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.constants import GM_earth

def kepler_to_cartesian(a, e, i, omega, Omega, nu):
    """
    Converts Keplerian elements to Cartesian coordinates.
    """
    # Convert degrees to radians for inclination, omega, and Omega
    i = np.radians(i)
    omega = np.radians(omega)
    Omega = np.radians(Omega)
    nu = np.radians(nu)

    # Create orbit object
    orb = Orbit.from_classical(Earth, a * u.km, e * u.one, i * u.rad, Omega * u.rad, omega * u.rad, nu * u.rad)

    # Extract Cartesian coordinates
    x, y, z = orb.r.to_value(u.km)  # Position (km)
    vx, vy, vz = orb.v.to_value(u.km / u.s)  # Velocity (km/s)

    return np.array([x, y, z, vx, vy, vz])


class OrbitalEnv(gym.Env):
    def __init__(self):
        super(OrbitalEnv, self).__init__()

        # Define action space (Δvx, Δvy, Δvz)
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(3,), dtype=np.float32)

        # Define observation space (Keplerian elements or Cartesian)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Define the target orbit (Keplerian elements: [a, e, i, omega, Omega, nu])
        self.target_orbit = np.array([7000, 0.01, 0, 0, 0, 0])  # Example: target circular orbit at 7000 km

        self.trajectory = []  # Initialize trajectory list
        self.actions_list = []
        self.orbital_elements_along_time = []

        # Initialize state
        self.state = self.reset()[0]  # Ensure reset returns (state, info)

    def step(self, action):
        """
        Apply action and propagate orbit.
        """
        self.state = self.orbital_dynamics(self.state, action)
        reward = self.compute_reward(self.state, action)
        done = self.check_termination(self.state)
        self.actions_list.append(action)
        truncated = False  # Gymnasium requires truncated flag
        self.orbital_elements_along_time.append()
        info = {}

        return self.state, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        a = np.random.uniform(6800, 7200)  # Semi-major axis (km)
        e = np.random.uniform(0, 0.1)  # Eccentricity
        i = np.random.uniform(0, 10)  # Inclination (degrees)
        omega = np.random.uniform(0, 360)  # Argument of periapsis
        Omega = np.random.uniform(0, 360)  # Longitude of ascending node
        nu = np.random.uniform(0, 360)  # True anomaly

        # Convert Keplerian to Cartesian
        self.state = kepler_to_cartesian(a, e, i, omega, Omega, nu)

        return self.state, {}  # Return (state, info)

    
    def check_termination(self, state):
        """
        Checks if the episode should end.
        """
        a, e, _, _, _, _ = state  # Extract orbital parameters
        
        # Define limits
        perigee = a * (1 - e)  # Compute perigee distance
        apogee = a * (1 + e)  # Compute apogee distance
        
        if perigee < 6378 + 100:  # Earth radius + 100 km
            return True  # Deorbited (end episode)
        
        if apogee > 50000:  # Arbitrary high orbit escape
            return True  # Escaped
        
        if np.linalg.norm(state - self.target_orbit) < 50:  # Close to target orbit
            return True  # Mission success

        return False


    def render(self):
        """
        Render the simulation with:
        - Plot 1: Actions (Δvx, Δvy, Δvz) vs Time
        - Plot 2: 3D Trajectory (x, y, z)
        - Plot 3: Semi-Major Axis & Eccentricity vs Time
        """

        if not hasattr(self, 'time_steps'):
            self.time_steps = []
            self.actions_list = []
            self.trajectory = []
            self.orbital_elements = []

        # Append current data
        self.time_steps.append(len(self.time_steps))  # Time step counter
        self.trajectory.append(self.state[:3])  # (x, y, z)
        self.orbital_elements.append([self.state[0], self.state[1]])  # (a, e)
        
        if hasattr(self, 'last_action'):
            self.actions_list.append(self.last_action)
        else:
            self.actions_list.append([0, 0, 0])  # Default if no action applied yet

        fig = plt.figure(figsize=(12, 6))

        # ---- Plot 1: Actions vs Time ----
        ax1 = fig.add_subplot(131)
        time_array = np.array(self.time_steps)
        actions_array = np.array(self.actions_list)

        ax1.plot(time_array, actions_array[:, 0], label="Δvx", color='r')
        ax1.plot(time_array, actions_array[:, 1], label="Δvy", color='g')
        ax1.plot(time_array, actions_array[:, 2], label="Δvz", color='b')

        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Action (Δv in km/s)")
        ax1.set_title("Actions vs Time")
        ax1.legend()
        ax1.grid()

        # ---- Plot 2: 3D Trajectory ----
        ax2 = fig.add_subplot(132, projection='3d')
        traj_array = np.array(self.trajectory)
        
        ax2.plot(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2], label="Orbit Path")
        ax2.scatter(0, 0, 0, color='yellow', marker='o', label="Earth")  # Earth at origin

        ax2.set_xlabel("X (km)")
        ax2.set_ylabel("Y (km)")
        ax2.set_zlabel("Z (km)")
        ax2.set_title("3D Trajectory")
        ax2.legend()

        # ---- Plot 3: Semi-Major Axis & Eccentricity vs Time ----
        ax3 = fig.add_subplot(133)
        orbital_array = np.array(self.orbital_elements)

        ax3.plot(time_array, orbital_array[:, 0], label="Semi-Major Axis (a)", color='purple')
        ax3.plot(time_array, orbital_array[:, 1], label="Eccentricity (e)", color='orange')

        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Orbital Parameters")
        ax3.set_title("Semi-Major Axis & Eccentricity vs Time")
        ax3.legend()
        ax3.grid()

        plt.tight_layout()
        plt.show()


    def orbital_dynamics(self, state, action, dt=10):
        """
        Simulate spacecraft motion for dt seconds.
        """
        def equations(t, y):
            x, y, z, vx, vy, vz = y
            r = np.sqrt(x**2 + y**2 + z**2)
            mu = 398600  # Earth's gravitational parameter (km^3/s^2)
            ax, ay, az = action  # Thrust acceleration

            return [vx, vy, vz, -mu*x/r**3 + ax, -mu*y/r**3 + ay, -mu*z/r**3 + az]

        sol = solve_ivp(equations, [0, dt], state, method='RK45')
        return sol.y[:, -1]  # Return final state after dt seconds

    def compute_reward(self, state, action):
        """
        Reward = -fuel_used - deviation_from_target_orbit
        """
        target_orbit = np.array([7000, 0.01, 0, 0, 0, 0])  # Example target (Keplerian)
        deviation = np.linalg.norm(state - target_orbit)
        fuel_penalty = np.linalg.norm(action)  # Penalize large thrusts

        return -deviation - 0.1 * fuel_penalty


env = OrbitalEnv()
#env = DummyVecEnv([lambda: env])  # Wrap for Stable-Baselines3 compatibility

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save the trained policy
model.save("orbital_policy")


model = PPO.load("orbital_policy")

obs, _ = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
    if done:
        break

env.render()
