import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import solve_ivp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

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

        # Initialize state
        self.state = self.reset()[0]  # Ensure reset returns (state, info)

    def step(self, action):
        """
        Apply action and propagate orbit.
        """
        self.state = self.orbital_dynamics(self.state, action)
        reward = self.compute_reward(self.state, action)
        done = self.check_termination(self.state)
        truncated = False  # Gymnasium requires truncated flag
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

        self.state = np.array([a, e, i, omega, Omega, nu])

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
        Render the orbit (simplified 2D visualization).
        """
        if not hasattr(self, 'trajectory'):
            self.trajectory = []

        self.trajectory.append(self.state[:2])  # Store (x, y) positions

        if len(self.trajectory) > 1:
            x, y = zip(*self.trajectory)
            plt.figure(figsize=(6,6))
            plt.plot(x, y, label="Orbit Path")
            plt.scatter(0, 0, color='yellow', marker='o', label="Earth")
            plt.xlabel("x (km)")
            plt.ylabel("y (km)")
            plt.legend()
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

obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
    if done:
        break
