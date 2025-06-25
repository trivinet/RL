import numpy as np
import math
import gym
from gym import spaces
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

J2 = 1.08263e-3
R_EARTH = 6378.137

class OrbitalEnv2D(gym.Env):
    def __init__(self):
        super().__init__()
        self.final_t = 5 # days
        # Constants
        self.mu = 398600.4418  # km^3/s^2, Earth's gravitational parameter
        self.T_max = 5e-6   # km/s^2, max thrust acceleration
        self.mass_initial = 500.0  # kg
        self.dt = 10.0         # seconds per step
        self.include_j2 = False

        self.use_rk = True  # Toggle RK integration

        # Observation space: scaled Keplerian elements + mass ratio
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        # Action space: thrust direction (R, S) + throttle (0–1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Goal
        self.goal = np.array([7500, 0.02, 51*np.pi/180, 120*np.pi/180, 45.0*np.pi/180]) # [a, e, i, raan, argp]

        # Initial orbit
        self.state_0 = np.array([7000, 0.02, 51*np.pi/180, 120*np.pi/180, 45.0*np.pi/180, 0.0, self.mass_initial]) # [a, e, i, raan, argp, v, m]

        # K parameters
        self.k_parameters = np.array([2.0, 0.0, 0.0, 0.0, 0.0])  # only a matters

        self.last_throttle = 0

        self.state = None  # [a, e, i, raan, argp, v, m]
        self.trajectory = []
        self.actions = []
    
    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        self.steps_in_goal = 0
        self.state = self.state_0
        self.trajectory = [self.state.copy()]
        self.actions = []
        return self._get_obs()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)  # force 1D array

        raw_direction = np.array([0.0, action[1]])  # ignore Fr
        throttle = np.clip(action[2], 0.0, 1.0)
        self.last_throttle = throttle

        # ✅ Log raw action (what the policy actually chose)
        # Fw is always 0 — we explicitly include it for plotting consistency
        self.actions.append(np.array([raw_direction[0], raw_direction[1], 0.0, throttle]))

        # Normalize and apply scaling
        direction_unit = raw_direction / (np.linalg.norm(raw_direction) + 1e-8)
        a_rsw = np.array([direction_unit[0], direction_unit[1], 0.0]) * self.T_max * throttle

        assert abs(a_rsw[2]) < 1e-8, f"Fw component is not zero: {a_rsw[2]}"

        if self.use_rk:
            self.state = self._rk_integrate(self.state, a_rsw)
        else:
            self.state = self._propagate(self.state, a_rsw)

        self.trajectory.append(self.state.copy())
        reward = self._compute_reward(self.state)

        max_steps = int((self.final_t * 24 * 60 * 60) / self.dt)
        done = (
            self.state[-1] <= 0
            or len(self.trajectory) >= max_steps
            or self.steps_in_goal >= 100
        )

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        a, e, i, raan, argp, v, m = self.state
        return np.array([a/10000, e, i/np.pi, raan/(2*np.pi), argp/(2*np.pi), m/self.mass_initial])

    def _compute_reward(self, state):
      a, e, i, raan, argp, *_ = state
      goal = self.goal
      max_steps = int((self.final_t*24*60*60)/self.dt)
      # Normalize differences
      """ da = abs(a - goal[0]) / 1000         # km-scale
      de = abs(e - goal[1])
      di = abs(i - goal[2]) / np.pi        # normalize radians
      draan = abs(raan - goal[3]) / (2*np.pi)
      dargp = abs(argp - goal[4]) / (2*np.pi)

      [ka, ke, ki, kraan, kargp] = self.k_parameters

      # Orbital state reward (negative sum of normalized errors)
      orbital_reward = -(ka * da + ke * de + ki * di + kraan * draan + kargp * dargp) """

      da = abs(a - goal[0]) / 1000  # km-scale
      orbital_reward = -100.0 * da

      # Optional: penalize mass loss (reward conserving fuel)
      m = state[-1]
      """ mass_penalty = -0.0001 * (self.mass_initial - m) """

      # Optional: penalize time (to encourage faster convergence)
      """ time_penalty = -0.001 * len(self.trajectory) """

      # Goal band enforcement
      goal_band = 10 # 10 km tolerance
      goal_a_min = goal[0] - goal_band
      goal_a_max = goal[0] + goal_band

      if goal_a_min <= a <= goal_a_max:
        """ goal_reward = +10.0 * (1 - len(self.trajectory) / max_steps)  # bigger bonus if reached early """
        goal_reward = 50.0  # huge reward on success
        self.steps_in_goal += 1
      else:
        goal_reward = 0.0

        if hasattr(self, "steps_in_goal") and self.steps_in_goal > 0:
          goal_reward -= 5.0 # penalize for leaving the goal after reward
          self.steps_in_goal = 0 # reset

      """ throttle_penalty = -0.2 * (1 - self.last_throttle) # penalize low throttle """
      throttle_reward = +0.2* self.last_throttle # encourage thrust

      return orbital_reward + mass_penalty + time_penalty + goal_reward + throttle_reward #+ throttle_penalty

    def _gauss_rhs(self, t, state, a_rsw):
        a, e, i, raan, argp, v, m = state
        x = np.sqrt(1 - e**2)
        p = a * (1 - e**2)
        r = p / (1 + e * np.cos(v))
        h = np.sqrt(self.mu * p)
        n = np.sqrt(self.mu / a**3)
        u = raan + v

        Isp = 3000  # seconds, for ion engine
        g0 = 9.80665 / 1000  # km/s²
        thrust_acc = np.linalg.norm(a_rsw)

        Fr, Fs, Fw = a_rsw

        # Gauss Equations
        da = (2 * e * np.sin(v) / (n * x)) * Fr + (2 * a * x / (n * r)) * Fs
        de = (x * np.sin(v) / (n * a)) * Fr + (x / (n * a**2 * e)) * ((a**2 * x**2 / r - r) * Fs)
        di = (r * np.cos(u) / (n * a**2 * x)) * Fw
        draan = (r * np.sin(u) / (n * a**2 * x * np.sin(i))) * Fw if np.abs(np.sin(i)) > 1e-6 else 0.0
        dargp = (-x * np.cos(v) / (n * a * e)) * Fr + (p * np.sin(v) / (e * h)) * (1 + 1 / (1 + e * np.cos(v))) * Fs \
             - (r * np.cos(i) * np.sin(u) / (n * a**2 * x * np.sin(i))) * Fw if np.abs(np.sin(i)) > 1e-6 else 0.0
        dv = h / r**2 + (1 / (e * h)) * (p * np.cos(v) * Fr - (p + r) * np.sin(v) * Fs)
        dm = -m * thrust_acc / (Isp * g0)

        # J2 perturbations
        if self.include_j2:
          factor = (3/2) * J2 * (R_EARTH / p)**2 * n
          draan += -factor * np.cos(i)
          dargp += factor * (0.5 * (5 * np.cos(i)**2 - 1))

        return [da, de, di, draan, dargp, dv, dm]

    def _propagate(self, state, a_rsw):
      return state + np.array(self._gauss_rhs(0, state, a_rsw)) * self.dt
    
    def _rk_integrate(self, state, a_rsw):
      sol = solve_ivp(self._gauss_rhs, [0, self.dt], state, args=(a_rsw,), method = 'RK45')
      return sol.y[:, -1]

    def _keplerian_to_cartesian(self, a, e, i, raan, argp, v):
      # 1. Compute perifocal position and velocity
      p = a * (1 - e**2)
      r = p / (1 + e * np.cos(v))
      r_pf = r * np.array([np.cos(v), np.sin(v), 0])
      v_pf = np.sqrt(self.mu / p) * np.array([-np.sin(v), e + np.cos(v), 0])

      # 2. Rotation matrix: Perifocal to ECI
      cos_O, sin_O = np.cos(raan), np.sin(raan)
      cos_w, sin_w = np.cos(argp), np.sin(argp)
      cos_i, sin_i = np.cos(i), np.sin(i)

      R = np.array([
          [cos_O*cos_w - sin_O*sin_w*cos_i, -cos_O*sin_w - sin_O*cos_w*cos_i, sin_O*sin_i],
          [sin_O*cos_w + cos_O*sin_w*cos_i, -sin_O*sin_w + cos_O*cos_w*cos_i, -cos_O*sin_i],
          [sin_w*sin_i,                     cos_w*sin_i,                      cos_i]
      ])

      r_eci = R @ r_pf
      v_eci = R @ v_pf
      return r_eci, v_eci

    def plot_trajectory(self):
      traj = np.array(self.trajectory)
      time = np.arange(len(traj)) * self.dt
      fig, axs = plt.subplots(3, 2, figsize=(14,10))
      labels = ['a (km)', 'e', 'i (º)', 'Ω (º)', 'ω (º)', 'mass (kg)']
      for i, ax in enumerate(axs.flat[:-1]):
        if i>1 and i<5:
          ax.plot(time, traj[:, i]*180/np.pi)
        else:
          ax.plot(time, traj[:, i])
        ax.set_ylabel(labels[i])
        ax.grid()
      axs[2, 1].plot(time, traj[:, -1])
      axs[2, 1].set_ylabel('mass (kg)')
      axs[2, 1].grid()

      for ax in axs.flat:
          ax.set_xlabel('Time (s)')
      plt.tight_layout()
      plt.show()

    def plot_xyz_trajectory(self):
      # Extract orbital trajectory and convert to ECI positions
      traj = np.array(self.trajectory)
      r_eci_all = []
      for state in traj:
          r_eci, _ = self._keplerian_to_cartesian(*state[:6])
          r_eci_all.append(r_eci)
      r_eci_all = np.array(r_eci_all)

      # --- 2D Plot (XY plane) ---
      plt.figure(figsize=(6,6))
      plt.plot(r_eci_all[:,0], r_eci_all[:,1], label='Trajectory')
      # Draw Earth as a filled circle with real radius

      earth = plt.Circle((0, 0), R_EARTH, color='blue', alpha=0.5, label='Earth')
      plt.gca().add_patch(earth)
      plt.gca().set_aspect('equal')  # ensure circular shape
      plt.xlabel('X (km)')
      plt.ylabel('Y (km)')
      plt.title('Orbit projection on XY plane')
      plt.axis('equal')
      plt.grid(True)
      plt.legend()
      plt.show()

      # --- 3D Plot ---
      fig = plt.figure(figsize=(8,8))
      ax = fig.add_subplot(111, projection='3d')
      ax.plot(r_eci_all[:,0], r_eci_all[:,1], r_eci_all[:,2], label='Orbit', color='pink')

      # Draw Earth as a semi-transparent sphere
      u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
      x = R_EARTH * np.cos(u) * np.sin(v)
      y = R_EARTH * np.sin(u) * np.sin(v)
      z = R_EARTH * np.cos(v)
      ax.plot_surface(x, y, z, color='blue', alpha=0.1)

      ax.set_xlabel('X (km)')
      ax.set_ylabel('Y (km)')
      ax.set_zlabel('Z (km)')
      ax.set_title('3D Orbit with Earth')
      ax.legend()
      plt.tight_layout()
      plt.show()

    def plot_actions(self):
      actions = np.array(self.actions)
      time = np.arange(len(actions)) * self.dt

      labels = ['Fr (R)', 'Fs (S)', 'Fw (W)', 'Throttle']
      fig, axs = plt.subplots(2, 2, figsize=(12, 8))

      for i, ax in enumerate(axs.flat):
          ax.plot(time, actions[:, i])
          ax.set_title(labels[i])
          ax.set_ylim(-1.2 if i < 3 else -0.05, 1.2 if i < 3 else 1.05)  # Throttle is [0,1]
          ax.set_xlabel("Time (s)")
          ax.set_ylabel("Action value")
          ax.grid(True)

      plt.suptitle("Agent Actions vs Time", fontsize=16)
      plt.tight_layout(rect=[0, 0.03, 1, 0.95])
      plt.show()


