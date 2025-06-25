import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

J2 = 1.08263e-3
R_EARTH = 6378.137

class OrbitalEnvDiscr(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Constants
        self.mu = 398600.4418  # km^3/s^2, Earth's gravitational parameter
        self.T_max = 5e-6   # km/s^2, max thrust acceleration
        self.mass_initial = 500.0  # kg
        self.dt = 60.0         # seconds per step
        self.include_j2 = False

        self.success_counter = 0

        self.final_t = 5

        self.use_rk = True  # Toggle RK integration

        self.episode_counter = 0 # 3 times initiated by checker

        # Observation space: scaled Keplerian elements + mass ratio
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Action space: thrust direction S
        self.action_space = spaces.Discrete(3)

        # Target orbit for reward
        self.goal = np.array([7500, 0.02, 51*np.pi/180, 120*np.pi/180, 45.0*np.pi/180])  # [a, e, i, raan, argp]

        # Initial orbit
        self.state_0 = np.array([8000, 0.02, 51*np.pi/180, 120*np.pi/180, 45.0*np.pi/180, 0.0, self.mass_initial])

        # K parameters
        self.k_parameters = np.array([2,1.0,1.0,1.0,1.0]) # only a matters

        self.last_throttle = 0

        self.state = None  # [a, e, i, raan, argp, v, m]
        self.trajectory = []
        self.actions = []
            
    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self, initial_state=None, goal=None, seed=None, options=None):
      super().reset(seed=seed)
      self.steps_in_goal = 0
      self.episode_counter += 1

      """ # Curriculum: increase difficulty every 500 episodes
      max_delta = min(100 + 10 * (self.episode_counter // 500), 1000)  # max 1000 km

      if initial_state is None:
          a0 = np.random.uniform(7000, 9000)
          initial_state = np.array([
              a0, 0.02, 51*np.pi/180, 120*np.pi/180, 45*np.pi/180, 0.0, self.mass_initial
          ])
      self.state_0 = initial_state

      if goal is None:
          delta = np.random.uniform(-max_delta, max_delta)
          goal_a = np.clip(self.state_0[0] + delta, 7000, 9000)
          goal = np.array([
              goal_a, 0.02, 51*np.pi/180, 120*np.pi/180, 45*np.pi/180
          ]) """
      
      if self.episode_counter % 2 == 0:
        # GOAL: increase a
        a0 = np.random.uniform(7500, 7700)
        goal_a = a0 + np.random.uniform(200, 400)
      else:
        # GOAL: decrease a
        a0 = np.random.uniform(8300, 8500)
        goal_a = a0 - np.random.uniform(200, 400)

      initial_state = np.array([
          a0, 0.02, 51*np.pi/180, 120*np.pi/180, 45*np.pi/180, 0.0, self.mass_initial
      ])
      goal = np.array([
          goal_a, 0.02, 51*np.pi/180, 120*np.pi/180, 45*np.pi/180
      ])
      self.goal = goal
      self.state_0 = initial_state
      self.state = self.state_0.copy()
      self.trajectory = [self.state.copy()]
      self.actions = []

      return self._get_obs(), {}


    def step(self, action):
        # Map discrete action to Fs
        action_map = {0: -1.0, 1: 0.0, 2: 1.0}
        fs = action_map[int(action)]
        throttle = 0.2 # THROTTLE CAPPED TO 20%
        self.last_throttle = fs
        self.last_action = np.array([0.0, fs, 0.0])  # store full RSW for logs

        a_rsw = np.array([0.0, fs, 0.0]) * self.T_max * throttle
        # Log original direction and throttle
        self.actions.append(np.array([0.0, fs, 0.0, throttle]))

        if self.use_rk:
            self.state = self._rk_integrate(self.state, a_rsw)
        else:
            self.state = self._propagate(self.state, a_rsw)

        self.trajectory.append(self.state.copy())
        reward = self._compute_reward(self.state)

        # --- Termination conditions ---
        max_steps = int((10*24*60*60)/self.dt) # 10 days
        state_diff = np.linalg.norm(self.state[:5] - self.goal)

        """ state_diff < 1e-3                     # Reached goal """
        done = (
          self.state[-1] <= 0                   # Ran out of fuel
          or len(self.trajectory) >= max_steps  # Max time exceeded
          or self.steps_in_goal >= 100          # Reached goal for enough time
        )

        if abs(self.state[0] - self.goal[0]) > 1200:
          done = True
          reward -= 100  # heavy penalty for "going crazy"

        if len(self.trajectory) % 200 == 0:
          """ print(f"[t={len(self.trajectory)*self.dt}s] a={self.state[0]:.2f}, goal={self.goal[0]:.2f}, Fs={fs:.4f}, reward={reward:.3f}") """

        if done and self.steps_in_goal >= 100:
          self.success_counter += 1
          print(f"✅ Success #{self.success_counter} of #{self.episode_counter} episodes")
        elif done:
          print(f"⚠️ Failure #{self.episode_counter - self.success_counter} of #{self.episode_counter} episodes")


        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
      a, _, _, _, _, _, m = self.state
      goal_a = self.goal[0]

      return np.array([
        (a - goal_a) / 1000
      ], dtype=np.float32)
      """ return np.array([a/10000, ga / 10000, e, i/np.pi, raan/(2*np.pi), argp/(2*np.pi), m/self.mass_initial], dtype=np.float32) """


    def _compute_reward(self, state):
      a, *_ = state
      goal = self.goal
      fs = self.last_throttle
      step = len(self.trajectory)

      a_error = abs(goal[0] - a)
      reward = -a_error / 10

      # Early penalty: wrong direction in first few steps
      if step < 5 and fs != 0:
          if np.sign(fs) != np.sign(goal[0] - a):
              reward -= 20.0  # ❗️Big penalty to force learning

      # Penalize wasting time
      reward -= 0.01 * step

      # Movement shaping
      if step >= 2:
          prev_a = self.trajectory[-2][0]
          da = a - prev_a
          expected_direction = np.sign(goal[0] - a)

          if np.sign(da) == expected_direction:
              reward += 2.0
          else:
              reward -= 2.0

          if expected_direction != np.sign(da) and abs(goal[0] - a) > 200:
              reward -= 50

      # Action direction shaping
      if fs != 0:
          if np.sign(fs) == np.sign(goal[0] - a):
              reward += 1.0
          else:
              reward -= 2.0

      # Coasting near target
      if a_error < 10:
          self.steps_in_goal += 1
          if fs == 0.0:
              reward += 3.0  # Encourage stillness
          else:
              reward -= 2.0  # Penalize any push when close
      else:
         self.steps_in_goal = 0

      # Goal holding bonus
      if a_error < 5 and fs == 0.0:
          reward += 50
      else:
          if self.steps_in_goal > 0:
              reward -= 5

      return reward


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


