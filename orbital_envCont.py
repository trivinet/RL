import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

J2 = 1.08263e-3
R_EARTH = 6378.137

class OrbitalEnvCont(gym.Env):
    def __init__(self, timepenalty=0.05, masspenalty=0, log=True):
        super().__init__()
        self.time_penalty = timepenalty
        self.mass_penalty = masspenalty
        self.log = log
        if self.log:
          self.log_dir = f"{(int(timepenalty*100)):02}t_{masspenalty:02}m"
          print(f"Logging case {self.log_dir}")
        # Constants
        self.mu = 398600.4418  # km^3/s^2, Earth's gravitational parameter
        self.T_max = 5e-6   # km/s^2, max thrust acceleration
        self.mass_initial = 500.0  # kg
        self.dt = 60.0         # seconds per step
        self.include_j2 = True

        self.success_counter = 0

        self.final_t = 5

        self.use_rk = True  # Toggle RK integration

        self.episode_counter = 0 # 3 times initiated by checker
        self.worker_id = os.getpid()
        if self.log:
          self.log_file = open(f"{self.log_dir}/check_{self.worker_id}.txt", "w")
        # Observation space: scaled Keplerian elements + mass ratio
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Action space: thrust direction S
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

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
            
    def close(self):
      if self.log:
        if hasattr(self, "log_file"):
          self.log_file.close()
           
    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self, initial_state=None, goal=None, seed=None, options=None):
      super().reset(seed=seed)
      self.steps_in_goal = 0
      self.episode_counter += 1
      if self.log:
        self.log_file.write("\n ------------------------------------------------- \n")
      
      if self.episode_counter+1 % 2 == 0:
        # GOAL: increase a
        a0 = np.random.uniform(7500, 7700)
        goal_a = a0 + np.random.uniform(200, 400)
      else:
        # GOAL: decrease a
        a0 = np.random.uniform(10300, 10500)
        goal_a = a0 - np.random.uniform(2500, 2700)
      
      if initial_state == None:
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
      if self.log:
        self.log_file.flush()

      return self._get_obs(), {}


    def step(self, action):
        fs = np.clip(action[0], -1.0, 1.0)
        throttle = abs(fs)  # Use magnitude for throttle

        self.last_throttle = fs
        self.last_action = np.array([0.0, fs, 0.0]) 

        a_rsw = np.array([0.0, fs, 0.0]) * self.T_max
        
        # Log original direction and throttle
        self.actions.append(np.array([0.0, fs, 0.0, throttle]))

        if self.use_rk:
            self.state = self._rk_integrate(self.state, a_rsw)
        else:
            self.state = self._propagate(self.state, a_rsw)

        self.trajectory.append(self.state.copy())
        reward = self._compute_reward(self.state)

        # --- Termination conditions ---
        max_steps = int((self.final_t*24*60*60)/self.dt) # 10 days
        state_diff = np.linalg.norm(self.state[:5] - self.goal)

        """ state_diff < 1e-3                     # Reached goal """
        done = (
          self.state[-1] <= 0                   # Ran out of fuel
          or len(self.trajectory) >= max_steps  # Max time exceeded
          or self.steps_in_goal >= 100          # Reached goal for enough time
        )

        if abs(self.state[0] - self.goal[0]) > 3200:
          done = True
          reward -= 100  # heavy penalty for "going crazy"

        if self.steps_in_goal >= 30:
          reward += 20

        if done and self.steps_in_goal >= 100:
          reward += 200

        if len(self.trajectory) % 200 == 0:
          if self.log:
            self.log_file.write(f"[t={len(self.trajectory)*self.dt}s] a={self.state[0]:.2f}, goal={self.goal[0]:.2f}, Fs={fs:.4f}, reward={reward:.3f}\n")

        if done and self.steps_in_goal >= 100:
          self.success_counter += 1
          if self.log:
            self.log_file.write(f" SUCCESS #{self.success_counter} of #{self.episode_counter} episodes\n")
        elif done:
          if self.log:
            self.log_file.write(f" FAILURE #{self.episode_counter - self.success_counter} of #{self.episode_counter} episodes\n")
        
        if self.log:
          self.log_file.flush()

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
      a, _, _, _, _, _, m = self.state
      goal_a = self.goal[0]
      fuel_obs = 1.0 - 2.0 * (m / self.mass_initial)  # Maps [500→0] kg to [–1 → +1]

      return np.array([
        (a - goal_a) / 1000,
        fuel_obs
      ], dtype=np.float32)


    def _compute_reward(self, state):
      a, _, _, _, _, _, m = state
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
      reward -= self.time_penalty * step

      # Option A: penalize fuel linearly
      reward -= self.mass_penalty * abs(fs)

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
      
      if a_error < 3 and abs(fs) < 0.01:
          reward += 25
      elif a_error < 3:
          reward -= 10  # penalize pushing when very close

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
      time = np.arange(len(traj)) * self.dt / 86400

      fig, axs = plt.subplots(3, 2, figsize=(13, 10))
      labels = ['a (km)', 'e', 'i (º)', 'Ω (º)', 'ω (º)', 'mass (kg)']
      linestyle = '--'  # dashed for elegance

      for i, ax in enumerate(axs.flat[:-1]):
          y = traj[:, i] * 180/np.pi if i > 1 and i < 5 else traj[:, i]
          ax.plot(time, y, linestyle=linestyle, color='black', linewidth=1)
          ax.set_ylabel(labels[i])
          ax.set_xlabel('Time (days)')
          ax.grid(True, linestyle=':', color='gray', alpha=0.6)

      axs[2, 1].plot(time, traj[:, -1], linestyle=linestyle, color='black', linewidth=1)
      axs[2, 1].set_ylabel('mass (kg)')
      axs[2, 1].set_xlabel('Time (days)')
      axs[2, 1].grid(True, linestyle=':', color='gray', alpha=0.6)

      plt.tight_layout()
      plt.show()

    def plot_orbit_3d(self, title='3D Orbit with Earth', zoom=False):
      # Extract orbital trajectory and convert to ECI positions
      traj = np.array(self.trajectory)
      r_eci_all = []
      for state in traj:
          r_eci, _ = self._keplerian_to_cartesian(*state[:6])
          r_eci_all.append(r_eci)
      r_eci_all = np.array(r_eci_all)

      fig = plt.figure(figsize=(10, 8))
      ax = fig.add_subplot(111, projection='3d')

      # Plot orbit trajectory
      ax.plot(r_eci_all[:,0], r_eci_all[:,1], r_eci_all[:,2],
              label='Trajectory', color='#A7C4A0', linewidth=2)

      # Mark start and end points
      ax.scatter(*r_eci_all[0], color='#C2EABA', s=50, label='Start')
      ax.scatter(*r_eci_all[-1], color='#8F8389', s=50, label='End')

      # Draw Earth
      u, v = np.mgrid[0:2*np.pi:150j, 0:np.pi:150j]
      x = R_EARTH * np.cos(u) * np.sin(v)
      y = R_EARTH * np.sin(u) * np.sin(v)
      z = R_EARTH * np.cos(v)
      ax.plot_surface(x, y, z, color='steelblue', alpha=0.2, zorder=0)

      # Axis setup
      ax.set_xlabel('X (km)')
      ax.set_ylabel('Y (km)')
      ax.set_zlabel('Z (km)')
      # Fix aspect ratio by setting equal axis limits
      max_range = np.array([
          r_eci_all[:,0].max() - r_eci_all[:,0].min(),
          r_eci_all[:,1].max() - r_eci_all[:,1].min(),
          r_eci_all[:,2].max() - r_eci_all[:,2].min()
      ]).max() / 2.0

      mid_x = (r_eci_all[:,0].max() + r_eci_all[:,0].min()) * 0.5
      mid_y = (r_eci_all[:,1].max() + r_eci_all[:,1].min()) * 0.5
      mid_z = (r_eci_all[:,2].max() + r_eci_all[:,2].min()) * 0.5

      ax.set_xlim(mid_x - max_range, mid_x + max_range)
      ax.set_ylim(mid_y - max_range, mid_y + max_range)
      ax.set_zlim(mid_z - max_range, mid_z + max_range)
      ax.set_title(title)
      ax.legend()

      if zoom:
          # Adjust limits to zoom on orbital arc
          buffer = 1000  # adjust if needed
          xmid = np.mean(r_eci_all[:,0])
          ymid = np.mean(r_eci_all[:,1])
          zmid = np.mean(r_eci_all[:,2])
          ax.set_xlim(xmid-buffer, xmid+buffer)
          ax.set_ylim(ymid-buffer, ymid+buffer)
          ax.set_zlim(zmid-buffer, zmid+buffer)

      plt.tight_layout()
      plt.show()

    def plot_actions(self):
  
      actions = np.array(self.actions)
      time = np.arange(len(actions)) * self.dt / 86400

      labels = ['F$_R$ (radial)', 'F$_S$ (along-track)', 'F$_W$ (cross-track)', 'Throttle']
      fig, axs = plt.subplots(2, 2, figsize=(12, 8))
      linestyle = '--'

      for i, ax in enumerate(axs.flat):
          ax.plot(time, actions[:, i], linestyle=linestyle, color='black', linewidth=1)
          ax.set_title(labels[i], fontsize=11)
          ax.set_ylim(-1.2 if i < 3 else -0.05, 1.2 if i < 3 else 1.05)
          ax.set_xlabel("Time (days)")
          ax.set_ylabel("Action value")
          ax.grid(True, linestyle=':', color='gray', alpha=0.6)

      plt.suptitle("Agent Actions Over Time", fontsize=14, fontweight='normal')
      plt.tight_layout(rect=[0, 0.03, 1, 0.95])
      plt.show()


