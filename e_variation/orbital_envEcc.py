import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

J2 = 1.08263e-3
R_EARTH = 6378.137

class OrbitalEnvEcc(gym.Env):
    def __init__(self, timepenalty=0.05, masspenalty=0.5, log=True):
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
        self.include_j2 = False

        self.success_counter = 0

        self.final_t = 10

        self.use_rk = True  # Toggle RK integration

        self.episode_counter = 0 # 3 times initiated by checker
        self.worker_id = os.getpid()
        if self.log:
          self.log_file = open(f"{self.log_dir}/check_ecc_{self.worker_id}.txt", "w")
        # Observation space: scaled Keplerian elements + mass ratio
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)

        # Action space: thrust direction S
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

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
      
      e0 = np.random.uniform(0.0, 0.25) # Initial e from 0.0 to 0.25
      goal_e = np.random.uniform(0.0, 0.25) # Target e from 0.0 to 0.25

      initial_state = np.array([
          8500, e0, 51*np.pi/180, 120*np.pi/180, 45*np.pi/180, 0.0, self.mass_initial
      ])
      goal = np.array([
          8500, goal_e, 51*np.pi/180, 120*np.pi/180, 45*np.pi/180
      ])

      self.goal = goal
      self.state_0 = initial_state
      self.state = self.state_0.copy()
      self.trajectory = [self.state.copy()]
      self.actions = []
      self.last_throttle = 0.0 # Reset throttle
      self.last_action = np.array([0.0, 0.0, 0.0]) # Reset last action
      if self.log:
        self.log_file.flush()

      return self._get_obs(), {}


    def step(self, action):
        fr = float(np.clip(action[0], -1.0, 1.0))
        fs = float(np.clip(action[1], -1.0, 1.0))
        throttle = np.linalg.norm([fr, fs])

        self.last_throttle = np.clip(throttle, 0.0, 1.0) # Ensure throttle is 0-1
        self.last_action = np.array([fr, fs, 0.0]) 

        a_rsw = np.array([fr, fs, 0.0]) * self.T_max
        
        # Log original direction and throttle
        self.actions.append(np.array([fr, fs, 0.0, throttle]))

        if self.use_rk:
            self.state = self._rk_integrate(self.state, a_rsw)
        else:
            self.state = self._propagate(self.state, a_rsw)

        # Enforce e >= 0 after propagation
        if self.state[1] < 0:
            self.state[1] = 0.0 # Set eccentricity to 0 if it goes negative

        self.trajectory.append(self.state.copy())
        reward = self._compute_reward(self.state)

        # --- Termination conditions ---
        max_steps = int((self.final_t*24*60*60)/self.dt) # final_t days
        done = False
        # If mass runs out or max time exceeded
        if self.state[-1] <= 0 or len(self.trajectory) >= max_steps:
            done = True
        
        # Check if the "in goal" criteria have been met for enough steps
        ecc_in_tolerance = abs(self.state[1] - self.goal[1]) < 0.005 # Tolerance for eccentricity
        a_in_tolerance = abs(self.state[0] - self.goal[0]) < 50 # Tolerance for semi-major axis

        if ecc_in_tolerance and a_in_tolerance:
            self.steps_in_goal += 1
            if self.steps_in_goal >= 100: # Reached goal for enough time
                done = True
        else:
            self.steps_in_goal = 0 # Reset if not in goal

        # Hard penalty for going crazy
        if abs(self.state[0] - self.goal[0]) > 1000:
            done = True

        if len(self.trajectory) % 200 == 0:
            if self.log:
              self.log_file.write(f"[t={len(self.trajectory)*self.dt}s] a={self.state[0]:.0f}, e={self.state[1]:.4f}, goal_e={self.goal[1]:.4f}, Fr={fr:.4f}, Fs={fs:.4f}, reward={reward:.3f}\n")

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
        a, e, i, raan, argp, v, m = self.state
        goal_a, goal_e, goal_i, goal_raan, goal_argp = self.goal

        # Normalize relevant state variables and errors
        # Eccentricity error, scaled to a reasonable range (e.g., max expected error of 0.2)
        norm_e_error = (e - goal_e) / 0.2
        norm_goal_e = (goal_e - 0.125) / 0.125

        # Semi-major axis error, scaled. Assuming a max a_error of +/- 50 km
        norm_a_error = (a - goal_a) / 50.0

        # Normalized fuel remaining (0 to 1, or -1 to 1 if you prefer symmetrical)
        # Using 2 * (m / self.mass_initial) - 1 to get -1 to 1 for easier interpretation by policy
        norm_fuel_remaining = 2.0 * (m / self.mass_initial) - 1.0

        # Calculate sine/cosine from v
        sin_v = np.sin(v)
        cos_v = np.cos(v)

        # Sign indicators for corrective behavior
        sign_e_error = np.sign(e - goal_e)
        sign_a_error = np.sign(a - goal_a)

        # Direct derivative of eccentricity to show how e is changing per step
        # This gives immediate feedback on thrust effectiveness
        delta_e = 0.0
        if len(self.trajectory) > 1:
            # Calculate eccentricity derivative from previous state
            # This requires storing previous state correctly.
            # A more robust way might be to numerically estimate derivative or just use the current change.
            # For simplicity, let's use the actual change from the previous step.
            prev_e = self.trajectory[-2][1]
            delta_e = (e - prev_e) * 100 # Scale delta_e for observation
            delta_e = np.clip(delta_e, -0.5, 0.5) # Clip to keep within bounds

        # Consider adding angular errors, but let's start simpler for now.
        # If the goal includes RAAN/argp changes, these would be crucial.
        # For now, since the goal is fixed for i, RAAN, argp, these might not be the most important observations.

        return np.array([
            norm_e_error,        # 1. Normalized eccentricity error
            norm_a_error,        # 2. Normalized semi-major axis error (important for maintaining orbit)
            norm_fuel_remaining, # 3. Normalized fuel remaining
            delta_e,             # 4. Rate of change of eccentricity (highly important feedback)
            self.last_throttle,  # 5. The last throttle value used (could help prevent over-thrusting)
            norm_goal_e,         # 6. Normalized target eccentricity
            sin_v,               # 7. sin of v
            cos_v,               # 8. cos of v
            sign_e_error,        # 9. e_error sign
            sign_a_error         # 10. a_error sign
        ], dtype=np.float32)


    def _compute_reward(self, state):
        a, e, i, raan, argp, v, m = state
        goal_a, goal_e, goal_i, goal_raan, goal_argp = self.goal

        step = len(self.trajectory)
        throttle = self.last_throttle

        e_effective = np.sqrt(e**2 + 1e-6) # Calculate effective eccentricity for reward smoothing
        e_error = goal_e - e_effective # Use e_effective here
        abs_e_error = abs(e_error)
        a_error = abs(goal_a - a)

        base_step_penalty = -200.0
        reward = base_step_penalty

         # Define the ranges for importance as specified by you
        e_importance_range = 0.2
        a_importance_range = 50.0

        # Normalize errors based on their respective importance ranges
        e_normalized_for_gaussian = abs_e_error / e_importance_range
        a_normalized_for_gaussian = a_error / a_importance_range

        gaussian_std_for_normalized_error = 0.5

        max_gaussian_reward_per_parameter = 200 

        # 1. Primary Reward: Proximity to goal eccentricity (Gaussian)
        reward += max_gaussian_reward_per_parameter * np.exp(-(e_normalized_for_gaussian**2) / (2 * gaussian_std_for_normalized_error**2))

        # 2. Secondary Reward: Proximity to goal semi-major axis (Gaussian)
        reward += max_gaussian_reward_per_parameter * np.exp(-(a_normalized_for_gaussian**2) / (2 * gaussian_std_for_normalized_error**2))

        # 3. Fuel Penalty
        reward -= self.mass_penalty * throttle # Reduced immediate penalty

        # 4. Time Penalty (scaled, only penalize if not close to goal)
        time_penalty_deviation_score = (e_normalized_for_gaussian * 1.0) + (a_normalized_for_gaussian * 0.5)

        reward -= self.time_penalty * step/1000

        if abs_e_error > 0.01 or a_error > 50:
            reward -= self.time_penalty * (step / 1000.0) * time_penalty_deviation_score

        if abs_e_error < 0.02 and self.last_action[0] < 0.1:
            reward += 20
        
        if abs(a_error) < 20 and abs(self.last_action[1]) < 0.2:
            reward += 20
        
        # 5. Thrust Direction Bonus (based on observed delta_e)
        if len(self.trajectory) > 1:
            prev_e = self.trajectory[-2][1]
            actual_delta_e = e - prev_e
            
            desired_delta_e_sign = np.sign(goal_e - e) 
            e_tolerance_std = 0.005

            # Apply directional reward/penalty when outside the tight eccentricity tolerance
            if abs_e_error >= e_tolerance_std:
                # Threshold for meaningful change slightly to capture smaller movements
                if desired_delta_e_sign * actual_delta_e > 0.00001: # Reduced from 0.00005
                    reward += 200.0 # Small bonus for moving in the right direction
                # Threshold for wrong-way penalty slightly
                elif throttle > 0.1 and desired_delta_e_sign * actual_delta_e < -0.00001: # Reduced from -0.00005
                    reward -= 2000 * abs(actual_delta_e)


            if (e > goal_e and actual_delta_e > 0) or (e < goal_e and actual_delta_e < 0):
              overshoot_amount = abs(e - goal_e)
              """ if overshoot_amount > e_tolerance_std: # Only if significant overshoot """
              reward -= 1000 * overshoot_amount

        # 6. Success Bonus/Continuous Reward
        ecc_in_tolerance = abs_e_error < 0.01 
        a_in_tolerance = a_error < 50 

        if ecc_in_tolerance and a_in_tolerance:
            reward += 50 # Small continuous reward for being in the goal state
            self.steps_in_goal+=1
            if throttle < 0.01: 
                reward += 100 # Encourage stopping thrusting when arrived
            else:
                reward -= 5 # Small penalty for wasting fuel when at target

        # 7. Terminal Rewards/Penalties
        max_steps = int((self.final_t*24*60*60)/self.dt)
        
        # Add a linear penalty for significant 'a' deviation (e.g., beyond 100km)
        a_warning_threshold = 40.0 # If 'a' deviates by more than 40 km
        a_linear_penalty_strength = 5 # Tune this value

        if abs(a_error) > a_warning_threshold:
            # Penalize linearly for every km beyond the threshold
            reward -= a_linear_penalty_strength * (a_error - a_warning_threshold)

        done = False
        if self.state[-1] <= 0 or len(self.trajectory) >= max_steps:
            done = True
        if abs(self.state[0] - self.goal[0]) > 1000: # Large deviation in semi-major axis
            done = True

        if done:
            if self.steps_in_goal >= 100: # Actual success condition
                reward += 5000 
                # reward -= (len(self.trajectory) / max_steps) * 100 # Can add time penalty if successful but slow
            else:
                # Failure penalty, proportional to remaining error
                final_error = abs_e_error + a_error/100 
                if (goal_e == 0.0 and e < goal_e) or \
                   (goal_e == 0.2 and e > goal_e + 0.01):
                   final_error += 0.5 # Add extra penalty for overshoot at terminal
                reward -= 1000 * final_error + 200 # Base penalty + error penalty

        return reward


    def _gauss_rhs(self, t, state, a_rsw):
        a, e, i, raan, argp, v, m = state

        # It's important for circularization maneuvers.
        e_safe = max(e, 1e-6) # Use a small positive value if e is zero or near

        x = np.sqrt(1 - e_safe**2)
        p = a * (1 - e_safe**2)
        r = p / (1 + e_safe * np.cos(v))
        h = np.sqrt(self.mu * p)
        n = np.sqrt(self.mu / a**3)
        u = argp + v

        Isp = 3000  # seconds, for ion engine
        g0 = 9.80665 / 1000  # km/s²
        thrust_acc = np.linalg.norm(a_rsw)

        Fr, Fs, Fw = a_rsw

        # Gauss Equations
        da = (2 * e_safe * np.sin(v) / (n * x)) * Fr + (2 * a * x / (n * r)) * Fs
        de = (x * np.sin(v) / (n * a)) * Fr + (x / (n * a**2 * e_safe)) * ((a**2 * x**2 / r - r) * Fs)
        di = (r * np.cos(u) / (n * a**2 * x)) * Fw
        draan = (r * np.sin(u) / (n * a**2 * x * np.sin(i))) * Fw if np.abs(np.sin(i)) > 1e-6 else 0.0
        dargp = (-x * np.cos(v) / (n * a * e_safe)) * Fr + (p * np.sin(v) / (e_safe * h)) * (1 + 1 / (1 + e_safe * np.cos(v))) * Fs \
             - (r * np.cos(i) * np.sin(u) / (n * a**2 * x * np.sin(i))) * Fw if np.abs(np.sin(i)) > 1e-6 else 0.0
        dv = h / r**2 + (1 / (e_safe * h)) * (p * np.cos(v) * Fr - (p + r) * np.sin(v) * Fs)
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


