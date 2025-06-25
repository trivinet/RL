import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

J2 = 1.08263e-3
R_EARTH = 6378.137

class OrbitalEnvGTO(gym.Env):
    def __init__(self, timepenalty=0.05, masspenalty=0.5, log=True, log_dir=None):
        super().__init__()
        self.time_penalty = timepenalty
        self.mass_penalty = masspenalty
        self.log = log
        if self.log:
            if log_dir is None:
                print("Warning: log_dir not provided to OrbitalEnvGTO. Using default path in current directory.")
                
                default_log_subdir = f"{(int(timepenalty*100)):02}t_{masspenalty:02}m"
                os.makedirs(default_log_subdir, exist_ok=True)
                self.log_dir = default_log_subdir
            else:
                self.log_dir = log_dir

            self.worker_id = os.getpid() 
            log_filepath = os.path.join(self.log_dir, f"check_gto_{self.worker_id}.txt")
            self.log_file = open(log_filepath, "w")
            print(f"Logging case {self.log_dir.split(os.sep)[-1]}") 
        else:
            self.log_file = None 
        # Constants
        self.mu = 398600.4418  # km^3/s^2, Earth's gravitational parameter
        self.T_max = 5e-7  # km/s^2, max thrust acceleration
        self.mass_initial = 500.0  # kg
        self.dt = 60.0          # seconds per step
        self.include_j2 = False

        self.success_counter = 0

        self.final_t = 100

        self.use_rk = True  # Toggle RK integration

        self.episode_counter = 0 # 3 times initiated by checker
        self.worker_id = os.getpid()
        if self.log:
            self.log_file = open(f"{self.log_dir}/check_gto_{self.worker_id}.txt", "w")
        
        # Observation space: scaled Keplerian elements + mass ratio + last action
        # 1. Normalized eccentricity error
        # 2. Normalized semi-major axis error
        # 3. Normalized inclination error
        # 4. Normalized fuel remaining
        # 5. Rate of change of eccentricity
        # 6. Rate of change of semi-major axis
        # 7. Rate of change of inclination
        # 8. The last Fr value used
        # 9. The last Fs value used
        # 10. The last Fw value used
        # 11. Normalized target eccentricity
        # 12. Normalized target semi-major axis
        # 13. Normalized target inclination
        # 14. sin of true anomaly (v)
        # 15. cos of true anomaly (v)
        # 16. sign of e_error
        # 17. sign of a_error
        # 18. sign of i_error
        self.observation_space = spaces.Box(low=-1, high=1, shape=(18,), dtype=np.float32)

        # Action space: thrust directions (Fr, Fs, Fw)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Target orbit for reward: GEO (a ≈ 42164 km, e ≈ 0, i ≈ 0)
        self.goal = np.array([42164.0, 0.0, 0.0, 0.0, 0.0])  # [a, e, i, raan, argp]

        # Initial orbit: GTO (e.g., a ≈ 24361 km, e ≈ 0.72, i ≈ 0)
        self.state_0 = np.array([24361.0, 0.72, 28.5, 0.0, 0.0, 0.0, self.mass_initial]) # Added initial inclination

        # K parameters (not used in provided reward, kept for context if needed for future reward shaping)
        self.k_parameters = np.array([2,1.0,1.0,1.0,1.0]) # only a matters

        self.last_action = np.array([0.0, 0.0, 0.0]) # Reset last action to 3 components

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
        
        # Fixed initial GTO
        initial_state = np.array([24361.0, 0.72, np.radians(5.2), 0.0, 0.0, 0.0, self.mass_initial]) # Initial inclination for GTO
        goal = np.array([42164.0, 0.0, 0.0, 0.0, 0.0])


        self.goal = goal
        self.state_0 = initial_state
        self.state = self.state_0.copy()
        self.trajectory = [self.state.copy()]
        self.actions = []
        self.last_action = np.array([0.0, 0.0, 0.0]) # Reset last action
        if self.log:
            self.log_file.flush()

        return self._get_obs(), {}


    def step(self, action):
        fr = float(np.clip(action[0], -1.0, 1.0))
        fs = float(np.clip(action[1], -1.0, 1.0))
        fw = float(np.clip(action[2], -1.0, 1.0)) # New Fw component

        # Total throttle based on the magnitude of the thrust vector
        throttle = np.linalg.norm([fr, fs, fw]) 

        self.last_action = np.array([fr, fs, fw]) 

        a_rsw = np.array([fr, fs, fw]) * self.T_max
        
        # Log original direction and throttle
        self.actions.append(np.array([fr, fs, fw, throttle]))

        if self.use_rk:
            self.state = self._rk_integrate(self.state, a_rsw)
        else:
            self.state = self._propagate(self.state, a_rsw)

        # Enforce e >= 0 and i >= 0 after propagation
        if self.state[1] < 0:
            self.state[1] = 0.0 
        if self.state[2] < 0: # Inclination cannot be negative
            self.state[2] = 0.0 

        self.trajectory.append(self.state.copy())
        reward = self._compute_reward(self.state)

        # --- Termination conditions ---
        max_steps = int((self.final_t*24*60*60)/self.dt) # final_t days
        done = False
        # If mass runs out or max time exceeded
        if self.state[-1] <= 0 or len(self.trajectory) >= max_steps:
            done = True
        
        # Check if the "in goal" criteria have been met for enough steps
        ecc_in_tolerance = abs(self.state[1] - self.goal[1]) < 0.01 # Tighter tolerance for eccentricity
        a_in_tolerance = abs(self.state[0] - self.goal[0]) < 50 # Tolerance for semi-major axis
        i_in_tolerance = abs(self.state[2] - self.goal[2]) < 0.5 # Tolerance for inclination

        if ecc_in_tolerance and a_in_tolerance and i_in_tolerance:
            self.steps_in_goal += 1
            if self.steps_in_goal >= 100: # Reached goal for enough time
                done = True
        else:
            self.steps_in_goal = 0 # Reset if not in goal

        # Penalize large overshoots in 'a' or 'i'
        if self.state[0] > self.goal[0] + 10000:  
            done = True
            if self.log:
                self.log_file.write(f" TERMINATED EARLY due to semimajor axis overshoot (a = {self.state[0]:.1f})\n")


        if len(self.trajectory) % 200 == 0:
            if self.log:
                self.log_file.write(f"[t={len(self.trajectory)*self.dt}s] a={self.state[0]:.0f}, e={self.state[1]:.4f}, i={self.state[2]:.4f}, Fr={fr:.4f}, Fs={fs:.4f}, Fw={fw:.4f}, reward={reward:.3f}\n")

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
        # Eccentricity error, scaled to a reasonable range (e.g., max expected error of 0.72)
        norm_e_error = np.clip((e - goal_e) / self.state_0[1], -1.0, 1.0) 
        
        # Semi-major axis error, scaled by initial GTO a for context
        norm_a_error = np.clip((a - goal_a) / (self.goal[0] - self.state_0[0]), -1.0, 1.0)
        
        # Inclination error, scaled by initial GTO i
        norm_i_error = np.clip((i - goal_i) / self.state_0[2], -1.0, 1.0)


        # Normalized fuel remaining (0 to 1, or -1 to 1 if you prefer symmetrical)
        norm_fuel_remaining = 2.0 * (m / self.mass_initial) - 1.0

        # Calculate sine/cosine from v
        sin_v = np.sin(v)
        cos_v = np.cos(v)

        # Sign indicators for corrective behavior
        sign_e_error = np.sign(e - goal_e)
        sign_a_error = np.sign(a - goal_a)
        sign_i_error = np.sign(i - goal_i)


        # Direct derivative of orbital elements to show how they are changing per step
        delta_e = 0.0
        delta_a = 0.0
        delta_i = 0.0 # New delta_i

        if len(self.trajectory) > 1:
            prev_a = self.trajectory[-2][0]
            prev_e = self.trajectory[-2][1]
            prev_i = self.trajectory[-2][2] # Previous inclination

            delta_a = (a - prev_a) / 50.0  # Scale to range -1 to 1 approx
            delta_a = np.clip(delta_a, -1.0, 1.0)
            
            delta_e = (e - prev_e) * 100 # Scale delta_e for observation
            delta_e = np.clip(delta_e, -0.5, 0.5) # Clip to keep within bounds

            delta_i = (i - prev_i) * 10 # Scale delta_i
            delta_i = np.clip(delta_i, -0.5, 0.5) # Clip

        return np.array([
            norm_e_error,          # 1. Normalized eccentricity error
            norm_a_error,          # 2. Normalized semi-major axis error
            norm_i_error,          # 3. Normalized inclination error (NEW)
            norm_fuel_remaining,   # 4. Normalized fuel remaining
            delta_e,               # 5. Rate of change of eccentricity
            delta_a,               # 6. Rate of change of semi-major axis
            delta_i,               # 7. Rate of change of inclination (NEW)
            self.last_action[0],   # 8. Last Fr value used
            self.last_action[1],   # 9. Last Fs value used
            self.last_action[2],   # 10. Last Fw value used (NEW)
            (goal_e - 0.7) / 0.7,  # 11. Normalized target eccentricity (scaled for GTO context)
            (goal_a - 23000) / 20000, # 12. Normalized target semi-major axis (scaled for GTO context)
            (goal_i - 28.5) / 28.5, # 13. Normalized target inclination (scaled for GTO context) (NEW)
            sin_v,                 # 14. sin of v
            cos_v,                 # 15. cos of v
            sign_e_error,          # 16. e_error sign
            sign_a_error,          # 17. a_error sign
            sign_i_error           # 18. i_error sign (NEW)
        ], dtype=np.float32)


    def _compute_reward(self, state):
        a, e, i, *_ = state
        goal_a, goal_e, goal_i, *_ = self.goal

        reward = -1.0 * self.time_penalty
        
        # Extract individual thrust components
        fr = self.last_action[0]
        fs = self.last_action[1]
        fw = self.last_action[2]
        
        throttle = np.linalg.norm([fr, fs, fw]) # Total throttle

        time_frac = len(self.trajectory) / (self.final_t*24*60*60/self.dt)

        # Reward shaping for GTO-GEO maneuver
        # 1. Raise 'a' (semi-major axis)
        # Encourage tangential thrust (Fs) when 'a' is below target
        if a < goal_a - 100: # Significant distance from goal 'a'
            reward += 500.0 * np.clip(fs, 0, 1) # Positive Fs increases 'a'
        elif a < goal_a - 50: # Closer to goal 'a'
            reward += 1000.0 * np.clip(fs, 0, 1)

        # Penalize overshooting 'a'
        if a > goal_a:
            overshoot_a = a - goal_a
            reward -= 100.0 * overshoot_a  # Linear penalty for overshoot
            if overshoot_a > 200:
                reward -= 5000.0 # Heavier penalty for significant overshoot
            
            # Encourage negative Fs if overshot 'a' to bring it back down
            if fs < 0:
                reward += 500.0 * abs(fs)

        # 2. Turn 'e' (eccentricity) into 0
        # Encourage negative change in 'e' when 'e' is above target
        if e > goal_e + 0.01: # Still eccentric
            # Pro-grade tangential thrust (Fs) near periapsis reduces eccentricity
            # Retro-grade tangential thrust (Fs) near apoapsis reduces eccentricity
            # The model needs to learn WHEN to apply Fs and Fr for optimal e reduction.
            # For simplicity, we can encourage actions that lead to reduction.
            # More complex shaping might involve true anomaly (v)
            if len(self.trajectory) > 1:
                prev_e = self.trajectory[-2][1]
                delta_e = e - prev_e
                if delta_e < 0:
                    reward += 3000.0 * np.clip(-delta_e * 10, 0, 1) # Reward decreasing e

        # Penalize increasing 'e'
        if len(self.trajectory) > 1:
            prev_e = self.trajectory[-2][1]
            delta_e = e - prev_e
            if delta_e > 0:
                reward -= 6000.0 * np.clip(delta_e * 10, 0, 1)

        # 3. Turn 'i' (inclination) into 0
        # Encourage Fw (out-of-plane) thrust when 'i' is above target
        if i > goal_i + 0.1: # Significant inclination
            reward += 500.0 * np.clip(abs(fw), 0, 1) # Both positive/negative Fw reduces inclination depending on argument of latitude
            
            # Reward for reducing inclination
            if len(self.trajectory) > 1:
                prev_i = self.trajectory[-2][2]
                delta_i = i - prev_i
                if delta_i < 0:
                    reward += 3000.0 * np.clip(-delta_i * 10, 0, 1) # Reward decreasing i

        # Penalize increasing 'i'
        if len(self.trajectory) > 1:
            prev_i = self.trajectory[-2][2]
            delta_i = i - prev_i
            if delta_i > 0:
                reward -= 6000.0 * np.clip(delta_i * 10, 0, 1)


        # Proximity bonuses: Closer to GEO target
        a_tol = 50.0
        e_tol = 0.01
        i_tol = 0.5

        # Reward for being close to target 'a'
        if abs(a - goal_a) < a_tol:
            reward += 300.0 * (1 - abs(a - goal_a) / a_tol) # Smooth reward as it gets closer
        
        # Reward for being close to target 'e'
        if abs(e - goal_e) < e_tol:
            reward += 300.0 * (1 - abs(e - goal_e) / e_tol)
        
        # Reward for being close to target 'i'
        if abs(i - goal_i) < i_tol:
            reward += 300.0 * (1 - abs(i - goal_i) / i_tol)

        # High reward for achieving all goals simultaneously
        if abs(a - goal_a) < a_tol and abs(e - goal_e) < e_tol and abs(i - goal_i) < i_tol:
            reward += 5000.0 
            if throttle < 0.1: # Bonus for minimizing thrust once target is reached
                reward += 2000.0 
            else:
                reward -= 500.0 # Penalize fuel waste if still thrusting at target

        # Penalize high throttle when close to all goals to encourage fuel efficiency
        if abs(a - goal_a) < a_tol * 2 and abs(e - goal_e) < e_tol * 2 and abs(i - goal_i) < i_tol * 2:
            reward -= 100.0 * throttle # Slight penalty for any thrust near target

        # Always apply fuel cost
        reward -= self.mass_penalty * throttle

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

    def plot_orbit_3d(self, title='3D Orbit with Earth'):
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

        ax.set_title(title)
        ax.legend()

        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]))

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


