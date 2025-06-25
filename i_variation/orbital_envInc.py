import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

J2 = 1.08263e-3
R_EARTH = 6378.137

class OrbitalEnvInc(gym.Env):
    def __init__(self, timepenalty=0.05, masspenalty=0.5, log=True):
        super().__init__()
        self.time_penalty = timepenalty
        self.mass_penalty = masspenalty
        self.log = log
        if self.log:
            # Updated log directory for inclination
            self.log_dir = f"./{(int(timepenalty*100)):02}t_{masspenalty:02}m"
            os.makedirs(self.log_dir, exist_ok=True) # Ensure directory exists
            print(f"Logging case {self.log_dir}")

        # Constants
        self.mu = 398600.4418  # km^3/s^2, Earth's gravitational parameter
        self.T_max = 5e-6    # km/s^2, max thrust acceleration
        self.mass_initial = 500.0  # kg
        self.dt = 60.0       # seconds per step
        self.include_j2 = True

        self.success_counter = 0

        self.final_t = 30 # days

        self.use_rk = True   # Toggle RK integration

        self.episode_counter = 0 
        self.worker_id = os.getpid()
        if self.log:
            self.log_file = open(f"{self.log_dir}/check_inc_{self.worker_id}.txt", "w")

        # Observation space: scaled Keplerian elements + mass ratio (8 features)
        # Now includes inclination error, its rate of change, and fixed e/a errors
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)

        # Action space: thrust direction Fw
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Target orbit (fixed 'a' and 'e', randomized 'i')
        self.goal = np.array([8500, 0.01, 47.5*np.pi/180, 120*np.pi/180, 45.0*np.pi/180]) 

        # Initial orbit (fixed 'a' and 'e', randomized 'i')
        self.state_0 = np.array([8500, 0.01, 47.5*np.pi/180, 120*np.pi/180, 45.0*np.pi/180, 0.0, self.mass_initial])

        self.k_parameters = np.array([2,1.0,1.0,1.0,1.0]) # Only a matters for semi-latus rectum definition

        self.last_throttle = 0
        self.last_action = np.array([0.0, 0.0, 0.0]) # Stores the last Fr, Fs, Fw

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
        
        # Fixed semi-major axis and eccentricity
        a0 = 8500.0
        e0 = 0.01 
        
        # Random initial and target inclination within the specified range (45 to 50 degrees)
        i0 = 45 * np.pi / 180
        goal_i = np.random.uniform(40.0, 50.0) * np.pi / 180

        # Other orbital elements fixed
        raan0 = 120.0 * np.pi / 180
        argp0 = 45.0 * np.pi / 180
        v0 = 0.0 # True anomaly

        initial_state = np.array([
            a0, e0, i0, raan0, argp0, v0, self.mass_initial
        ])
        goal = np.array([
            a0, e0, goal_i, raan0, argp0
        ])

        self.goal = goal
        self.state_0 = initial_state
        self.state = self.state_0.copy()
        self.trajectory = [self.state.copy()]
        self.actions = []
        self.last_throttle = 0.0 # Reset throttle
        self.last_action = np.array([0.0, 0.0, 0.0]) # Reset last action (Fr, Fs, Fw)
        if self.log:
            self.log_file.flush()

        return self._get_obs(), {}

    def step(self, action):
        fw = float(np.clip(action[0], -1.0, 1.0))

        throttle = np.linalg.norm([0, 0, fw]) # Throttle from 3 components

        self.last_throttle = np.clip(throttle, 0.0, 1.0) # Ensure throttle is 0-1
        self.last_action = np.array([0, 0, fw]) 

        a_rsw = np.array([0, 0, fw]) * self.T_max
        
        # Log original direction and throttle
        self.actions.append(np.array([0, 0, fw, throttle]))

        if self.use_rk:
            self.state = self._rk_integrate(self.state, a_rsw)
        else:
            self.state = self._propagate(self.state, a_rsw)

        # Enforce e >= 0 and i within valid range [0, pi] after propagation
        if self.state[1] < 0: self.state[1] = 0.0
        self.state[2] = np.clip(self.state[2], 0.0, np.pi) 

        self.trajectory.append(self.state.copy())
        reward = self._compute_reward(self.state)

        # --- Termination conditions ---
        max_steps = int((self.final_t*24*60*60)/self.dt) # final_t days
        done = False
        # If mass runs out or max time exceeded
        if self.state[-1] <= 0 or len(self.trajectory) >= max_steps or abs(self.state[2] - self.goal[2]) > np.radians(10):
            done = True
        
        # Check if the "in goal" criteria have been met for enough steps
        inc_in_tolerance = abs(self.state[2] - self.goal[2]) < (0.5 * np.pi/180) # Tolerance for inclination (0.5 degrees)

        if inc_in_tolerance:
            self.steps_in_goal += 1
            if self.steps_in_goal >= 100: # Reached goal for enough time
                done = True
        else:
            self.steps_in_goal = 0 # Reset if not in goal

        # Hard penalty for going crazy in semi-major axis or inclination (e.g., > 1000km or > 20 degrees off from initial i)
        if abs(self.state[0] - self.goal[0]) > 1000 or \
           abs(self.state[2] - self.goal[2]) > (20 * np.pi/180): # If inclination goes 20 degrees off target
            done = True

        if len(self.trajectory) % 200 == 0:
            if self.log:

                p = self.state[0] * (1 - self.state[1]**2)
                r = p / (1 + self.state[1] * np.cos(self.state[5]))
                u = self.state[4] + self.state[5]

                term = r * np.cos(u)

                self.log_file.write(f"[t={len(self.trajectory)*self.dt}s], i={self.state[2]*180/np.pi:.2f}deg, goal_i={self.goal[2]*180/np.pi:.2f}deg, Fw={fw:.4f}, delta_i = {self.trajectory[-1][2]-self.trajectory[-2][2]}, term = {term}, reward={reward:.3f}\n")

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
        # Inclination error, scaled to a reasonable range (e.g., max expected error of 5 degrees)
        norm_i_error = (i - goal_i) / (5 * np.pi/180) 
        norm_goal_i = (goal_i - (45*np.pi/180)) / (5*np.pi/180) # Normalize target inclination within 45-50deg range
        norm_i = (i- (45*np.pi/180)) / (5*np.pi/180) # Normalize target inclination within 45-50deg range

        # Normalized fuel remaining (-1 to 1)
        norm_fuel_remaining = 2.0 * (m / self.mass_initial) - 1.0

        # Calculate cosine from v,u
        cos_v = np.cos(v); sin_v = np.sin(v)
        cos_u = np.cos(v+argp); sin_u = np.sin(v+argp)

        # Sign indicators for corrective behavior
        sign_i_error = np.sign(i - goal_i)

        # Direct derivative of inclination to show how i is changing per step
        delta_i = 0.0
        if len(self.trajectory) > 1:
            prev_i = self.trajectory[-2][2]
            delta_i = (i - prev_i) / np.radians(0.4) # Scale delta_i for observation
            delta_i = np.clip(delta_i, -1.0, 1.0) # Clip to keep within bounds

        return np.array([
            norm_i_error,       # 1. Normalized inclination error
            norm_fuel_remaining,# 2. Normalized fuel remaining
            delta_i,            # 3. Rate of change of inclination
            norm_goal_i,        # 4. Normalized target inclination
            sin_u,              # 5. sin u
            cos_u,              # 6. cos u
            sin_v,              # 7. sin v
            cos_v,              # 8. cos v
            sign_i_error,       # 9. i_error sign
            norm_i              # 10. norm of i
        ], dtype=np.float32)

    def _compute_reward(self, state):

        a, e, i, raan, argp, v, m = state
        goal_a, goal_e, goal_i, goal_raan, goal_argp = self.goal

        step = len(self.trajectory)
        throttle = self.last_throttle

        abs_i_error = abs(goal_i - i)
        delta_i = goal_i -i

        # 1. Base penalty per step: ensures rewards are mostly negative unless at goal.
        # This constant negative reward is the default "cost" of existing.
        base_step_penalty_per_step = -50 # A small, consistent negative value
        reward = base_step_penalty_per_step

        # Define the ranges for importance
        i_importance_range = 20 * np.pi/180 # Max expected inclination error for normalization

        # Normalize inclination error
        i_normalized_for_gaussian = abs_i_error / i_importance_range

        # 2. Gaussian reward for proximity to goal inclination.
        # This provides a small positive boost, peaking sharply near the goal.
        # Max reward is very small; it won't make the net reward positive unless very close.
        gaussian_std_for_normalized_error = 0.1 # Smaller std makes the peak sharper (rewards only when very close)
        max_gaussian_reward_per_parameter = 75.0 # Very small max value

        gaussian_component = max_gaussian_reward_per_parameter * np.exp(-(i_normalized_for_gaussian**2) / (2 * gaussian_std_for_normalized_error**2))
        reward += gaussian_component

        # 3. Fuel Penalty: Proportional to throttle usage.
        # A consistent negative contribution for using fuel.
        #reward -= self.mass_penalty * throttle * 0.1 # Reduced impact, but still penalizes fuel use

        # 4. Time Penalty: Consistent negative reward for each step, encouraging efficiency.
        # This penalty accumulates over time.
        reward -= self.time_penalty * step

        # 5. Thrust Direction Guidance (Small nudges)
        # These terms provide slight positive/negative feedback, but are generally too small
        # to overcome the base negative reward unless the agent is highly efficient.
        desired_delta_i_sign = np.sign(goal_i - i) # Desired direction to move
        if len(self.trajectory) > 1:
            prev_i = self.trajectory[-2][2]
            actual_delta_i = i - prev_i # Actual change in inclination

            desired_delta_i_sign = np.sign(goal_i - i) # Desired direction to move
            i_tolerance_strict = 0.05 * np.pi/180 # Strict tolerance for being near goal

            # Positive reinforcement for moving in the correct direction (only if outside strict tolerance)
            if abs_i_error >= i_tolerance_strict and desired_delta_i_sign * actual_delta_i > 0:
                reward += 20
                reward += 500.0 * throttle * abs(actual_delta_i) # Small positive reward for correct movement

            # Negative reinforcement for moving in the wrong direction (if thrusting)
            # Scaling `actual_delta_i` by 1e5 makes this penalty sensitive to the magnitude of the wrong move.
            if throttle > 0.01 and desired_delta_i_sign * actual_delta_i < 0:
                reward -= 30
                reward -= 500.0 * throttle * abs(actual_delta_i)  # Penalty for incorrect movement

        # 6. Penalty for overshooting the target inclination (applies during trajectory).
        # This discourages the agent from excessively passing the target.
        if (i > goal_i and desired_delta_i_sign < 0) or \
           (i < goal_i and desired_delta_i_sign > 0):
            overshoot_amount = abs(i - goal_i)
            reward -= 500.0 * overshoot_amount # Increased penalty for larger overshoot

        if (i > goal_i and delta_i < 0) or (i < goal_i and delta_i > 0):
            reward += 50 * throttle  # encourage recovery

        if abs_i_error < np.radians(0.2) and throttle < 0.01:
            reward += 50  # boost for staying near and idle

        if self.steps_in_goal >= 20:
            reward += 200  # bonus for "settled"

        # 7. Terminal Rewards/Penalties (Main drivers for episode outcome)
        # These large rewards/penalties will dominate at the end of the episode.
        max_steps = int((self.final_t*24*60*60)/self.dt)
        done = False # Local 'done' check (matches the step function's termination logic)

        if self.state[-1] <= 0 or len(self.trajectory) >= max_steps:
            done = True
        if abs(self.state[2] - self.goal[2]) > np.radians(10): # Large deviation in i
            done = True

        if done:
            if self.steps_in_goal >= 100: # Actual success condition (reached goal for enough time)
                reward += 1000.0 # High positive reward for success
                # Add small penalties for total time and fuel used even upon success
                reward -= self.time_penalty * len(self.trajectory) / 1000.0
                reward -= self.mass_penalty * (self.mass_initial - m) / self.mass_initial * 100 # Penalize remaining fuel mass at end
            else:
                # Failure penalty: always strongly negative if goal not achieved.
                final_error_normalized = abs_i_error / (10 * np.pi/180) # Normalized error (0 to 1)
                reward -= 50.0 # Flat penalty for failure
                reward -= 100.0 * final_error_normalized # Penalty proportional to final error


        return reward

    def _gauss_rhs(self, t, state, a_rsw):
        a, e, i, raan, argp, v, m = state

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
        # Handle division by sin(i) for raan and argp if i is near 0 or pi
        if np.abs(np.sin(i)) < 1e-6:
            dwaan = 0.0
            dargp_inclination_term = 0.0
        else:
            dwaan = (r * np.sin(u) / (n * a**2 * x * np.sin(i))) * Fw
            dargp_inclination_term = (r * np.cos(i) * np.sin(u) / (n * a**2 * x * np.sin(i))) * Fw

        dargp = (-x * np.cos(v) / (n * a * e_safe)) * Fr + (p * np.sin(v) / (e_safe * h)) * (1 + 1 / (1 + e_safe * np.cos(v))) * Fs \
              - dargp_inclination_term # This term was already correct for Fw contribution
        
        dv = h / r**2 + (1 / (e_safe * h)) * (p * np.cos(v) * Fr - (p + r) * np.sin(v) * Fs)
        dm = -m * thrust_acc / (Isp * g0)

        # J2 perturbations
        if self.include_j2:
            factor = (3/2) * J2 * (R_EARTH / p)**2 * n
            dwaan += -factor * np.cos(i)
            dargp += factor * (0.5 * (5 * np.cos(i)**2 - 1))

        return [da, de, di, dwaan, dargp, dv, dm]

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
            [sin_w*sin_i,                   cos_w*sin_i,                   cos_i]
        ])

        r_eci = R @ r_pf
        v_eci = R @ v_pf
        return r_eci, v_eci

    def plot_trajectory(self):
        traj = np.array(self.trajectory)
        time = np.arange(len(traj)) * self.dt / 86400

        fig, axs = plt.subplots(3, 2, figsize=(13, 10))
        labels = ['a (km)', 'e', 'i (º)', 'Ω (º)', 'ω (º)', 'mass (kg)']
        linestyle = '--'  

        for i, ax in enumerate(axs.flat[:-1]):
            # Convert angles to degrees for plotting
            y = traj[:, i] * 180/np.pi if i in [2, 3, 4] else traj[:, i]
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
        traj = np.array(self.trajectory)
        r_eci_all = []
        for state in traj:
            r_eci, _ = self._keplerian_to_cartesian(*state[:6])
            r_eci_all.append(r_eci)
        r_eci_all = np.array(r_eci_all)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(r_eci_all[:,0], r_eci_all[:,1], r_eci_all[:,2],
                label='Trajectory', color='#A7C4A0', linewidth=2)

        ax.scatter(*r_eci_all[0], color='#C2EABA', s=50, label='Start')
        ax.scatter(*r_eci_all[-1], color='#8F8389', s=50, label='End')

        u, v = np.mgrid[0:2*np.pi:150j, 0:np.pi:150j]
        x = R_EARTH * np.cos(u) * np.sin(v)
        y = R_EARTH * np.sin(u) * np.sin(v)
        z = R_EARTH * np.cos(v)
        ax.plot_surface(x, y, z, color='steelblue', alpha=0.2, zorder=0)

        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')

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
            buffer = 1000  
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