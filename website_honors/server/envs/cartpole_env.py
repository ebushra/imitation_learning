import gymnasium as gym
from gymnasium import spaces
import numpy as np

class WebCartPole:

    def __init__(self, render_mode="rgb_array"):
        self.base = gym.make("CartPole-v1", render_mode=render_mode)
        self.base = self.base.unwrapped

        self.action_space = spaces.Discrete(3)
        self.observation_space = self.base.observation_space

        self.total_reward = 0
        self.last_frame = None

        # IMPORTANT
        self.training_mode = False

    # -------- RESET --------
    def reset(self, training=False, **kwargs):
        self.total_reward = 0

        # turn training mode ON/OFF
        self.training_mode = training

        obs, _ = self.base.reset(**kwargs)
        self.last_frame = self.base.render()

        return obs

    # -------- STEP --------
    def step(self, action):

        # ---- get current state ----
        x, x_dot, theta, theta_dot = self.base.state

        gravity = self.base.gravity
        masscart = self.base.masscart
        masspole = self.base.masspole
        total_mass = masscart + masspole
        length = self.base.length
        polemass_length = masspole * length
        tau = self.base.tau
        force_mag = self.base.force_mag

        # ---- action mapping ----
        if action == 0:
            force = 0.0
        elif action == 1:
            force = force_mag
        elif action == 2:
            force = -force_mag
        else:
            raise ValueError("Invalid action")

        # ---- physics update ----
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass

        thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0/3.0 - masspole * costheta**2 / total_mass)
        )

        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        # Euler integration
        x = x + tau * x_dot
        x_dot = x_dot + tau * xacc
        theta = theta + tau * theta_dot
        theta_dot = theta_dot + tau * thetaacc

        self.base.state = (x, x_dot, theta, theta_dot)

        # ---------------- TERMINATION RULES ----------------

        # cart leaving screen ALWAYS fails
        out_of_bounds = (
            x < -self.base.x_threshold or
            x > self.base.x_threshold
        )

        # NORMAL MODE → 12° rule
        if not self.training_mode:
            angle_fail = abs(theta) > self.base.theta_threshold_radians
            terminated = out_of_bounds or angle_fail

        # TRAINING MODE → IGNORE 12° RULE
        else:
            terminated = out_of_bounds   # ONLY cart matters

        # time limit
        self.total_reward += 1
        truncated = self.total_reward >= 250

        reward = 1.0
        obs = np.array(self.base.state, dtype=np.float32)

        self.last_frame = self.base.render()

        return obs, reward, terminated, truncated, {}

    def render(self):
        return self.last_frame

    def close(self):
        self.base.close()

    def get_state(self):
        return self.base.state
