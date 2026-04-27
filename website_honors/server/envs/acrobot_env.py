import gymnasium as gym
import numpy as np
import time


class WebAcrobot:
    def __init__(self):
        self.env = gym.make("Acrobot-v1", render_mode="rgb_array")
        self.last_obs, _ = self.env.reset()
        self.success = False
        self.start_time = None

        # arm lengths (used for tip calculations)
        self.L1 = 1.0
        self.L2 = 1.0

    def reset(self):
        obs, _ = self.env.reset()
        self.last_obs = obs
        self.success = False
        self.start_time = time.time()
        return obs

    def step(self, action: int):
        obs, reward, terminated, truncated, _ = self.env.step(action)

        self.last_obs = obs
        done = terminated or truncated

        if terminated:
            self.success = True

        return obs, reward, done

    def render(self):
        frame = self.env.render()

        if frame is None:
            frame = np.zeros((400, 600, 3), dtype=np.uint8)

        return frame

    def close(self):
        self.env.close()

    def get_state(self):
        """
        Returns the state in the format expected by the frontend:
        [theta1, theta2, thetaDot1, thetaDot2]
        """

        cos1, sin1, cos2, sin2, thetaDot1, thetaDot2 = self.last_obs

        theta1 = np.arctan2(sin1, cos1)
        theta2 = np.arctan2(sin2, cos2)

        return [float(theta1), float(theta2), float(thetaDot1), float(thetaDot2)]

    def get_tip_position(self):
        """
        Returns the (x,y) position of the tip of the lower arm.
        """

        cos1, sin1, cos2, sin2, _, _ = self.last_obs

        theta1 = np.arctan2(sin1, cos1)
        theta2 = np.arctan2(sin2, cos2)

        x_tip = self.L1 * np.sin(theta1) + self.L2 * np.sin(theta1 + theta2)
        y_tip = -(self.L1 * np.cos(theta1) + self.L2 * np.cos(theta1 + theta2))

        return float(x_tip), float(y_tip)
