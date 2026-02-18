import gymnasium as gym
import numpy as np
import time

class WebMountainCar:
    def __init__(self):
        self.env = gym.make("MountainCar-v0", render_mode="rgb_array")
        self.last_obs, _ = self.env.reset()

        self.start_time = time.time()
        self.lap_times = []
        self.round = 1

        # training system
        self.training_mode = False
        self.training_goal = 0.5  # default = real flag

    # -------- RESET --------
    def reset(self, training_mode=False, goal_x=0.5):
        self.training_mode = training_mode
        self.training_goal = goal_x

        self.last_obs, _ = self.env.reset()
        self.start_time = time.time()
        return self.last_obs

    # -------- STEP --------
    def step(self, action: int):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        self.last_obs = obs

        position = float(obs[0])

        # NORMAL GAME
        if not self.training_mode:
            done = terminated or truncated
            success = terminated

        # TRAINING MODE (CUSTOM GOAL)
        else:
            success = position >= self.training_goal
            done = success or truncated

        # lap timer (only when success)
        if success:
            elapsed = time.time() - self.start_time
            self.lap_times.append(round(elapsed, 2))
            self.round += 1
            self.start_time = time.time()

        return obs, reward, bool(done), bool(success)

    # -------- RENDER --------
    def render(self):
        frame = self.env.render()
        if frame is None:
            frame = np.zeros((400, 600, 3), dtype=np.uint8)
        return frame

    def close(self):
        self.env.close()
