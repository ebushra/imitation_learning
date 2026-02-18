import json
import os
from datetime import datetime

class HumanDataLogger:
    def __init__(self, game_name):
        self.game = game_name
        self.episode = 0
        self.step = 0
        self.episode_data = []
        self.start_time = None

        os.makedirs("human_data", exist_ok=True)

    def new_episode(self):
        if self.episode_data:
            self.save_episode()

        self.episode += 1
        self.step = 0
        self.episode_data = []
        self.start_time = datetime.now()

    def log_step(self, state, action, reward, done, elapsed):
        self.step += 1

        entry = {
            "game": self.game,
            "episode": self.episode,
            "step": self.step,
            "state": list(map(float, state)),
            "action": int(action),
            "reward": float(reward),
            "done": bool(done),
            "time": float(elapsed)
        }

        self.episode_data.append(entry)

        if done:
            self.save_episode()

    def save_episode(self):
        filename = f"human_data/{self.game}_episode_{self.episode}.json"
        with open(filename, "w") as f:
            json.dump(self.episode_data, f, indent=2)
