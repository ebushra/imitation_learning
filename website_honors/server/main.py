# website_honors/server/main_flask.py
import os
import time
import csv
import base64
from io import BytesIO

import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, send_from_directory, redirect

# Import your environment classes and utilities
from .envs.acrobot_env import WebAcrobot
from .envs.mountaincar_env import WebMountainCar
from .envs.cartpole_env import WebCartPole
from .utils.render import render_frame

# =====================
# Setup Flask app
# =====================
app = Flask(__name__, static_folder="../static")

# =====================
# Data directory for recording games
# =====================
DATA_DIR = "human_data"
os.makedirs(DATA_DIR, exist_ok=True)

# =====================
# GameRecorder
# =====================
class GameRecorder:
    def __init__(self, name):
        self.name = name
        self.episode = 0
        self.step = 0
        self.start_time = time.time()
        self.file = open(f"{DATA_DIR}/{name}.csv", "a", newline="")
        self.writer = csv.writer(self.file)
        if os.stat(f"{DATA_DIR}/{name}.csv").st_size == 0:
            self.writer.writerow([
                "timestamp","episode","step","elapsed",
                "state","action","reward","done","success"
            ])

    def new_episode(self):
        self.episode += 1
        self.step = 0
        self.start_time = time.time()

    def log(self, state, action, reward, done, success):
        self.step += 1
        elapsed = time.time() - self.start_time
        self.writer.writerow([
            time.time(),
            self.episode,
            self.step,
            elapsed,
            list(map(float, state)),
            action,
            reward,
            done,
            success
        ])
        self.file.flush()

acrobot_rec = GameRecorder("acrobot")
mountaincar_rec = GameRecorder("mountaincar")
cartpole_rec = GameRecorder("cartpole")

# =====================
# Utility function
# =====================
def frame_to_base64(frame: np.ndarray) -> str:
    img = Image.fromarray(frame)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# =====================
# Initialize environments
# =====================
acrobot = WebAcrobot()
mountaincar = WebMountainCar()
cartpole = WebCartPole()
cartpole.training_mode = False

# =====================
# Routes for static pages
# =====================
@app.route("/")
def root():
    return redirect("/static/index.html")

@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# =====================
# API endpoints
# =====================
@app.route("/acrobot/reset", methods=["POST"])
def reset_acrobot():
    acrobot.reset()
    acrobot_rec.new_episode()
    frame = acrobot.render()
    return jsonify({"frame": frame_to_base64(frame), "success": False})

@app.route("/acrobot/step/<int:action>", methods=["POST"])
def step_acrobot(action):
    obs, reward, done = acrobot.step(action)
    frame = acrobot.render()
    x_tip, y_tip = acrobot.get_tip_position()

    acrobot_rec.log(
        state=obs,
        action=action,
        reward=reward,
        done=done,
        success=done
    )

    return jsonify({
        "frame": frame_to_base64(frame),
        "success": done,
        "tip_y": y_tip
    })

@app.route("/mountaincar/newsession", methods=["POST"])
def new_session():
    global mountaincar
    mountaincar.close()
    mountaincar = WebMountainCar()
    return jsonify({"status": "new session"})

@app.route("/mountaincar/reset", methods=["POST"])
def reset_mountaincar():
    data = request.json or {}
    training = data.get("training", False)
    goal = data.get("goalX", 0.5)

    mountaincar.reset(training_mode=training, goal_x=goal)
    mountaincar_rec.new_episode()

    return jsonify({
        "frame": frame_to_base64(mountaincar.render()),
        "laps": mountaincar.lap_times
    })

@app.route("/mountaincar/step", methods=["POST"])
def step_mountaincar():
    data = request.json or {}
    action = int(data.get("action", 0))

    obs, reward, done, success = mountaincar.step(action)

    mountaincar_rec.log(
        state=obs,
        action=action,
        reward=reward,
        done=done,
        success=success
    )

    position = float(obs[0])
    return jsonify({
        "done": done,
        "success": success,
        "laps": mountaincar.lap_times,
        "frame": frame_to_base64(mountaincar.render()),
        "position": position
    })

from fastapi.responses import JSONResponse
from .utils.render import render_frame  # make sure you have this utility

@app.post("/cartpole/reset")
async def reset_cartpole(data: dict = Body(default={})):
    training = data.get("training", False)

    # IMPORTANT: pass into env reset
    cartpole.reset(training=training)
    cartpole_rec.new_episode()

    return {"frame": render_frame(cartpole)}

@app.post("/cartpole/step/{action}")
def step_cartpole(action: int):

    obs, reward, terminated, truncated, info = cartpole.step(action)

    x, x_dot, theta, theta_dot = obs

    frame_b64 = frame_to_base64(cartpole.render())

    # ONLY failure ends a round
    done = terminated

    cartpole_rec.log(
        state=obs,
        action=action,
        reward=reward,
        done=done,
        success=not done
    )

    return JSONResponse({
        "frame": frame_b64,
        "done": bool(done),
        "truncated": bool(truncated),  # <-- new
        "theta": float(theta),
        "cart_x": float(x)
    })
