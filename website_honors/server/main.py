import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import time
import csv
import base64
from io import BytesIO
from flask import session
import uuid
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, send_from_directory, redirect

from .envs.acrobot_env import WebAcrobot
from .envs.mountaincar_env import WebMountainCar
from .envs.cartpole_env import WebCartPole
from .utils.render import render_frame

app = Flask(__name__, static_folder="../static")
app.secret_key = "dev-secret-key"

DATA_DIR = "/var/data/human_data"
os.makedirs(DATA_DIR, exist_ok=True)

class GameRecorder:
    def __init__(self, name):
        self.name = name
        self.episode = 0
        self.step = 0
        self.start_time = time.time()

        sid = get_session_id()
        self.user_id = sid

        self.filepath = f"{DATA_DIR}/{name}_{sid}.csv"

        file_exists = os.path.exists(self.filepath)
        file_empty = not file_exists or os.stat(self.filepath).st_size == 0

        self.file = open(self.filepath, "a", newline="")
        self.writer = csv.writer(self.file)

        # write header once
        if file_empty:
            self.writer.writerow([
                "user_id",
                "episode",
                "step",
                "t",
                "action",
                "reward",
                "done",
                "success",
                "training",
                "state"
            ])
            self.file.flush()

    def new_episode(self):
        self.episode += 1
        self.step = 0
        self.start_time = time.time()

    def log(self, state, action, reward, done, success, training=False):
        self.step += 1

        t = time.time() - self.start_time

        # flatten state (NO brackets)
        state_flat = [float(x) for x in state]

        self.writer.writerow([
            self.user_id,
            self.episode,
            self.step,
            t,
            action,
            reward,
            done,
            success,
            training,
            *state_flat
        ])

        if self.step % 20 == 0 or done:
            self.file.flush()

# Create recorders
recorders = {}

def get_recorders():
    sid = get_session_id()
    if sid not in recorders:
        recorders[sid] = {
            "acrobot": GameRecorder(f"acrobot_{sid}"),
            "mountaincar": GameRecorder(f"mountaincar_{sid}"),
            "cartpole": GameRecorder(f"cartpole_{sid}")
        }
    return recorders[sid]

# Environment storage
envs = {}

@app.route("/get_session_id")
def get_session():
    return jsonify({"session_id": get_session_id()})

def get_session_id():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]

def get_envs():
    sid = get_session_id()
    if sid not in envs:
        envs[sid] = {
            "acrobot": WebAcrobot(),
            "mountaincar": WebMountainCar(),
            "cartpole": WebCartPole()
        }
    return envs[sid]

@app.route("/")
def root():
    return redirect("/static/consent.html")

@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route("/acrobot/reset", methods=["POST"])
def reset_acrobot():
    env = get_envs()
    rec = get_recorders()

    env["acrobot"].reset()
    rec["acrobot"].new_episode()

    return jsonify({
        "state": env["acrobot"].get_state(),
        "success": False
    })
    
@app.route("/acrobot/step/<int:action>", methods=["POST"])
def step_acrobot(action):
    env = get_envs()
    rec = get_recorders()

    data = request.get_json(silent=True) or {}
    training = data.get("training")

    obs, reward, done = env["acrobot"].step(action)
    rec["acrobot"].log(
        state=obs,
        action=action,
        reward=reward,
        done=done,
        success=done,
        training=training
    )

    return jsonify({
        "state": env["acrobot"].get_state(),
        "success": bool(done)
    })
    
@app.route("/mountaincar/newsession", methods=["POST"])
def new_session():
    env = get_envs()
    
    env["mountaincar"].close()
    env["mountaincar"] = WebMountainCar()

    return jsonify({"status": "new session"})

@app.route("/mountaincar/reset", methods=["POST"])
def reset_mountaincar():
    env = get_envs()
    rec = get_recorders()

    data = request.json or {}
    training = data.get("training", False)
    goal = data.get("goalX", 0.5)

    obs = env["mountaincar"].reset(training_mode=training, goal_x=goal)
    rec["mountaincar"].new_episode()

    return jsonify({
        "state": list(map(float, obs)),
        "laps": env["mountaincar"].lap_times
    })

@app.route("/mountaincar/step", methods=["POST"])
def step_mountaincar():
    env = get_envs()
    rec = get_recorders()

    data = request.json or {}
    action = int(data.get("action", 0))

    obs, reward, done, success = env["mountaincar"].step(action)

    data = request.json or {}
    training = data.get("training", False)
    rec["mountaincar"].log(
        state=obs,
        action=action,
        reward=reward,
        done=done,
        success=done,
        training=training
    )

    return jsonify({
        "state": list(map(float, obs)),
        "done": done,
        "success": success,
        "laps": env["mountaincar"].lap_times
    })
    
from fastapi.responses import JSONResponse
from .utils.render import render_frame  

@app.route("/cartpole/reset", methods=["POST"])
def reset_cartpole():
    env = get_envs()
    rec = get_recorders()

    data = request.json or {}
    training = data.get("training", False)

    env["cartpole"].reset(training=training)
    rec["cartpole"].new_episode()

    x, x_dot, theta, theta_dot = env["cartpole"].get_state()

    return jsonify({
        "state": list(map(float, env["cartpole"].get_state())),
        "theta": float(theta),
        "cart_x": float(x),
        "done": False,
        "truncated": False
    })


@app.route("/cartpole/step/<int:action>", methods=["POST"])
def step_cartpole(action):
    env = get_envs()
    rec = get_recorders()

    obs, reward, terminated, truncated, info = env["cartpole"].step(action)

    x, x_dot, theta, theta_dot = obs
    done = terminated
    data = request.json or {}
    training = data.get("training", False)
    rec["cartpole"].log(
        state=obs,
        action=action,
        reward=reward,
        done=done,
        success=not done,
        training=training
    )

    return jsonify({
        "state": list(map(float, obs)),
        "theta": float(theta),
        "cart_x": float(x),
        "done": bool(done),
        "truncated": bool(truncated)
    })
