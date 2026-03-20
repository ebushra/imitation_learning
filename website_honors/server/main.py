import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import time
import csv
import base64
from io import BytesIO

import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, send_from_directory, redirect, session
import uuid
import threading

app = Flask(__name__, static_folder="../static")
app.secret_key = "super_secret_key"  # REQUIRED for sessions

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
                "session_id",   # NEW
                "timestamp","episode","step","elapsed",
                "state","action","reward","done","success"
            ])

    def new_episode(self):
        self.episode += 1
        self.step = 0
        self.start_time = time.time()

    def log(self, session_id, state, action, reward, done, success):
        self.step += 1
        elapsed = time.time() - self.start_time
        self.writer.writerow([
            session_id,   # NEW
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
# Initialize environments
# =====================
# Store environments per session
envs = {}
locks = {}

def get_session_id():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]

def get_env(session_id):
    if session_id not in envs:
        envs[session_id] = {
            "acrobot": WebAcrobot(),
            "mountaincar": WebMountainCar(),
            "cartpole": WebCartPole()
        }
        locks[session_id] = threading.Lock()
    return envs[session_id], locks[session_id]
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
# Flask /acrobot/reset
@app.route("/acrobot/reset", methods=["POST"])
def reset_acrobot():
    sid = get_session_id()
    env, lock = get_env(sid)

    with lock:
        env["acrobot"].reset()
        acrobot_rec.new_episode()

        return jsonify({
            "state": env["acrobot"].get_state(),
            "success": False
        })
    
@app.route("/acrobot/step/<int:action>", methods=["POST"])
def step_acrobot(action):
    sid = get_session_id()
    env, lock = get_env(sid)

    with lock:
        obs, reward, done = env["acrobot"].step(action)

        acrobot_rec.log(
            sid,
            state=obs,
            action=action,
            reward=reward,
            done=done,
            success=done
        )

        return jsonify({
            "state": env["acrobot"].get_state(),
            "success": bool(done)
        })
    
@app.route("/mountaincar/newsession", methods=["POST"])
def new_session():
    global mountaincar
    mountaincar.close()
    mountaincar = WebMountainCar()
    return jsonify({"status": "new session"})

@app.route("/mountaincar/reset", methods=["POST"])
def reset_mountaincar():
    sid = get_session_id()
    env, lock = get_env(sid)

    data = request.json or {}
    training = data.get("training", False)
    goal = data.get("goalX", 0.5)

    with lock:
        obs = env["mountaincar"].reset(training_mode=training, goal_x=goal)
        mountaincar_rec.new_episode()

        return jsonify({
            "state": list(map(float, obs)),
            "laps": env["mountaincar"].lap_times
        })

@app.route("/mountaincar/step", methods=["POST"])
def step_mountaincar():
    sid = get_session_id()
    env, lock = get_env(sid)

    data = request.json or {}
    action = int(data.get("action", 0))

    with lock:
        obs, reward, done, success = env["mountaincar"].step(action)

        mountaincar_rec.log(
            sid,
            state=obs,
            action=action,
            reward=reward,
            done=done,
            success=success
        )

        return jsonify({
            "state": list(map(float, obs)),
            "done": done,
            "success": success,
            "laps": env["mountaincar"].lap_times
        })
    
from fastapi.responses import JSONResponse
from .utils.render import render_frame  # make sure you have this utility

@app.route("/cartpole/reset", methods=["POST"])
def reset_cartpole():
    sid = get_session_id()
    env, lock = get_env(sid)

    data = request.json or {}
    training = data.get("training", False)

    with lock:
        env["cartpole"].reset(training=training)
        cartpole_rec.new_episode()

        x, x_dot, theta, theta_dot = env["cartpole"].get_state()

        return jsonify({
            "state": list(map(float, [x, x_dot, theta, theta_dot])),
            "done": False
        })


@app.route("/cartpole/step/<int:action>", methods=["POST"])
def step_cartpole(action):
    sid = get_session_id()
    env, lock = get_env(sid)

    with lock:
        obs, reward, terminated, truncated, _ = env["cartpole"].step(action)

        done = terminated

        cartpole_rec.log(
            sid,
            state=obs,
            action=action,
            reward=reward,
            done=done,
            success=not done
        )

        return jsonify({
            "state": list(map(float, obs)),
            "done": bool(done),
            "truncated": bool(truncated)
        })
