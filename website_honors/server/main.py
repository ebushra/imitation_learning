import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
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
def frame_to_base64_jpeg(frame, quality=50):
    img = Image.fromarray(frame)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
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
    # Reset environment
    acrobot.reset()
    acrobot_rec.new_episode()
    
    # Render safely
    try:
        frame = acrobot.render(mode="rgb_array")  # headless mode
    except Exception as e:
        print("Render failed on reset:", e)
        import numpy as np
        frame = np.zeros((350, 350, 3), dtype=np.uint8)  # fallback black frame

    return jsonify({
        "frame": frame_to_base64_jpeg(frame, quality=50),
        "success": False
    })

@app.route("/acrobot/step/<int:action>", methods=["POST"])
def step_acrobot(action):
    try:
        # Step the environment a few times for smoother motion
        for _ in range(3):
            obs, reward, done = acrobot.step(action)

        # Render (no mode argument, use whatever your wrapper provides)
        try:
            frame = acrobot.render()  # just call without mode
        except Exception as e:
            print("Render failed on step:", e)
            import numpy as np
            frame = np.zeros((350, 350, 3), dtype=np.uint8)  # fallback black frame

        # Convert tip_y to float to be JSON serializable
        x_tip, y_tip = acrobot.get_tip_position()
        y_tip = float(y_tip)

        # Log the step
        acrobot_rec.log(
            state=obs,
            action=action,
            reward=reward,
            done=done,
            success=done
        )

        return jsonify({
            "frame": frame_to_base64_jpeg(frame, quality=50),
            "success": bool(done),
            "tip_y": y_tip
        })
    except Exception as e:
        # safer error reporting
        tb = traceback.format_exc()
        print("Error in step_acrobot:", tb)
        return jsonify({"error": str(e), "trace": tb}), 500

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
        "frame": frame_to_base64_jpeg(mountaincar.render(), quality=50),
        "laps": mountaincar.lap_times
    })

@app.route("/mountaincar/step", methods=["POST"])
def step_mountaincar():
    data = request.json or {}
    action = int(data.get("action", 0))

    for _ in range(3):
        obs, reward, done, success = mountaincar.step(action)
        if done:
            break

    mountaincar_rec.log(
        state=obs,
        action=action,
        reward=reward,
        done=done,
        success=success
    )

    frame = mountaincar.render()
    frame_b64 = frame_to_base64_jpeg(frame)

    position = float(obs[0])
    return jsonify({
        "done": done,
        "success": success,
        "laps": mountaincar.lap_times,
        "frame": frame_b64,
        "position": position
    })

from fastapi.responses import JSONResponse
from .utils.render import render_frame  # make sure you have this utility

@app.route("/cartpole/reset", methods=["POST"])
def reset_cartpole():
    data = request.json or {}
    training = data.get("training", False)

    cartpole.reset()
    cartpole_rec.new_episode()

    x, x_dot, theta, theta_dot = cartpole.get_state()

    return jsonify({
        "frame": frame_to_base64(cartpole.render()),
        "done": False,
        "truncated": False,
        "theta": float(theta),
        "cart_x": float(x)
    })


@app.route("/cartpole/step/<int:action>", methods=["POST"])
def step_cartpole(action):
    obs, reward, terminated, truncated, info = cartpole.step(action)

    x, x_dot, theta, theta_dot = obs
    done = terminated

    cartpole_rec.log(
        state=obs,
        action=action,
        reward=reward,
        done=done,
        success=not done
    )

    return jsonify({
        "frame": frame_to_base64(cartpole.render()),
        "done": bool(done),
        "truncated": bool(truncated),
        "theta": float(theta),
        "cart_x": float(x)
    })
