import os
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import RedirectResponse
from pydantic import BaseModel


from .envs.acrobot_env import WebAcrobot
from .envs.mountaincar_env import WebMountainCar
from .envs.cartpole_env import WebCartPole
from .utils.render import render_frame

import base64
from io import BytesIO
from PIL import Image
import numpy as np

import csv
import time

DATA_DIR = "human_data"
os.makedirs(DATA_DIR, exist_ok=True)


class GameRecorder:
    def __init__(self, name):
        self.name = name
        self.episode = 0
        self.step = 0
        self.start_time = time.time()

        self.file = open(f"{DATA_DIR}/{name}.csv", "a", newline="")
        self.writer = csv.writer(self.file)

        # write header once
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


def frame_to_base64(frame: np.ndarray) -> str:
    """
    Convert an RGB frame (numpy array) to base64 string for browser display.
    """
    img = Image.fromarray(frame)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")

app = FastAPI()
app.mount("/website_honors/static", StaticFiles(directory="static"), name="static")

acrobot = WebAcrobot()
mountaincar = WebMountainCar()
cartpole = WebCartPole()
cartpole.training_mode = False  # default

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

@app.get("/index.html")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/cartpole.html")
def cartpole_page():
    return FileResponse(os.path.join(STATIC_DIR, "cartpole.html"))

@app.get("/cartpole_training.html")
def cartpole_page():
    return FileResponse(os.path.join(STATIC_DIR, "cartpole_training.html"))

@app.get("/mountaincar.html")
def mountaincar_page():
    return FileResponse(os.path.join(STATIC_DIR, "mountaincar.html"))

@app.get("/mountaincar_training.html")
def mountaincar_page():
    return FileResponse(os.path.join(STATIC_DIR, "mountaincar_training.html"))

@app.post("/acrobot/reset")
def reset_acrobot():
    acrobot.reset()
    acrobot_rec.new_episode()
    frame = acrobot.render()
    return {"frame": frame_to_base64(frame), "success": False}


@app.post("/acrobot/step/{action}")
def step_acrobot(action: int):
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

    return {
        "frame": frame_to_base64(frame),
        "success": done,
        "tip_y": y_tip  # send tip y-coordinate directly to frontend
    }

from pydantic import BaseModel

class StepRequest(BaseModel):
    action: int
    training: bool = False
    goalX: float = 0.5


@app.post("/mountaincar/newsession")
def new_session():
    global mountaincar
    mountaincar.close()
    mountaincar = WebMountainCar()
    return {"status": "new session"}


from fastapi import Body


@app.post("/mountaincar/reset")
async def reset_mountaincar(data: dict = Body(default={})):
    training = data.get("training", False)
    goal = data.get("goalX", 0.5)

    mountaincar.reset(training_mode=training, goal_x=goal)
    mountaincar_rec.new_episode()

    return {
        "frame": frame_to_base64(mountaincar.render()),
        "laps": mountaincar.lap_times
    }


from fastapi import FastAPI
from fastapi.responses import JSONResponse

@app.post("/mountaincar/step")
def step_mountaincar(req: StepRequest):
    obs, reward, done, success = mountaincar.step(req.action)

    mountaincar_rec.log(
        state=obs,
        action=req.action,
        reward=reward,
        done=done,
        success=success
    )

    position = float(obs[0])
    return JSONResponse({
        "done": done,
        "success": success,
        "laps": mountaincar.lap_times,
        "frame": frame_to_base64(mountaincar.render()),
        "position": position
    })



from fastapi import Body

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

