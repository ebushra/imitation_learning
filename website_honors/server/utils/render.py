import base64
from PIL import Image
from io import BytesIO
import numpy as np

def render_frame(env):
    """
    Converts env.render() (RGB numpy array) to base64 PNG
    """
    frame = env.render()
    if frame is None:
        return ""
    img = Image.fromarray(frame)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
