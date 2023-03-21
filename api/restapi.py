"""
Run a rest API exposing the yolov8n_custom object detection model
"""
from ultralytics import YOLO
import argparse
import io
from PIL import Image

import torch
from flask import Flask, redirect, request

app = Flask(__name__)

DETECTION_URL = "/object-detection"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        results = model(img)
        res_plotted = results[0].plot(show_conf =True)
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            masks = result.masks  # Masks object for segmenation masks outputs
            probs = result.probs
        img_savename = f"static/result.png"
        return redirect(img_savename)
        #return results.pandas().xyxy[0].to_json(orient="records")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = YOLO('best.pt')
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
