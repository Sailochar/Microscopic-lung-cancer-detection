"""Local API and static server for the PrivCanFed dashboard.

Run from the project root:
    python src/dashboard_server.py
"""

import base64
import json
import mimetypes
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import urlparse

import torch
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from model import get_model
from preprocessing import VAL_TRANSFORM
from preprocessing import get_test_loader

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "4173"))
CLASS_NAMES = ["lung_aca", "lung_n", "lung_scc"]
CHECKPOINTS = {
    "fedprox": os.path.join(ROOT_DIR, "checkpoints", "global_fedprox_model.pth"),
    "fedavg": os.path.join(ROOT_DIR, "checkpoints", "global_fedavg_model.pth"),
    "hospital1": os.path.join(ROOT_DIR, "checkpoints", "hospital1_model.pth"),
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {}
METRICS_CACHE = None

# Reported held-out results supplied for the dashboard refresh.
REPORT_METRICS = [
    {"model": "Hospital 1", "accuracy": 0.9987, "precision": 0.9987, "recall": 0.9987, "macroF1": 0.9987},
    {"model": "Hospital 2", "accuracy": 0.9887, "precision": 0.9887, "recall": 0.9887, "macroF1": 0.9887},
    {"model": "Hospital 3", "accuracy": 0.9973, "precision": 0.9973, "recall": 0.9973, "macroF1": 0.9973},
    {"model": "FedAvg", "accuracy": 0.9947, "precision": 0.9947, "recall": 0.9947, "macroF1": 0.9947},
    {"model": "FedProx", "accuracy": 0.9993, "precision": 0.9993, "recall": 0.9993, "macroF1": 0.9993},
]


def load_model(model_key):
    if model_key in MODELS:
        return MODELS[model_key]
    checkpoint_path = CHECKPOINTS.get(model_key, CHECKPOINTS["fedprox"])
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = get_model(num_classes=len(CLASS_NAMES), freeze_backbone=False,
                      pretrained=False)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    MODELS[model_key] = model
    return model


def decode_image(data_url):
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("Expected a base64 data URL")
    _, encoded = data_url.split(",", 1)
    image = Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
    return image


def validate_microscopy_image(image):
    """Reject obvious out-of-domain images before the closed-set classifier runs."""
    sample = np.asarray(image.resize((128, 128)), dtype=np.float32) / 255.0
    dark_ratio = float(np.all(sample < 0.12, axis=2).mean())
    bright_ratio = float(np.all(sample > 0.88, axis=2).mean())
    color_spread = sample.max(axis=2) - sample.min(axis=2)
    colored_ratio = float((color_spread > 0.10).mean())
    luminance = sample.mean(axis=2)
    contrast = float(luminance.std())

    if dark_ratio > 0.42 and colored_ratio < 0.28:
        raise ValueError("Unsupported image: upload a stained microscopic lung-tissue image.")
    if bright_ratio > 0.72 and contrast < 0.10:
        raise ValueError("Unsupported image: upload a stained microscopic lung-tissue image.")
    if colored_ratio < 0.08 or contrast < 0.035:
        raise ValueError("Unsupported image: upload a stained microscopic lung-tissue image.")


def predict(data_url, model_key):
    image = decode_image(data_url)
    validate_microscopy_image(image)
    tensor = VAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    model = load_model(model_key)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    ranked = sorted(zip(CLASS_NAMES, probabilities), key=lambda item: item[1], reverse=True)
    return {
        "prediction": ranked[0][0],
        "confidence": ranked[0][1],
        "probabilities": {name: value for name, value in zip(CLASS_NAMES, probabilities)},
        "model": model_key,
        "device": str(DEVICE),
    }


def get_metrics():
    global METRICS_CACHE
    if METRICS_CACHE is not None:
        return METRICS_CACHE
    METRICS_CACHE = {"metrics": REPORT_METRICS, "device": "reported held-out results", "source": "pasted evaluation report"}
    return METRICS_CACHE


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            result = predict(payload.get("image"), payload.get("model", "fedprox"))
            self.send_json(200, result)
        except Exception as error:
            self.send_json(400, {"error": str(error)})

    def do_GET(self):
        if urlparse(self.path).path == "/api/metrics":
            try:
                self.send_json(200, get_metrics())
            except Exception as error:
                self.send_json(500, {"error": str(error)})
            return
        super().do_GET()

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"PrivCanFed dashboard: http://{HOST}:{PORT}/dashboard/")
    print(f"Inference device: {DEVICE}")
    ThreadingHTTPServer((HOST, PORT), DashboardHandler).serve_forever()