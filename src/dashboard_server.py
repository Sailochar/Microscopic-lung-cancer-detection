"""Local API and static server for the PrivCanFed dashboard.

Run from the project root:
    python src/dashboard_server.py
"""

import base64
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")

sys.path.insert(0, SRC_DIR)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from model import get_model
from preprocessing import VAL_TRANSFORM
from preprocessing import get_test_loader


# ============================================================
# SERVER CONFIGURATION
# ============================================================

HOST = "0.0.0.0"

# Render provides the PORT environment variable.
# For local development, 4173 is used.
PORT = int(os.environ.get("PORT", "4173"))


# ============================================================
# CORS CONFIGURATION
# ============================================================

# Your deployed Vercel frontend.
VERCEL_ORIGIN = "https://microscopic-lung-cancer-detection.vercel.app"

# Local origins are included so the dashboard still works
# when you run it directly on your computer.
ALLOWED_ORIGINS = {
    VERCEL_ORIGIN,
    "http://localhost:4173",
    "http://127.0.0.1:4173",
}


# ============================================================
# MODEL CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "lung_aca",
    "lung_n",
    "lung_scc",
]

CHECKPOINTS = {
    "fedprox": os.path.join(
        ROOT_DIR,
        "checkpoints",
        "global_fedprox_model.pth",
    ),

    "fedavg": os.path.join(
        ROOT_DIR,
        "checkpoints",
        "global_fedavg_model.pth",
    ),

    "hospital1": os.path.join(
        ROOT_DIR,
        "checkpoints",
        "hospital1_model.pth",
    ),
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL CACHE
# ============================================================

MODELS = {}


# ============================================================
# METRICS CACHE
# ============================================================

METRICS_CACHE = None


# ============================================================
# REPORTED HELD-OUT RESULTS
# ============================================================

REPORT_METRICS = [
    {
        "model": "Hospital 1",
        "accuracy": 0.9987,
        "precision": 0.9987,
        "recall": 0.9987,
        "macroF1": 0.9987,
    },

    {
        "model": "Hospital 2",
        "accuracy": 0.9887,
        "precision": 0.9887,
        "recall": 0.9887,
        "macroF1": 0.9887,
    },

    {
        "model": "Hospital 3",
        "accuracy": 0.9973,
        "precision": 0.9973,
        "recall": 0.9973,
        "macroF1": 0.9973,
    },

    {
        "model": "FedAvg",
        "accuracy": 0.9947,
        "precision": 0.9947,
        "recall": 0.9947,
        "macroF1": 0.9947,
    },

    {
        "model": "FedProx",
        "accuracy": 0.9993,
        "precision": 0.9993,
        "recall": 0.9993,
        "macroF1": 0.9993,
    },
]


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_key):
    """
    Load the requested PyTorch checkpoint.

    Models are cached after the first load so repeated
    predictions do not reload the checkpoint from disk.
    """

    if model_key in MODELS:
        return MODELS[model_key]

    # Only allow supported model names.
    if model_key not in CHECKPOINTS:
        model_key = "fedprox"

    checkpoint_path = CHECKPOINTS[model_key]

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print(
        f"Loading model '{model_key}' "
        f"from: {checkpoint_path}"
    )

    model = get_model(
        num_classes=len(CLASS_NAMES),
        freeze_backbone=False,
        pretrained=False,
    )

    state = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    # Support checkpoints stored as:
    # {"model_state_dict": ...}
    if (
        isinstance(state, dict)
        and "model_state_dict" in state
    ):
        state = state["model_state_dict"]

    model.load_state_dict(
        state,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    MODELS[model_key] = model

    print(
        f"Model '{model_key}' loaded successfully."
    )

    return model


# ============================================================
# IMAGE DECODING
# ============================================================

def decode_image(data_url):
    """
    Convert a base64 data URL into a PIL RGB image.
    """

    if (
        not isinstance(data_url, str)
        or "," not in data_url
    ):
        raise ValueError(
            "Expected a base64 data URL"
        )

    _, encoded = data_url.split(",", 1)

    try:
        image_bytes = base64.b64decode(
            encoded,
            validate=True,
        )
    except Exception as error:
        raise ValueError(
            "Invalid base64 image data"
        ) from error

    try:
        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")
    except Exception as error:
        raise ValueError(
            "Unable to decode the uploaded image"
        ) from error

    return image


# ============================================================
# MICROSCOPY IMAGE VALIDATION
# ============================================================

def validate_microscopy_image(image):
    """
    Reject obvious out-of-domain images before the
    closed-set classifier runs.
    """

    sample = np.asarray(
        image.resize((128, 128)),
        dtype=np.float32,
    ) / 255.0

    dark_ratio = float(
        np.all(
            sample < 0.12,
            axis=2,
        ).mean()
    )

    bright_ratio = float(
        np.all(
            sample > 0.88,
            axis=2,
        ).mean()
    )

    color_spread = (
        sample.max(axis=2)
        - sample.min(axis=2)
    )

    colored_ratio = float(
        (
            color_spread > 0.10
        ).mean()
    )

    luminance = sample.mean(axis=2)

    contrast = float(
        luminance.std()
    )

    if (
        dark_ratio > 0.42
        and colored_ratio < 0.28
    ):
        raise ValueError(
            "Unsupported image: upload a stained microscopic lung-tissue image."
        )

    if (
        bright_ratio > 0.72
        and contrast < 0.10
    ):
        raise ValueError(
            "Unsupported image: upload a stained microscopic lung-tissue image."
        )

    if (
        colored_ratio < 0.08
        or contrast < 0.035
    ):
        raise ValueError(
            "Unsupported image: upload a stained microscopic lung-tissue image."
        )


# ============================================================
# PREDICTION
# ============================================================

def predict(data_url, model_key):
    """
    Run inference using the selected model.
    """

    image = decode_image(data_url)

    validate_microscopy_image(image)

    tensor = (
        VAL_TRANSFORM(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    model = load_model(model_key)

    with torch.no_grad():
        probabilities = torch.softmax(
            model(tensor),
            dim=1,
        )[0].cpu().tolist()

    ranked = sorted(
        zip(CLASS_NAMES, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "prediction": ranked[0][0],

        "confidence": ranked[0][1],

        "probabilities": {
            name: value
            for name, value
            in zip(
                CLASS_NAMES,
                probabilities,
            )
        },

        "model": model_key,

        "device": str(DEVICE),
    }


# ============================================================
# METRICS
# ============================================================

def get_metrics():
    """
    Return the reported held-out evaluation metrics.
    """

    global METRICS_CACHE

    if METRICS_CACHE is not None:
        return METRICS_CACHE

    METRICS_CACHE = {
        "metrics": REPORT_METRICS,
        "device": "reported held-out results",
        "source": "pasted evaluation report",
    }

    return METRICS_CACHE


# ============================================================
# HTTP HANDLER
# ============================================================

class DashboardHandler(SimpleHTTPRequestHandler):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            directory=ROOT_DIR,
            **kwargs,
        )


    # --------------------------------------------------------
    # CORS
    # --------------------------------------------------------

    def add_cors_headers(self):
        """
        Add CORS headers required for the Vercel frontend
        to communicate with the Render backend.
        """

        origin = self.headers.get(
            "Origin"
        )

        if origin in ALLOWED_ORIGINS:

            self.send_header(
                "Access-Control-Allow-Origin",
                origin,
            )

            self.send_header(
                "Vary",
                "Origin",
            )

            self.send_header(
                "Access-Control-Allow-Methods",
                "GET, POST, OPTIONS",
            )

            self.send_header(
                "Access-Control-Allow-Headers",
                "Content-Type",
            )

            self.send_header(
                "Access-Control-Max-Age",
                "600",
            )


    # --------------------------------------------------------
    # HTTP RESPONSE HEADERS
    # --------------------------------------------------------

    def end_headers(self):

        self.send_header(
            "Cache-Control",
            "no-store",
        )

        self.add_cors_headers()

        super().end_headers()


    # --------------------------------------------------------
    # OPTIONS / CORS PREFLIGHT
    # --------------------------------------------------------

    def do_OPTIONS(self):
        """
        Handle browser CORS preflight requests.

        This is required because the frontend sends JSON
        using Content-Type: application/json.
        """

        path = urlparse(
            self.path
        ).path

        if path not in (
            "/api/predict",
            "/api/metrics",
        ):
            self.send_error(404)
            return

        self.send_response(204)

        self.send_header(
            "Content-Length",
            "0",
        )

        self.end_headers()


    # --------------------------------------------------------
    # POST /api/predict
    # --------------------------------------------------------

    def do_POST(self):

        path = urlparse(
            self.path
        ).path

        if path != "/api/predict":
            self.send_error(404)
            return

        try:

            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

            # Prevent obviously invalid requests.
            if content_length <= 0:
                raise ValueError(
                    "Request body is empty"
                )

            # 15 MB request limit.
            max_request_size = 15 * 1024 * 1024

            if content_length > max_request_size:
                raise ValueError(
                    "Image request is too large"
                )

            raw_body = self.rfile.read(
                content_length
            )

            payload = json.loads(
                raw_body
            )

            if not isinstance(
                payload,
                dict,
            ):
                raise ValueError(
                    "Invalid request payload"
                )

            image_data = payload.get(
                "image"
            )

            model_key = payload.get(
                "model",
                "fedprox",
            )

            if not image_data:
                raise ValueError(
                    "No image was provided"
                )

            # Supported models only.
            allowed_models = set(
                CHECKPOINTS.keys()
            )

            if model_key not in allowed_models:
                model_key = "fedprox"

            result = predict(
                image_data,
                model_key,
            )

            self.send_json(
                200,
                result,
            )

        except json.JSONDecodeError:
            self.send_json(
                400,
                {
                    "error":
                        "Invalid JSON request"
                },
            )

        except ValueError as error:
            self.send_json(
                400,
                {
                    "error": str(error)
                },
            )

        except FileNotFoundError as error:
            print(
                f"Checkpoint error: {error}"
            )

            self.send_json(
                500,
                {
                    "error":
                        "Model checkpoint is unavailable"
                },
            )

        except Exception as error:

            print(
                "Prediction error:",
                repr(error),
            )

            self.send_json(
                500,
                {
                    "error":
                        "Prediction failed on the backend"
                },
            )


    # --------------------------------------------------------
    # GET /api/metrics
    # --------------------------------------------------------

    def do_GET(self):

        path = urlparse(
            self.path
        ).path

        if path == "/api/metrics":

            try:

                self.send_json(
                    200,
                    get_metrics(),
                )

            except Exception as error:

                print(
                    "Metrics error:",
                    repr(error),
                )

                self.send_json(
                    500,
                    {
                        "error":
                            "Unable to load metrics"
                    },
                )

            return

        # Everything else is served as a static file.
        super().do_GET()


    # --------------------------------------------------------
    # JSON RESPONSE
    # --------------------------------------------------------

    def send_json(
        self,
        status,
        payload,
    ):

        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(
            status
        )

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.end_headers()

        self.wfile.write(
            body
        )


# ============================================================
# SERVER START
# ============================================================

if __name__ == "__main__":

    print(
        "=================================================="
    )

    print(
        "PrivCanFed Dashboard Backend"
    )

    print(
        "=================================================="
    )

    print(
        f"Host: {HOST}"
    )

    print(
        f"Port: {PORT}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Project root: {ROOT_DIR}"
    )

    print(
        f"Vercel frontend: {VERCEL_ORIGIN}"
    )

    print(
        "=================================================="
    )

    server = ThreadingHTTPServer(
        (HOST, PORT),
        DashboardHandler,
    )

    print(
        f"Server running on port {PORT}"
    )

    try:
        server.serve_forever()

    except KeyboardInterrupt:

        print(
            "\nServer stopped."
        )

    finally:

        server.server_close()