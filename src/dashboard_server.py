"""PrivCanFed dashboard API.

Frontend:
    Vercel

Backend:
    Render

Local:
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


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    ROOT_DIR,
    "src"
)

sys.path.insert(
    0,
    SRC_DIR
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from model import get_model
from preprocessing import VAL_TRANSFORM


# ============================================================
# SERVER
# ============================================================

HOST = "0.0.0.0"

PORT = int(
    os.environ.get(
        "PORT",
        "4173"
    )
)


# ============================================================
# CLASSES
# ============================================================

CLASS_NAMES = [
    "lung_aca",
    "lung_n",
    "lung_scc",
]


# ============================================================
# CHECKPOINTS
# ============================================================

CHECKPOINTS = {
    "fedprox": os.path.join(
        ROOT_DIR,
        "checkpoints",
        "global_fedprox_model.pth"
    ),

    "fedavg": os.path.join(
        ROOT_DIR,
        "checkpoints",
        "global_fedavg_model.pth"
    ),

    "hospital1": os.path.join(
        ROOT_DIR,
        "checkpoints",
        "hospital1_model.pth"
    ),

    "hospital2": os.path.join(
        ROOT_DIR,
        "checkpoints",
        "hospital2_model.pth"
    ),

    "hospital3": os.path.join(
        ROOT_DIR,
        "checkpoints",
        "hospital3_model.pth"
    ),
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL CACHE
# ============================================================

MODELS = {}


# ============================================================
# CORS
# ============================================================

VERCEL_PRODUCTION_ORIGIN = (
    "https://microscopic-lung-cancer-detection.vercel.app"
)

ALLOWED_ORIGINS = {
    VERCEL_PRODUCTION_ORIGIN,

    "http://localhost:3000",
    "http://127.0.0.1:3000",

    "http://localhost:4173",
    "http://127.0.0.1:4173",
}


def is_allowed_origin(origin):
    if not origin:
        return False

    if origin in ALLOWED_ORIGINS:
        return True

    try:
        parsed = urlparse(origin)

        if parsed.scheme != "https":
            return False

        hostname = parsed.hostname

        if not hostname:
            return False

        # Allow Vercel preview deployments belonging
        # to this project.
        if (
            hostname.startswith(
                "microscopic-lung-cancer-detection"
            )
            and hostname.endswith(
                ".vercel.app"
            )
        ):
            return True

    except Exception:
        return False

    return False


# ============================================================
# METRICS
# ============================================================

REPORT_METRICS = [
    {
        "model": "Hospital 1",
        "accuracy": 0.9987,
        "precision": 0.9987,
        "recall": 0.9987,
        "macroF1": 0.9987
    },

    {
        "model": "Hospital 2",
        "accuracy": 0.9887,
        "precision": 0.9887,
        "recall": 0.9887,
        "macroF1": 0.9887
    },

    {
        "model": "Hospital 3",
        "accuracy": 0.9973,
        "precision": 0.9973,
        "recall": 0.9973,
        "macroF1": 0.9973
    },

    {
        "model": "FedAvg",
        "accuracy": 0.9947,
        "precision": 0.9947,
        "recall": 0.9947,
        "macroF1": 0.9947
    },

    {
        "model": "FedProx",
        "accuracy": 0.9993,
        "precision": 0.9993,
        "recall": 0.9993,
        "macroF1": 0.9993
    }
]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_key):

    if model_key not in CHECKPOINTS:
        model_key = "fedprox"

    if model_key in MODELS:
        return MODELS[model_key]

    checkpoint_path = CHECKPOINTS[
        model_key
    ]

    if not os.path.exists(
        checkpoint_path
    ):
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    print(
        f"[MODEL] Loading: {model_key}"
    )

    print(
        f"[MODEL] Checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"[MODEL] Device: {DEVICE}"
    )

    model = get_model(
        num_classes=len(
            CLASS_NAMES
        ),
        freeze_backbone=False,
        pretrained=False
    )

    state = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    if (
        isinstance(state, dict)
        and
        "model_state_dict" in state
    ):
        state = state[
            "model_state_dict"
        ]

    model.load_state_dict(
        state,
        strict=True
    )

    model.to(DEVICE)

    model.eval()

    MODELS[
        model_key
    ] = model

    print(
        f"[MODEL] Loaded successfully: "
        f"{model_key}"
    )

    return model


# ============================================================
# DECODE IMAGE
# ============================================================

def decode_image(data_url):

    if (
        not isinstance(
            data_url,
            str
        )
        or ","
        not in data_url
    ):
        raise ValueError(
            "Expected a base64 data URL"
        )

    _, encoded = data_url.split(
        ",",
        1
    )

    try:

        image_bytes = (
            base64.b64decode(
                encoded,
                validate=True
            )
        )

    except Exception as error:

        raise ValueError(
            "Invalid base64 image data"
        ) from error

    try:

        image = Image.open(
            BytesIO(
                image_bytes
            )
        ).convert("RGB")

    except Exception as error:

        raise ValueError(
            "Unable to decode the uploaded image"
        ) from error

    return image


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_microscopy_image(image):

    width, height = image.size

    if (
        width < 128
        or height < 128
        or width / height < 0.65
        or width / height > 1.55
    ):
        raise ValueError(
            "Unsupported image: upload a stained microscopic lung-tissue image."
        )

    sample = np.asarray(
        image.resize(
            (128, 128)
        ),
        dtype=np.float32
    ) / 255.0

    dark_ratio = float(
        np.all(
            sample < 0.12,
            axis=2
        ).mean()
    )

    bright_ratio = float(
        np.all(
            sample > 0.88,
            axis=2
        ).mean()
    )

    color_spread = (
        sample.max(axis=2)
        -
        sample.min(axis=2)
    )

    colored_ratio = float(
        (
            color_spread > 0.10
        ).mean()
    )

    luminance = sample.mean(
        axis=2
    )

    contrast = float(
        luminance.std()
    )

    texture_score = float(
        np.abs(np.diff(luminance, axis=1)).mean()
        +
        np.abs(np.diff(luminance, axis=0)).mean()
    )

    red = sample[:, :, 0]
    green = sample[:, :, 1]
    blue = sample[:, :, 2]

    stained_ratio = float(
        (
            (red > green * 0.95)
            &
            (blue > green * 1.03)
            &
            (color_spread > 0.10)
        ).mean()
    )

    if (
        dark_ratio > 0.42
        and
        colored_ratio < 0.28
    ):
        raise ValueError(
            "Unsupported image: upload a stained microscopic lung-tissue image."
        )

    if (
        bright_ratio > 0.72
        and
        contrast < 0.10
    ):
        raise ValueError(
            "Unsupported image: upload a stained microscopic lung-tissue image."
        )

    if (
        colored_ratio < 0.08
        or
        contrast < 0.035
        or
        texture_score < 0.035
        or
        stained_ratio < 0.10
    ):
        raise ValueError(
            "Unsupported image: upload a stained microscopic lung-tissue image."
        )


# ============================================================
# PREDICTION
# ============================================================

def predict(
    data_url,
    model_key
):

    image = decode_image(
        data_url
    )

    validate_microscopy_image(
        image
    )

    tensor = (
        VAL_TRANSFORM(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    model = load_model(
        model_key
    )

    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=1
            )[0]
            .cpu()
            .tolist()
        )

    ranked = sorted(
        zip(
            CLASS_NAMES,
            probabilities
        ),
        key=lambda item: item[1],
        reverse=True
    )

    prediction = ranked[0][0]

    confidence = ranked[0][1]

    return {
        "prediction": prediction,

        "confidence": confidence,

        "probabilities": {
            name: value
            for name, value in zip(
                CLASS_NAMES,
                probabilities
            )
        },

        "model": model_key,

        "device": str(
            DEVICE
        )
    }


# ============================================================
# METRICS
# ============================================================

def get_metrics():

    return {
        "metrics": REPORT_METRICS,

        "device":
            "reported held-out results",

        "source":
            "pasted evaluation report"
    }


# ============================================================
# HTTP HANDLER
# ============================================================

class DashboardHandler(
    SimpleHTTPRequestHandler
):

    def __init__(
        self,
        *args,
        **kwargs
    ):

        super().__init__(
            *args,
            directory=ROOT_DIR,
            **kwargs
        )


    # --------------------------------------------------------
    # CORS HEADERS
    # --------------------------------------------------------

    def add_cors_headers(self):

        origin = self.headers.get(
            "Origin"
        )

        if is_allowed_origin(
            origin
        ):

            self.send_header(
                "Access-Control-Allow-Origin",
                origin
            )

            self.send_header(
                "Access-Control-Allow-Methods",
                "GET, POST, OPTIONS"
            )

            self.send_header(
                "Access-Control-Allow-Headers",
                "Content-Type"
            )

            self.send_header(
                "Access-Control-Max-Age",
                "600"
            )

            self.send_header(
                "Vary",
                "Origin"
            )


    # --------------------------------------------------------
    # HEADERS
    # --------------------------------------------------------

    def end_headers(self):

        self.send_header(
            "Cache-Control",
            "no-store"
        )

        self.add_cors_headers()

        super().end_headers()


    # --------------------------------------------------------
    # OPTIONS
    # --------------------------------------------------------

    def do_OPTIONS(self):

        path = urlparse(
            self.path
        ).path

        if path not in (
            "/api/predict",
            "/api/metrics"
        ):

            self.send_error(
                404
            )

            return

        origin = self.headers.get(
            "Origin"
        )

        if (
            origin
            and
            not is_allowed_origin(
                origin
            )
        ):

            self.send_error(
                403,
                "Origin not allowed"
            )

            return

        self.send_response(
            204
        )

        self.send_header(
            "Content-Length",
            "0"
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

            self.send_error(
                404
            )

            return

        try:

            origin = self.headers.get(
                "Origin"
            )

            if (
                origin
                and
                not is_allowed_origin(
                    origin
                )
            ):

                self.send_json(
                    403,
                    {
                        "error":
                            "Origin not allowed"
                    }
                )

                return

            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )

            if content_length <= 0:

                raise ValueError(
                    "Request body is empty"
                )

            max_request_size = (
                15 * 1024 * 1024
            )

            if (
                content_length
                > max_request_size
            ):

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
                dict
            ):

                raise ValueError(
                    "Invalid request payload"
                )

            image_data = payload.get(
                "image"
            )

            if not image_data:

                raise ValueError(
                    "No image was provided"
                )

            model_key = payload.get(
                "model",
                "fedprox"
            )

            if model_key not in CHECKPOINTS:

                model_key = "fedprox"

            result = predict(
                image_data,
                model_key
            )

            self.send_json(
                200,
                result
            )

        except json.JSONDecodeError:

            self.send_json(
                400,
                {
                    "error":
                        "Invalid JSON request"
                }
            )

        except ValueError as error:

            self.send_json(
                400,
                {
                    "error":
                        str(error)
                }
            )

        except FileNotFoundError as error:

            print(
                "[ERROR] Checkpoint:",
                error
            )

            self.send_json(
                500,
                {
                    "error":
                        "Model checkpoint is unavailable"
                }
            )

        except Exception as error:

            print(
                "[ERROR] Prediction:",
                repr(error)
            )

            self.send_json(
                500,
                {
                    "error":
                        "Prediction failed on backend"
                }
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
                    get_metrics()
                )

            except Exception as error:

                print(
                    "[ERROR] Metrics:",
                    repr(error)
                )

                self.send_json(
                    500,
                    {
                        "error":
                            "Unable to load metrics"
                    }
                )

            return

        super().do_GET()


    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    def send_json(
        self,
        status,
        payload
    ):

        body = json.dumps(
            payload,
            ensure_ascii=False
        ).encode(
            "utf-8"
        )

        self.send_response(
            status
        )

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8"
        )

        self.send_header(
            "Content-Length",
            str(len(body))
        )

        self.end_headers()

        self.wfile.write(
            body
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print(
        "=========================================="
    )

    print(
        "PrivCanFed Dashboard Backend"
    )

    print(
        "=========================================="
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
        f"Root: {ROOT_DIR}"
    )

    print(
        "CORS: enabled"
    )

    print(
        "=========================================="
    )

    server = ThreadingHTTPServer(
        (
            HOST,
            PORT
        ),
        DashboardHandler
    )

    print(
        f"Server listening on "
        f"{HOST}:{PORT}"
    )

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print(
            "\nServer stopped."

        )

    finally:

        server.server_close()