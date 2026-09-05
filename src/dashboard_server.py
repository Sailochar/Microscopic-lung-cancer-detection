"""Local API and static server for the PrivCanFed dashboard.

Run from the project root:
    python src/dashboard_server.py

Deployment:
    Render Web Service

Frontend:
    Vercel
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
# SERVER CONFIGURATION
# ============================================================

# IMPORTANT:
# Render requires the server to listen on 0.0.0.0.
HOST = "0.0.0.0"

# Render automatically supplies PORT.
# Local development defaults to 4173.
PORT = int(
    os.environ.get(
        "PORT",
        "4173"
    )
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

# Main production Vercel URL.
VERCEL_PRODUCTION_ORIGIN = (
    "https://microscopic-lung-cancer-detection.vercel.app"
)


# Exact local development origins.
ALLOWED_ORIGINS = {
    VERCEL_PRODUCTION_ORIGIN,

    "http://localhost:4173",
    "http://127.0.0.1:4173",

    "http://localhost:3000",
    "http://127.0.0.1:3000",
}


def is_allowed_origin(origin):
    """
    Check whether a browser Origin is allowed.

    Supports:
      - Production Vercel URL
      - Vercel deployment/preview URLs for this project
      - Local development
    """

    if not origin:
        return False

    # Exact known origins.
    if origin in ALLOWED_ORIGINS:
        return True

    try:
        parsed = urlparse(origin)

        scheme = parsed.scheme
        hostname = parsed.hostname

        if not hostname:
            return False

        # Allow HTTPS Vercel deployment URLs belonging
        # to this project.
        #
        # Example:
        # https://microscopic-lung-cancer-detection-1etdyxrf6.vercel.app
        #
        if (
            scheme == "https"
            and hostname.startswith(
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
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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
    Load a PyTorch checkpoint.

    Models are cached after the first load so subsequent
    requests do not reload the checkpoint.
    """

    # Return cached model if already loaded.
    if model_key in MODELS:
        return MODELS[model_key]

    # Security / validation:
    # Only allow models defined in CHECKPOINTS.
    if model_key not in CHECKPOINTS:
        model_key = "fedprox"

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
        f"[MODEL] Loading {model_key}"
    )

    print(
        f"[MODEL] Checkpoint: "
        f"{checkpoint_path}"
    )

    model = get_model(
        num_classes=len(
            CLASS_NAMES
        ),
        freeze_backbone=False,
        pretrained=False,
    )

    state = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    # Support checkpoints saved as:
    #
    # {
    #     "model_state_dict": ...
    # }
    #
    # as well as raw state dictionaries.
    if (
        isinstance(state, dict)
        and "model_state_dict" in state
    ):
        state = state[
            "model_state_dict"
        ]

    model.load_state_dict(
        state,
        strict=True,
    )

    model.to(DEVICE)

    model.eval()

    MODELS[
        model_key
    ] = model

    print(
        f"[MODEL] {model_key} loaded successfully"
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
# MICROSCOPY IMAGE VALIDATION
# ============================================================

def validate_microscopy_image(image):
    """
    Reject obvious out-of-domain images before the
    closed-set classifier runs.
    """

    sample = np.asarray(
        image.resize(
            (128, 128)
        ),
        dtype=np.float32,
    ) / 255.0

    # Percentage of very dark pixels.
    dark_ratio = float(
        np.all(
            sample < 0.12,
            axis=2,
        ).mean()
    )

    # Percentage of very bright pixels.
    bright_ratio = float(
        np.all(
            sample > 0.88,
            axis=2,
        ).mean()
    )

    # Estimate color variation.
    color_spread = (
        sample.max(axis=2)
        - sample.min(axis=2)
    )

    colored_ratio = float(
        (
            color_spread > 0.10
        ).mean()
    )

    # Estimate image luminance.
    luminance = sample.mean(
        axis=2
    )

    # Estimate contrast.
    contrast = float(
        luminance.std()
    )

    # Reject mostly black images.
    if (
        dark_ratio > 0.42
        and colored_ratio < 0.28
    ):
        raise ValueError(
            "Unsupported image: upload a stained "
            "microscopic lung-tissue image."
        )

    # Reject mostly white / blank images.
    if (
        bright_ratio > 0.72
        and contrast < 0.10
    ):
        raise ValueError(
            "Unsupported image: upload a stained "
            "microscopic lung-tissue image."
        )

    # Reject images with extremely low information.
    if (
        colored_ratio < 0.08
        or contrast < 0.035
    ):
        raise ValueError(
            "Unsupported image: upload a stained "
            "microscopic lung-tissue image."
        )


# ============================================================
# PREDICTION
# ============================================================

def predict(
    data_url,
    model_key
):
    """
    Run inference using the requested model.
    """

    # Decode image.
    image = decode_image(
        data_url
    )

    # Validate image.
    validate_microscopy_image(
        image
    )

    # Apply validation preprocessing.
    tensor = (
        VAL_TRANSFORM(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    # Load requested checkpoint.
    model = load_model(
        model_key
    )

    # Inference.
    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0].cpu().tolist()

    # Rank classes.
    ranked = sorted(
        zip(
            CLASS_NAMES,
            probabilities,
        ),
        key=lambda item: item[1],
        reverse=True,
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

        "device":
            "reported held-out results",

        "source":
            "pasted evaluation report",
    }

    return METRICS_CACHE


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


    # ========================================================
    # CORS HEADERS
    # ========================================================

    def add_cors_headers(self):
        """
        Add CORS headers for allowed Vercel/local origins.
        """

        origin = self.headers.get(
            "Origin"
        )

        if is_allowed_origin(
            origin
        ):

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


    # ========================================================
    # COMMON RESPONSE HEADERS
    # ========================================================

    def end_headers(self):

        # Do not cache API responses.
        self.send_header(
            "Cache-Control",
            "no-store",
        )

        # Add CORS.
        self.add_cors_headers()

        # Finish HTTP headers.
        super().end_headers()


    # ========================================================
    # CORS PREFLIGHT
    # ========================================================

    def do_OPTIONS(self):
        """
        Handle browser CORS preflight requests.

        The frontend sends:
            Content-Type: application/json

        Therefore browsers can send an OPTIONS request
        before the actual POST request.
        """

        path = urlparse(
            self.path
        ).path

        # Only allow preflight for API endpoints.
        if path not in (
            "/api/predict",
            "/api/metrics",
        ):
            self.send_error(
                404
            )
            return

        self.send_response(
            204
        )

        self.send_header(
            "Content-Length",
            "0",
        )

        self.end_headers()


    # ========================================================
    # POST /api/predict
    # ========================================================

    def do_POST(self):

        path = urlparse(
            self.path
        ).path

        # Only prediction endpoint accepts POST.
        if path != "/api/predict":

            self.send_error(
                404
            )

            return

        try:

            # Get request size.
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

            # Limit request size to 15 MB.
            # This prevents accidental huge uploads.
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

            # Read request body.
            raw_body = self.rfile.read(
                content_length
            )

            # Parse JSON.
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

            # Get image.
            image_data = payload.get(
                "image"
            )

            if not image_data:
                raise ValueError(
                    "No image was provided"
                )

            # Get selected model.
            model_key = payload.get(
                "model",
                "fedprox"
            )

            # Supported models.
            allowed_models = set(
                CHECKPOINTS.keys()
            )

            if (
                model_key
                not in allowed_models
            ):
                model_key = "fedprox"

            # Run prediction.
            result = predict(
                image_data,
                model_key
            )

            # Send result.
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
                        "Prediction failed on the backend"
                }
            )


    # ========================================================
    # GET /api/metrics
    # ========================================================

    def do_GET(self):

        path = urlparse(
            self.path
        ).path

        # API metrics endpoint.
        if path == "/api/metrics":

            try:

                result = get_metrics()

                self.send_json(
                    200,
                    result
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

        # Everything else is served normally.
        super().do_GET()


    # ========================================================
    # JSON RESPONSE
    # ========================================================

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
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
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
        f"Vercel production origin: "
        f"{VERCEL_PRODUCTION_ORIGIN}"
    )

    print(
        "CORS: enabled"
    )

    print(
        "=================================================="
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
        f"0.0.0.0:{PORT}"
    )

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print(
            "\nServer stopped."
        )

    finally:

        server.server_close()