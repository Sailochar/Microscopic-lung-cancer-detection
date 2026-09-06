"""Generate a Grad-CAM explanation for one image and one checkpoint.

Run from the project root:
    python src/explain_prediction.py --image external_data/lung_aca/sample.jpg \
        --checkpoint checkpoints/global_fedprox_model.pth \
        --output results/sample_gradcam.png

Use the same PRIVCANFED_STAIN_NORMALIZATION setting used during training.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from model import get_model
from preprocessing import VAL_TRANSFORM

CLASS_NAMES = ["lung_aca", "lung_n", "lung_scc"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint):
    model = get_model(num_classes=3, freeze_backbone=False, pretrained=False)
    state = torch.load(checkpoint, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/gradcam.png")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    activations = []
    gradients = []
    target_layer = model.backbone.features[-1]

    def save_activation(_, __, output):
        activations.append(output)

    def save_gradient(_, __, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(save_activation)
    backward_handle = target_layer.register_full_backward_hook(save_gradient)
    try:
        image = Image.open(args.image).convert("RGB")
        tensor = VAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        class_index = int(probabilities.argmax())
        model.zero_grad(set_to_none=True)
        logits[0, class_index].backward()

        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        heatmap = F.relu((weights * activations[0]).sum(dim=1)).squeeze(0)
        heatmap = heatmap / heatmap.max().clamp_min(1e-8)
        heatmap = F.interpolate(
            heatmap[None, None], size=image.size[::-1], mode="bilinear",
            align_corners=False
        ).squeeze().detach().cpu().numpy()
    finally:
        forward_handle.remove()
        backward_handle.remove()

    output_path = os.path.join(ROOT_DIR, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Input")
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(heatmap, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.title(f"{CLASS_NAMES[class_index]} ({probabilities[class_index].item():.1%})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Prediction: {CLASS_NAMES[class_index]}")
    print(f"Probabilities: {[round(p, 4) for p in probabilities.tolist()]}")
    print(f"Grad-CAM: {output_path}")


if __name__ == "__main__":
    main()