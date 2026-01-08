import math
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
import cv2

try:
    from pytorch_grad_cam import GradCAM
except Exception:
    GradCAM = None


def _reshape_transform_vit(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape ViT token embeddings to a 2D spatial map for Grad-CAM.

    Input: (B, N, C) where N includes CLS token. We drop CLS and reshape remaining tokens to (B, C, H, W).
    """
    if tensor.dim() != 3:
        raise ValueError(f"Expected (B, N, C), got {tensor.shape}")
    tensor = tensor[:, 1:, :]
    n_tokens = tensor.shape[1]
    h = w = int(math.sqrt(n_tokens))
    if h * w != n_tokens:
        # Fallback: try to infer rectangular
        h = int(round(math.sqrt(n_tokens)))
        w = n_tokens // h
    result = tensor.reshape(tensor.shape[0], h, w, tensor.shape[2])
    result = result.permute(0, 3, 1, 2).contiguous()
    return result


def generate_gradcam(model, input_tensor: torch.Tensor, model_type: str = "resnet") -> np.ndarray:
    """Compute a Grad-CAM heatmap.

    Supports:
      - resnet: torchvision resnet50 or wrapper containing `.model`
      - vit: timm ViT (DINOv2) or wrapper containing `.backbone`

    Returns:
      heatmap: (H, W) float32 array in [0,1]
    """
    if GradCAM is None:
        raise ImportError("pytorch-grad-cam is required for Grad-CAM. Please install pytorch-grad-cam.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    input_tensor = input_tensor.to(device)

    if model_type == "resnet":
        # unwrap
        m = getattr(model, "model", model)
        target_layers = [m.layer4[-1]]
        cam = GradCAM(model=m, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        grayscale_cam = cam(input_tensor=input_tensor)[0]
    elif model_type == "vit":
        m = getattr(model, "backbone", model)
        # timm ViT: use last block norm as target layer
        target_layers = [m.blocks[-1].norm1]
        cam = GradCAM(
            model=m,
            target_layers=target_layers,
            use_cuda=torch.cuda.is_available(),
            reshape_transform=_reshape_transform_vit,
        )
        grayscale_cam = cam(input_tensor=input_tensor)[0]
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    grayscale_cam = np.clip(grayscale_cam, 0, None)
    if grayscale_cam.max() > 0:
        grayscale_cam = grayscale_cam / (grayscale_cam.max() + 1e-8)
    return grayscale_cam.astype(np.float32)


def overlay_cam_on_image(img: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Overlay a CAM heatmap onto an image."""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    heatmap_size = (cam.shape[1], cam.shape[0])
    img_resized = img.resize(heatmap_size)
    img_np = np.array(img_resized).astype(np.float32)

    cam_uint8 = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET).astype(np.float32)

    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)
