import torch
import argparse
from PIL import Image
from torchvision import transforms

from models.resnet50_model import ResNet50Classifier
from models.vit_dino_model import ViTClassifier
from models.hybrid_cnn_vit_model import HybridCNNViTModel
from utils.augmentation import val_transform_cnn, val_transform_vit
from utils.grad_cam import generate_gradcam, overlay_cam_on_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualization for a model and image")
    parser.add_argument("--model", choices=["resnet", "vit", "hybrid"], required=True, help="Model type")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pth)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="gradcam_output.png", help="Output image path")
    args = parser.parse_args()
    # Load model and weights
    num_classes = 38
    if args.model == "resnet":
        model = ResNet50Classifier(num_classes=num_classes, pretrained=False); model_type = "resnet"
    elif args.model == "vit":
        model = ViTClassifier(num_classes=num_classes, pretrained=False); model_type = "vit"
    else:
        model = HybridCNNViTModel(num_classes=num_classes); model_type = "hybrid"
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()
    # Load and preprocess image
    image = Image.open(args.image).convert("RGB")
    if args.model == "resnet":
        img_tensor = val_transform_cnn(image).unsqueeze(0)
        cam = generate_gradcam(model, img_tensor, model_type="resnet")
    elif args.model == "vit":
        img_tensor = val_transform_vit(image).unsqueeze(0)
        cam = generate_gradcam(model, img_tensor, model_type="vit")
    else:
        img_cnn = val_transform_cnn(image)
        img_vit = val_transform_vit(image)
        cam_cnn = generate_gradcam(model.cnn_branch, img_cnn.unsqueeze(0), model_type="resnet")
        cam_vit = generate_gradcam(model.vit_branch, img_vit.unsqueeze(0), model_type="vit")
        cam = (cam_cnn + cam_vit) / 2.0
    overlay = overlay_cam_on_image(image, cam)
    overlay.save(args.output)
    print(f"Grad-CAM visualization saved to {args.output}")
