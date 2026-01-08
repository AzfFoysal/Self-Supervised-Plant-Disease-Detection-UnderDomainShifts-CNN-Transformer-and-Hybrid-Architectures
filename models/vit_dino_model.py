import torch
import torch.nn as nn
import timm

class ViTClassifier(nn.Module):
    """DINOv2 ViT-Base classifier.

    Uses timm model: vit_base_patch14_dinov2 (ViT-B/14, 768-dim).
    """

    def __init__(self, num_classes: int = 38, pretrained: bool = True, model_name: str = "vit_base_patch14_dinov2"):
        super().__init__()
        self.model_name = model_name
        # num_classes=0 -> feature extractor
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="")
        feat_dim = getattr(self.backbone, "num_features", 768)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # timm ViT returns (B, tokens, C) if global_pool="".
        feats = self.backbone.forward_features(x)
        # Use CLS token
        if feats.dim() == 3:
            cls = feats[:, 0]
        else:
            cls = feats
        return self.classifier(cls)
