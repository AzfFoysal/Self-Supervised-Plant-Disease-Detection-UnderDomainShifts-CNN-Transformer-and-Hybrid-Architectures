import torch
import torch.nn as nn
from torchvision import models
import timm


class HybridCNNViTModel(nn.Module):
    """Hybrid CNN–ViT architecture with learnable softmax fusion.

    - CNN branch: ResNet50 backbone -> 2048-d pooled feature
    - ViT branch: DINOv2 ViT-Base (ViT-B/14) -> 768-d CLS feature
    - Fusion: learnable logits -> softmax -> weights sum to 1
    - Classifier: MLP over concatenated weighted features
    """

    def __init__(self, num_classes: int = 38, vit_model_name: str = "vit_base_patch14_dinov2", pretrained: bool = True,
                 mlp_hidden: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.vit_model_name = vit_model_name

        # CNN branch
        cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        cnn.fc = nn.Identity()
        self.cnn_branch = cnn
        self.cnn_dim = 2048

        # ViT branch (feature extractor)
        vit = timm.create_model(vit_model_name, pretrained=pretrained, num_classes=0, global_pool="")
        self.vit_branch = vit
        self.vit_dim = getattr(vit, "num_features", 768)

        # Learnable fusion logits -> softmax weights (alpha, beta)
        self.fusion_logits = nn.Parameter(torch.zeros(2))

        fused_dim = self.cnn_dim + self.vit_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes),
        )

    def fusion_weights(self):
        w = torch.softmax(self.fusion_logits, dim=0)
        return w[0], w[1]

    def forward(self, x_cnn: torch.Tensor, x_vit: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn_branch(x_cnn)  # (B, 2048)

        vit_feats = self.vit_branch.forward_features(x_vit)
        if vit_feats.dim() == 3:
            vit_feat = vit_feats[:, 0]  # CLS
        else:
            vit_feat = vit_feats

        alpha, beta = self.fusion_weights()
        fused = torch.cat([alpha * cnn_feat, beta * vit_feat], dim=1)
        return self.classifier(fused)
