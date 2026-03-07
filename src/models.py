import timm
import torch
import torch.nn as nn

class TimmClassifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)