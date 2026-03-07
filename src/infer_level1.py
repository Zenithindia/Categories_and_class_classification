import json
from typing import Tuple, Dict, Any, List
from pathlib import Path

import torch
import timm
from PIL import Image
import torchvision.transforms as T


def load_artifacts(model_dir: str) -> Tuple[torch.nn.Module, Dict[int, str], Dict[str, Any]]:
    model_dir = Path(model_dir)

    class_to_idx = json.loads((model_dir / "class_to_idx.json").read_text())
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    cfg = json.loads((model_dir / "config_level1.json").read_text())
    model_name = cfg["model_name"]
    num_classes = cfg["num_classes"]

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    # Load weights (best)
    state = torch.load(model_dir / "level1_model_best.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model, idx_to_class, cfg


def build_preprocess(image_size: int):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


@torch.no_grad()
def predict_topk(model: torch.nn.Module, idx_to_class: Dict[int, str], img: Image.Image, image_size: int, k: int = 3):
    x = build_preprocess(image_size)(img.convert("RGB")).unsqueeze(0)  # [1,3,H,W]
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)  # [C]

    topk = torch.topk(probs, k=min(k, probs.numel()))
    results = []
    for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        results.append({"label": idx_to_class[int(idx)], "confidence": float(score)})

    return results