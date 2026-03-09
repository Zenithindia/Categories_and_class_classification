import json
from pathlib import Path

import torch
import timm
import torchvision.transforms as T
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_best_weight(model_dir: Path) -> Path:
    best_files = list(model_dir.glob("*_best.pt"))
    if not best_files:
        raise FileNotFoundError(f"No *_best.pt file found in {model_dir}")
    return best_files[0]


def _find_config_file(model_dir: Path) -> Path:
    candidates = list(model_dir.glob("config*.json"))
    if not candidates:
        raise FileNotFoundError(f"No config*.json file found in {model_dir}")
    return candidates[0]


def load_submodel(model_dir: str):
    model_dir = Path(model_dir)

    with open(model_dir / "class_to_idx.json", "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    config_path = _find_config_file(model_dir)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    model_name = cfg["model_name"]
    num_classes = cfg["num_classes"]

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes
    )

    weight_path = _find_best_weight(model_dir)
    state = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    return model, idx_to_class, cfg


def build_transform(image_size: int):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])


@torch.no_grad()
def predict_subclass_topk(model, idx_to_class, img: Image.Image, image_size: int, k: int = 3):
    transform = build_transform(image_size)
    x = transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    topk = torch.topk(probs, k=min(k, probs.numel()))

    results = []
    for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        results.append({
            "label": idx_to_class[int(idx)],
            "confidence": float(score)
        })

    return results