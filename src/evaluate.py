from typing import Dict
import torch
from sklearn.metrics import accuracy_score

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    return {"val_acc": float(acc)}