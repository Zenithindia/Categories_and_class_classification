import os
import time
from typing import Dict
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.config import get_cfg
from src.utils import set_seed, get_device, ensure_dir
from src.datasets import make_loaders
from src.models import TimmClassifier
from src.evaluate import evaluate

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp: bool) -> float:
    model.train()
    running_loss = 0.0
    n = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        running_loss += loss.item() * bs
        n += bs

    return running_loss / max(n, 1)

def save_checkpoint(path: str, model: nn.Module, cfg: Dict, class_to_idx: Dict, metrics: Dict) -> None:
    ckpt = {
        "model_state": model.state_dict(),
        "cfg": cfg,
        "class_to_idx": class_to_idx,
        "metrics": metrics,
    }
    torch.save(ckpt, path)

def main(config_path: str = "config.yaml") -> None:
    cfg = get_cfg(config_path)

    set_seed(int(cfg["project"]["seed"]))
    device = get_device(cfg["train"]["device"])

    run_name = cfg["logging"]["run_name"]
    out_dir = os.path.join(cfg["logging"]["out_dir"], run_name)
    save_dir = cfg["logging"]["save_dir"]
    ensure_dir(out_dir)
    ensure_dir(save_dir)

    train_loader, val_loader, class_to_idx = make_loaders(
        train_dir=cfg["data"]["train_dir"],
        val_dir=cfg["data"]["val_dir"],
        image_size=int(cfg["train"]["image_size"]),
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
    )

    num_classes = int(cfg["data"]["num_classes"])
    model = TimmClassifier(
        backbone=cfg["model"]["backbone"],
        num_classes=num_classes,
        pretrained=bool(cfg["model"]["pretrained"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    use_amp = bool(cfg["train"]["mixed_precision"]) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_acc = -1.0
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
        metrics = evaluate(model, val_loader, device)
        dt = time.time() - t0

        msg = f"Epoch {epoch}: loss={tr_loss:.4f} val_acc={metrics['val_acc']:.4f} time={dt:.1f}s"
        print(msg)

        # Save best
        if bool(cfg["logging"]["save_best"]) and metrics["val_acc"] > best_acc:
            best_acc = metrics["val_acc"]
            best_path = os.path.join(save_dir, f"{run_name}_best.pt")
            save_checkpoint(best_path, model, cfg, class_to_idx, {"best_val_acc": best_acc})
            print(f"✅ Saved best checkpoint: {best_path}")

    # Save last
    last_path = os.path.join(save_dir, f"{run_name}_last.pt")
    save_checkpoint(last_path, model, cfg, class_to_idx, {"best_val_acc": best_acc})
    print(f"✅ Saved last checkpoint: {last_path}")

if __name__ == "__main__":
    main()