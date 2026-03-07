from typing import Any, Dict
import yaml

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(
            f"Config file '{path}' is empty or invalid YAML. "
            "Open it and ensure it contains valid YAML keys."
        )
    if not isinstance(data, dict):
        raise ValueError(f"Config file '{path}' must load into a dict, got: {type(data)}")
    return data

def get_cfg(path: str) -> Dict[str, Any]:
    cfg = load_yaml(path)
    assert "data" in cfg and "train_dir" in cfg["data"], "Missing data.train_dir in config"
    assert "data" in cfg and "val_dir" in cfg["data"], "Missing data.val_dir in config"
    assert "train" in cfg and "epochs" in cfg["train"], "Missing train.epochs in config"
    return cfg