from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
class AlbumentationsWrapper:
    """
    Wraps torchvision ImageFolder samples with Albumentations transforms.
    ImageFolder returns PIL images; we convert to numpy BGR via cv2 for Albumentations.
    """
    def __init__(self, base_ds: ImageFolder, transform: Optional[A.Compose] = None):
        self.base_ds = base_ds
        self.transform = transform

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        pil_img, label = self.base_ds[idx]
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        if self.transform is not None:
            out = self.transform(image=img)
            img_t = out["image"]
        else:
            img_t = ToTensorV2()(image=img)["image"]
        return img_t, label

def build_transforms(image_size: int, train: bool) -> A.Compose:
    if train:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.1, rotate_limit=10, p=0.4),
            A.Normalize(),
            ToTensorV2()
        ])
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(),
        ToTensorV2()
    ])

def make_loaders(train_dir: str, val_dir: str, image_size: int, batch_size: int, num_workers: int):
    base_train = ImageFolder(train_dir)
    base_val = ImageFolder(val_dir)

    # Create wrapped datasets
    train_tf = build_transforms(image_size, train=True)
    val_tf = build_transforms(image_size, train=False)

    # Local import to avoid global dependency if user removes numpy
    import numpy as np
    train_ds = AlbumentationsWrapper(base_train, train_tf)
    val_ds = AlbumentationsWrapper(base_val, val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, base_train.class_to_idx