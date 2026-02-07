import os
from torch.utils.data import DataLoader
from torchvision import datasets

from utils.preprocessing import get_transformss

def get_datasets(
    data_dir,
    img_size = 224,
    preprocess_config = None
):
    if preprocess_config is None:
        preprocess_config = {}

    train_transform = get_transformss(
        img_size=img_size,
        train = True,
        **preprocess_config
    )

    test_transform = get_transformss(
        img_size = img_size,
        train = False,
        **preprocess_config
    )

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transform=test_transform
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "test"),
        transform=test_transform
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    data_dir,
    batch_size=16,
    img_size=224,
    preprocess_config=None,
    num_workers=0
):
    train_ds, val_ds, test_ds = get_datasets(
        data_dir=data_dir,
        img_size=img_size,
        preprocess_config=preprocess_config
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_ds.classes


if __name__ == "__main__":
    DATA_DIR = "../lung_ct_split"

    preprocess_cfg = {
        "windowing": True,
        "clahe": True,
        "median": True,
        "sharpen_flag": True,
        "norm_type": "minmax"
    }

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=8,
        preprocess_config=preprocess_cfg
    )

    print("Classes:", classes)
    print("Train batches:", len(train_loader))