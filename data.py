import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def find_images(root):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in extensions)


def make_split(root, seed=42):
    paths = find_images(root)
    if len(paths) < 3:
        raise ValueError(f"Need at least 3 images in {root}.")
    random.Random(seed).shuffle(paths)
    n_test = max(1, int(len(paths) * 0.1))
    return paths[n_test:], paths[:n_test]


class PaintingDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))
