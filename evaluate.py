import argparse
import json
import random
from pathlib import Path

import ignite
import torch
from ignite.metrics import FID, InceptionScore
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from data import PaintingDataset, make_split
from models import Generator, choose_device


def prepare_images(images):
    # The GAN uses [-1, 1]. Convert back before the tutorial's PIL resize.
    resized = []
    for image in images.detach().cpu():
        image = TF.to_pil_image(((image + 1) / 2).clamp(0, 1))
        image = image.resize((299, 299), Image.Resampling.BILINEAR)
        resized.append(TF.to_tensor(image))
    return torch.stack(resized)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="WikiArt image directory; defaults to training directory")
    parser.add_argument("--checkpoint", default="results/model.pt")
    parser.add_argument("--split", help="Saved split.json; defaults to checkpoint directory")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", help="Metrics JSON; defaults to checkpoint directory")
    args = parser.parse_args()
    if args.samples < 2 or args.batch_size < 1:
        parser.error("samples must be at least 2 and batch-size must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    root = Path(args.data or config["data"]).expanduser().resolve()
    split_path = Path(args.split) if args.split else Path(args.checkpoint).parent / "split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text())
        paths = [root / path for path in split["test"]]
    elif args.split:
        raise FileNotFoundError(f"Split file not found: {split_path}")
    else:
        _, paths = make_split(root, seed)
    random.Random(seed).shuffle(paths)
    paths = paths[: args.samples]
    if len(paths) < 2:
        raise ValueError("Need at least two held-out images to compute FID.")
    if len(paths) < args.samples:
        print(f"Using all {len(paths)} held-out images (requested {args.samples}).")

    device = choose_device(args.device)
    model = Generator(config["nz"], config["ngf"]).to(device)
    model.load_state_dict(checkpoint["generator"])
    model.eval()
    loader = DataLoader(PaintingDataset(paths), batch_size=args.batch_size)

    # Follow the assignment's Ignite tutorial, including its default features.
    # MPS cannot accumulate the float64 statistics used by these metrics.
    metric_device = device if device.type == "cuda" else torch.device("cpu")
    fid = FID(device=metric_device)
    inception_score = InceptionScore(device=metric_device)
    done = 0
    with torch.no_grad():
        for real in loader:
            noise = torch.randn(len(real), config["nz"], 1, 1, device=device)
            fake = prepare_images(model(noise)).to(metric_device)
            real = prepare_images(real).to(metric_device)
            fid.update((fake, real))
            inception_score.update(fake)
            done += len(real)
            print(f"Evaluated {done}/{len(paths)} images", flush=True)

    results = {
        "FID": float(fid.compute()),
        "IS": float(inception_score.compute()),
        "samples": len(paths),
        "seed": seed,
        "reference_split": "held-out 10% image split",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "implementation": "Ignite default 1000-output Inception; tutorial PIL preprocessing",
        "note": "FID uses classifier probabilities, not standard 2048-feature benchmark FID. Lower FID and higher IS are better; small samples are noisy.",
        "ignite_version": ignite.__version__,
    }
    output = Path(args.output) if args.output else Path(args.checkpoint).parent / "metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
