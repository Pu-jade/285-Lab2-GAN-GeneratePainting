import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from models import Generator, choose_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/model.pt")
    parser.add_argument("--output", default="results/new_samples.png")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    generator = Generator(config["nz"], config["ngf"]).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(args.samples, config["nz"], 1, 1, device=device)
        images = generator(noise).cpu()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, output, nrow=8, normalize=True, value_range=(-1, 1))
    print(f"Saved {args.samples} images to {output}")


if __name__ == "__main__":
    main()
