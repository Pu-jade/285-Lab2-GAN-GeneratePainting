import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import PaintingDataset, make_split
from models import Discriminator, Generator, choose_device, init_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/wikiart")
    parser.add_argument("--output", default="results")
    parser.add_argument("--init-from", help="Fine-tune saved G/D weights with fresh Adam optimizers.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Use a smaller training subset; 0 uses all training images.")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.nz, args.ngf, args.ndf) < 1:
        parser.error("epochs, batch-size and model dimensions must be positive")
    if args.workers < 0 or args.max_images < 0 or args.lr <= 0:
        parser.error("workers/max-images must be nonnegative and lr must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    output = Path(args.output)
    if (output / "model.pt").exists():
        raise FileExistsError("This output already has a model. Use a new --output folder.")
    (output / "samples").mkdir(parents=True, exist_ok=True)
    root = Path(args.data).resolve()
    checkpoint = None
    start_epoch = 0
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=True)
        old_config = checkpoint["config"]
        for name in ("nz", "ngf", "ndf", "seed"):
            if getattr(args, name) != old_config[name]:
                raise ValueError(f"{name} must match the initial checkpoint ({old_config[name]}).")
        split_path = Path(args.init_from).parent / "split.json"
        split = json.loads(split_path.read_text())
        train_paths = [root / path for path in split["train"]]
        test_paths = [root / path for path in split["test"]]
        start_epoch = checkpoint["epoch"]
    else:
        train_paths, test_paths = make_split(root, args.seed)
    if args.max_images:
        train_paths = train_paths[:args.max_images]
    dataset = PaintingDataset(train_paths)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers)

    config = vars(args).copy()
    config.update(image_size=64, start_epoch=start_epoch,
                  optimizer_initialization="fresh Adam", train_images=len(train_paths),
                  test_images=len(test_paths), actual_device=str(device))
    (output / "config.json").write_text(json.dumps(config, indent=2))
    split = {"train": [str(p.relative_to(root)) for p in train_paths],
             "test": [str(p.relative_to(root)) for p in test_paths]}
    (output / "split.json").write_text(json.dumps(split, indent=2))

    generator = Generator(args.nz, args.ngf).to(device)
    discriminator = Discriminator(args.ndf).to(device)
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    if checkpoint is not None:
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        print(f"Initialized G/D from epoch {start_epoch}; using fresh Adam optimizers.", flush=True)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)
    preview = torch.stack([dataset[i] for i in range(min(64, len(train_paths)))])
    save_image(preview, output / "real.png", nrow=8, normalize=True, value_range=(-1, 1))
    history = []
    print(f"Device: {device}, train: {len(train_paths)}, held out: {len(test_paths)}", flush=True)

    with (output / "loss.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "generator_loss", "discriminator_loss"])
        for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
            start = time.time()
            generator.train()
            discriminator.train()
            total_g, total_d, count = 0.0, 0.0, 0
            for real in loader:
                real = real.to(device)
                n = real.size(0)
                ones = torch.ones(n, device=device)
                zeros = torch.zeros(n, device=device)
                noise = torch.randn(n, args.nz, 1, 1, device=device)
                fake = generator(noise)

                opt_d.zero_grad()
                loss_d = (loss_fn(discriminator(real), ones)
                          + loss_fn(discriminator(fake.detach()), zeros))
                loss_d.backward()
                opt_d.step()

                # Keep gradients through D to G, but do not update D in this step.
                discriminator.requires_grad_(False)
                opt_g.zero_grad()
                loss_g = loss_fn(discriminator(fake), ones)
                loss_g.backward()
                opt_g.step()
                discriminator.requires_grad_(True)

                total_g += loss_g.item() * n
                total_d += loss_d.item() * n
                count += n

            avg_g, avg_d = total_g / count, total_d / count
            history.append((avg_g, avg_d))
            writer.writerow([epoch, avg_g, avg_d])
            file.flush()
            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_noise).cpu()
            save_image(samples, output / "samples" / f"epoch_{epoch:03d}.png",
                       nrow=8, normalize=True, value_range=(-1, 1))
            save_image(samples, output / "generated.png", nrow=8,
                       normalize=True, value_range=(-1, 1))
            torch.save({"generator": generator.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "config": config, "epoch": epoch}, output / "model.pt")
            print(f"Epoch {epoch}/{start_epoch + args.epochs}: G={avg_g:.4f}, D={avg_d:.4f}, "
                  f"{time.time() - start:.1f}s", flush=True)

    plt.figure(figsize=(7, 3))
    plt.plot(range(start_epoch + 1, start_epoch + args.epochs + 1), [x[0] for x in history], label="Generator")
    plt.plot(range(start_epoch + 1, start_epoch + args.epochs + 1), [x[1] for x in history], label="Discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "loss.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
