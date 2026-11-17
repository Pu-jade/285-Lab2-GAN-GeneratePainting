# WikiArt DCGAN

A small DCGAN for Homework 2. It generates 64 x 64 RGB paintings from 100-dimensional random noise using PyTorch.

## Setup

```bash
python -m pip install -r requirements.txt
```

Download the [WikiArt dataset](https://www.kaggle.com/datasets/steubk/wikiart) and extract its style folders under `data/wikiart_full/`. Images are loaded recursively. A fixed split uses 90% for training and holds out 10% for evaluation (seed 42).

## Run

```bash
python train.py --data data/wikiart_full --epochs 30 --workers 4 --output results_new
python evaluate.py --data data/wikiart_full --checkpoint results_new/model.pt --samples 2048
python generate.py --checkpoint results_new/model.pt --output results_new/new_samples.png
```

To fine-tune the existing model for 15 additional epochs with fresh Adam optimizers:

```bash
python train.py --data data/wikiart_full --init-from results_mps/model.pt --epochs 15 --lr 0.0001 --workers 4 --output results_finetune
```

Training automatically uses CUDA, Apple MPS, or CPU. Add `--device mps` to select Apple GPU explicitly. Use a new output directory for each training run.

Outputs include model weights, settings, the saved split, loss curves, generated images, and `metrics.json`. Evaluation follows the assignment's [PyTorch-Ignite tutorial](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/) using Ignite 0.5.5. Its default FID uses 1,000-dimensional classification probabilities and is not directly comparable with standard pooled-feature FID.

The completed 30-epoch local run is in `results_mps/`. The two-page report and LaTeX source are in `report/`.
