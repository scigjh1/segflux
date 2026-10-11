# SegFlux

PyTorch implementation of **SegFlux: Physical Morphology Flow and Analytical Routing for Volumetric Medical Image Segmentation**.

## Structure

- `segflux/models/` — model architecture and geometric modules
- `segflux/losses/` — training objectives
- `segflux/data/` — dataset utilities and split handling
- `segflux/metrics/` — evaluation metrics
- `scripts/` — training and evaluation scripts
- `configs/` — AMOS22, LiTS, and ACDC configurations

## Usage

Update the data and output paths in `configs/`.

```bash
python run.py configs/amos22.json train
python run.py configs/amos22.json evaluate
```

Use `configs/lits.json` or `configs/acdc.json` for the corresponding dataset.
