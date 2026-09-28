# Extinction-Inspired Morphological Residual Blocks (MER-Block)

This repository contains the official PyTorch implementation of the **MER-Block**, an end-to-end differentiable morphological residual block. It evaluates the impact of multiscale contrast, top-hat, and geometric-mean descriptors on intermediate CNN representations using DermaMNIST and EuroSAT datasets.

## 📂 Repository Structure

```text
├── mer_core.py             # Core logic: LocalExtinctionBlock2D, model definitions, and training loop.
├── run_eurosat.py          # Execution script for the EuroSAT 128x128 experiments.
├── run_dermamnist.py       # Execution script for the DermaMNIST 128x128 experiments.
├── README.md               # This file.