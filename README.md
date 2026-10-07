# Extinction-Inspired Morphological Residual Blocks (MER-Block)

This repository contains the official PyTorch implementation of the **MER-Block**, an end-to-end differentiable morphological residual block. It evaluates the impact of multiscale contrast, top-hat, and geometric-mean descriptors on intermediate CNN representations using DermaMNIST and EuroSAT datasets.

## 📂 Repository Structure

```text
├── mer_core.py             # Core logic: LocalExtinctionBlock2D, model definitions, and training loop.
├── run_eurosat.py          # Execution script for the EuroSAT 128x128 experiments.
├── run_dermamnist.py       # Execution script for the DermaMNIST 128x128 experiments.
├── README.md               # This file.
```

## 📖 Citation

Gonçalves Silva, Alexandre. *Extinction-Inspired Morphological Residual Blocks for End-to-End Image Classification*. Available at SSRN: [https://ssrn.com/abstract=7557216] or [http://dx.doi.org/10.2139/ssrn.7557216]

```text
@article{Silva2026MERBlock,
  title={Extinction-Inspired Morphological Residual Blocks for End-to-End Image Classification},
  author={Silva, Alexandre Gon{\c{c}}alves},
  journal={Available at SSRN 7557216},
  year={2026},
  doi={10.2139/ssrn.7557216},
  url={[https://ssrn.com/abstract=7557216](https://ssrn.com/abstract=7557216)}
}
```
