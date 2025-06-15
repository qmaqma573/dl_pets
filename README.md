# Privacy-Preserving Machine Learning in the Training Phase: A Comparative Study of Privacy Enhancing Technologies

## Overview
This repository contains the source code and experiments for the thesis: **Privacy-Preserving Machine Learning in the Training Phase: A Comparative Study of Privacy Enhancing Technologies**. The project compares different Privacy Enhancing Technologies (PETs) applied to machine learning training.

## Directory Structure

```
data/extracted_features    # Extracted feature data
data/models                # Model training history (loss, accuracy) and parameters
HE_finetuning/             # Homomorphic Encryption (HE) experiments
mpc/                       # Secure Multi-Party Computation (MPC) experiments
lib/                       # Python package for crypten
plaintext_finetuning/      # Baseline (plaintext) for HE experiments
result_summary.py          # Script to generate graphs from data/models/
```

---

## Setup Instructions

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install lib/crypten-0.4.0-py3-none-any.whl
   ```

---

## Homomorphic Encryption Experiments

- Training parameters are configured in `config.py`.
- Default: 20 training epochs.

### Run Training

- **Cross-entropy loss:**
  ```bash
  python he_ce_train.py
  ```
- **MSE loss:**
  ```bash
  python he_mse_train.py
  ```

---

## Secure Multi-Party Computation Experiments

See [mpc/README.md](mpc/README.md) for detailed instructions.

---

## Results Visualization

- Use `result_summary.py` to generate graphs based on results in `data/models/`.

---