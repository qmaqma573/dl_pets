# Secure Multi-Party Computation (MPC) Benchmark

This directory contains experiments and benchmarks for Secure Multi-Party Computation (MPC) as part of the thesis project on privacy-preserving machine learning.

---

## Overview
- Implements MPC-based training and evaluation for privacy-preserving machine learning.
- Includes plaintext baseline for comparison.
- Supports both local (single host) and distributed (AWS) execution.

---

## Quick Start

### 1. Run Plaintext Baseline

```bash
python baseline.py
```

### 2. Run MPC Benchmark (Simulate Two Parties on One Host)
```bash
cd benchmarks
python launcher.py --multiprocess
```

#### Default Parameters
- World size: 2
- Epochs: 20
- Start epoch: 0
- Batch size: 64
- Learning rate: 0.1
- Momentum: 0.99
- Seed: 42
- Loss function: cross entropy (ce)

#### Change Parameters (Example)
```bash
python launcher.py --multiprocess --loss-func mse
```

---

## AWS Deployment

### Prerequisites
- AWS account
- AWS CLI configured

### Steps
1. **Deploy two AWS instances** (e.g., t2.micro, 16 GB storage)
2. **Install dependencies** on each instance:
   - Python, pip
   - `pip install -r requirements.txt`
   - `pip install crypten-0.4.0-py3-none-any.whl`
3. **Copy training data** to each instance (`benchmarks/party0`, `benchmarks/party1`)
4. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
5. **Configure security groups** to allow communication between the two instances
6. **Verify directory structure** (example):
   ```
   -rw-r--r-- 1 ubuntu ubuntu 262592 May 20 10:37 crypten-0.4.0-py3-none-any.whl
   drwxr-xr-x 2 ubuntu ubuntu   4096 May 20 10:35 party1/
   -rw-r--r-- 1 ubuntu ubuntu    693 May 20 10:48 requirements.txt
   drwxrwxr-x 7 ubuntu ubuntu   4096 May 20 10:57 venv/
   ```

### Launch Distributed Training
```bash
python3 script/aws_launcher.py \
  --ssh_key_file=[SSH KEY FILE PATH] \
  --region=[INSTANCE REGION] \
  --instances=INSTANCE_ID1,INSTANCE_ID2 \
  --prepare_cmd="source venv/bin/activate" \
  --aux_files=benchmarks/mpc_benchmarks.py,multiprocess_launcher.py \
  benchmarks/launcher.py --epochs 20 --loss-func ce
```

---

## Help
- For more details on parameters, run:
  ```bash
  python launcher.py --help
  ```
- For any issues, please open an issue in the main repository.
