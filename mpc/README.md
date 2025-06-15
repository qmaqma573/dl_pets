# MPC benchmark

```
cd benchmarks
python launcher.py --help
```
## plaintext baseline

```
python baseline.py
```

## Simulate two parties on one host
```
cd benchmarks
python launcher.py --multiprocess
```
### Default parameters
```
world size = 2
epochs = 20
start epoch = 0
batch size = 64
learning rate = 0.1
momentum = 0.99
seed = 42
loss function = cross entropy (ce)
```

### change parameters
example
```
python launcher.py --multiprocess --loss-func mse
```

## AWS Instances
### prerequirement
- AWS account
- aws cli configured

### steps
1. Deploy two aws instances (16 GB storage, t2.micro)
2. Install necessary packages on it(pip, crypten, requirements.txt)
3. Copy the training data to it (benchmarks/party0, benchmarks/party1)
4. Create a python virtual enviroment using `python -m venv venv; source venv/bin/activate`
5. Install all dependencies:
    - `pip install -r requirements.txt`
    - `pip install crypten-0.4.0-py3-none-any.whl`
5. Configure the inbound rules to allow the communication between two instances

After uploading all the necessary files, the (default) directory should look like:
```
-rw-r--r-- 1 ubuntu ubuntu 262592 May 20 10:37 crypten-0.4.0-py3-none-any.whl
drwxr-xr-x 2 ubuntu ubuntu   4096 May 20 10:35 party1/
-rw-r--r-- 1 ubuntu ubuntu    693 May 20 10:48 requirements.txt
drwxrwxr-x 7 ubuntu ubuntu   4096 May 20 10:57 venv
```

```
python3 script/aws_launcher.py --ssh_key_file=[SSH KEY FILE PATH] --region=[INSTANCE REGION] --instances=INSTANCE_ID1,INSTANCE_ID2  --prepare_cmd="source venv/bin/activate" --aux_files=benchmarks/mpc_benchmarks.py,multiprocess_launcher.py  benchmarks/launcher.py --epochs 20 --loss-func ce
```