# Preparation

## Environment

This codebase was tested with the following environment configurations. It may work with other versions.
- Ubuntu 20.04
- CUDA  11.7
- Python 3.9
- PyTorch 2.0.0+ / 1.13.1

## Installation

We recommend using Anaconda for the installation process:

```shell
# Clone the repository
git clone [https://github.com/lzq3367785771/Elamamba.git](https://github.com/lzq3367785771/Elamamba.git)
cd Elamamba

# Create virtual env and install PyTorch
conda create -n elamamba python=3.9 -y
conda activate elamamba

# Install PyTorch (Adjust CUDA version based on your hardware)
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Install basic required packages
pip install -r requirements.txt

# Chamfer Distance & emd
cd ./extensions/chamfer_dist && python setup.py install --user
cd ../emd && python setup.py install --user
cd ../../

# PointNet++
pip install "git+[https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib](https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib)"

# GPU kNN
pip install --upgrade [https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl](https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl)

# Mamba
pip install causal-conv1d>=1.2.0
pip install mamba-ssm
```

You can double-check the `environment.yml` file for the same requirements.

## Training & Evaluation

### 1. Pre-training
To pre-train the ElaMamba backbone on the ShapeNet dataset:

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/pretrain.yaml --exp_name elamamba_pretrain
```

### 2. Classification on ModelNet40
To fine-tune the model from a pre-trained backbone:

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --finetune_model --config cfgs/finetune_modelnet.yaml --ckpts <path/to/pre-trained/model.pth> --exp_name elamamba_modelnet_finetune
```

### 3. Classification on ScanObjectNN
For different ScanObjectNN variants, replace the config file accordingly (e.g., `finetune_scan_hardest.yaml`, `finetune_scan_objbg.yaml`):

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --finetune_model --config cfgs/finetune_scan_hardest.yaml --ckpts <path/to/pre-trained/model.pth> --exp_name elamamba_scan_hardest_finetune
```

### 4. Evaluation (Testing)
To evaluate a trained checkpoint on ModelNet40:

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --test --config cfgs/finetune_modelnet.yaml --ckpts <path/to/best_checkpoint.pth> --exp_name evaluate_modelnet_best
```

To evaluate a trained checkpoint on ScanObjectNN:

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --test --config cfgs/finetune_scan_hardest.yaml --ckpts <path/to/best_checkpoint.pth> --exp_name evaluate_scan_best
```
