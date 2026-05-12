<div align="center">    
 </div>

<div align="center">
<h1>ElaMamba</h1>
<h3>Empowering State Space Models with Elastic Structural Scanning for 3D Point Clouds</h3>

[Your Name](https://your-homepage.com/)<sup>1</sup> \*, [Co-author Name](https://co-author.com/)<sup>1</sup> \*, [Co-author Name]()<sup>2</sup>, and [Advisor Name]()<sup>1†</sup>

<sup>1</sup> Your University / Institute, <sup>2</sup> Collaborating Institute

(\*) Equal contribution. ($\dagger$) Corresponding author.

[![arXiv](https://img.shields.io/badge/Arxiv-Coming_Soon-b31b1b.svg?logo=arXiv)](#)
[![Project](https://img.shields.io/badge/Homepage-project-orange.svg?logo=googlehome)](#)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

## 📣 News

- **[May 2026]** 🚀 We release the official implementation of **ElaMamba**, including the complete Elastic Structural Scanning (ESS) module and pre-trained weights!
- **[May 2026]** The codebase has been fully refactored to support multi-GPU and single-GPU setups seamlessly using `torch.distributed.launch`.
- **[Coming Soon]** The paper will be available on arXiv shortly.

## 📝 Abstract

State Space Models (SSMs), particularly Mamba, offer linear complexity for sequence modeling, yet their translation to unordered 3D point clouds is hindered by the fundamental mismatch between 1D sequences and 3D geometries[cite: 60]. Current approaches heavily depend on rigid serialization techniques that disrupt spatial adjacency, ultimately degrading feature representation and long-range memory retention[cite: 61]. To bridge this gap, we introduce ElaMamba, a novel architecture designed to inherently maintain structural dependencies within the state space[cite: 62]. Breaking away from static scanning, ElaMamba employs a differentiable Spatial Micro-Deformation Generation mechanism that adaptively seeks out salient geometric structures[cite: 63]. This spatial elasticity is controlled by a novel Adaptive Regularization Loss, utilizing the $L_{2}$ norm of spatial offsets to enforce a rigid topological scan on clean datasets while permitting necessary deformations to navigate real-world noise[cite: 64]. Additionally, an Offset-Aware Dynamic Gating Mechanism is integrated to selectively modulate feature propagation, preventing the network from absorbing irrelevant background clutter during the deformation process[cite: 65]. Extensive evaluations confirm that ElaMamba seamlessly adapts to various sequence lengths, achieving state-of-the-art overall accuracies of 93.0\% on ModelNet40 and 88.34\% on the highly challenging ScanObjectNN dataset, substantially outperforming existing Transformer and Mamba-based baselines[cite: 66].

## 🌟 Key Innovations

* **Elastic Structural Scanning (ESS):** Dynamically optimizes the scanning sequence offsets to capture local geometric priors.
* **Dynamic Gating Mechanism:** Intelligently fuses features from different scanning directions based on structural salience.
* **Adaptive Regularization Loss ($\mathcal{L}_{reg}$):** A novel constraint applied directly to the ESS offset predictions, preventing sequence degeneration during early training phases.
* **Golden Sample Hunter:** A built-in evaluation tool to automatically capture and save hard-to-classify point clouds where ElaMamba succeeds over traditional baselines.

## 🏛️ Architecture Overview

<div align="center">    
 <img src="./figure/architecture.png" width = "888"  align=center />
 *Figure 1: The overall architecture of ElaMamba and the detailed design of the ESS module.*
</div>

## 📊 Main Results

| Task | Dataset | Config | Overall Acc. | Download (ckpt/log) |
| :---- | :---- | :---- |:-------:|:---:|
| Pre-training | ShapeNet | [pretrain.yaml](./cfgs/pretrain.yaml) | N.A. | [ckpt](https://github.com/lzq3367785771/Elamamba/releases/download/v1.0/pretrain.pth) |
| Classification | ModelNet40 | [finetune_modelnet.yaml](./cfgs/finetune_modelnet.yaml) | **93.0%** | [ckpt](https://github.com/lzq3367785771/Elamamba/releases/download/v1.0/modelnet_best.pth)|
| Classification | ScanObjectNN | [finetune_scan_hardest.yaml](./cfgs/finetune_scan_hardest.yaml) | **88.34%** | [ckpt](https://github.com/lzq3367785771/Elamamba/releases/download/v1.0/scanobjectnn_best.pth)|

*(Note: Click the links above to download the pre-trained weights and training logs. Links will be activated upon paper release.)*

## 🛠️ Getting Started

### 1. Environment Setup

We recommend using Conda to manage your environment. You can also refer to the detailed [`docs/environment.yml`](./docs/environment.yml).
```bash
conda create -n elamamba python=3.9.18 -y
conda activate elamamba
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt

# Compile PointNet2 operations
cd extensions/pointnet2_ops_lib
python setup.py install
cd ../../
```

### 2. Data Preparation

Please organize your datasets in the `data/` directory as follows:

- **ModelNet40:** `data/ModelNet/modelnet40_normal_resampled`
- **ScanObjectNN:** `data/ScanObjectNN`

**👉 See [`docs/DATASET.md`](./docs/DATASET.md) for detailed dataset instructions.**

### 3. Usage


**👉 See [`docs/USAGE.md`](./docs/USAGE.md) for full commands on training, fine-tuning, and evaluation.**


## 🤝 Acknowledgement

Our codebase is built upon several excellent open-source projects. We express our gratitude to the authors of:

- [PointMamba](#)
- [Point-MAE](#)
- [Mamba](#)

## 📖 Citation

If you find our method, code useful in your research, please consider giving us a star ⭐ and citing our paper:

```bibtex
@article{ElaMamba2026,
      title={ElaMamba: Enhancing 3D Point Cloud Analysis with Elastic Structural Scanning}, 
      author={Your Name and Co-authors},
      journal={arXiv preprint},
      year={2026}
}
```