<div align="center">    
 </div>

<div align="center">
<h1>ElaMamba</h1>
<h3>Empowering State Space Models with Elastic Structural Scanning for 3D Point Clouds</h3>

[Zhiqiang Li]()<sup>1</sup>\*, [Qi Liu]()<sup>1</sup>\*, [Yufei Li]()<sup>2</sup>, [Qiang Feng]()<sup>3</sup>, [Shiyang Li]()<sup>1</sup>, [Xiyue Wang]()<sup>1</sup>, [Yining Sun]()<sup>1</sup>, [Xin Liu]()<sup>1</sup>, [Mengfan Ma]()<sup>1</sup>, [Yinghao Sun]()<sup>1</sup>, [Lei Yang]()<sup>1</sup>, [Xiaohong Yu]()<sup>1</sup>, [Chenfei Shi]()<sup>1</sup>, [Yixuan Ding]()<sup>1</sup>, and [Lin Da]()<sup>1†</sup>

<sup>1</sup> School of Mathematical Sciences, Inner Mongolia University <br>
<sup>2</sup> School of Computer Science and Communication Engineering, Northeastern University at Qinhuangdao <br>
<sup>3</sup> School of Electronic and Information Engineering, Inner Mongolia University 

(\*) Equal contribution. (†) Corresponding author.

[![arXiv](https://img.shields.io/badge/Arxiv-Coming_Soon-b31b1b.svg?logo=arXiv)](#)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

## 📣 News

- **[May 2026]** 🚀 We release the official implementation of **ElaMamba**, including the complete Elastic Structural Scanning (ESS) module and pre-trained weights!
- **[Coming Soon]** The paper will be available on arXiv shortly.

## 📝 Abstract

State Space Models (SSMs), particularly Mamba, offer linear complexity for sequence modeling, yet their translation to unordered 3D point clouds is hindered by the fundamental mismatch between 1D sequences and 3D geometries. Current approaches heavily depend on rigid serialization techniques that disrupt spatial adjacency, ultimately degrading feature representation and long-range memory retention. To bridge this gap, we introduce ElaMamba, a novel architecture designed to inherently maintain structural dependencies within the state space. 
Breaking away from static scanning, ElaMamba employs an Elastic Structural Scanning (ESS) mechanism to adaptively seek out salient geometric structures via differentiable micro-deformations. This spatial elasticity is controlled by a novel Adaptive Regularization Loss, utilizing the $L_{2}$ norm of spatial offsets to enforce a rigid topological scan on clean datasets while permitting necessary deformations to navigate real-world noise. Additionally, an Offset-Aware Dynamic Gating module is integrated to selectively modulate feature propagation, preventing the network from absorbing irrelevant background clutter. 
Extensive evaluations confirm that ElaMamba effectively overcomes the vulnerability of rigid serialization, achieving highly competitive overall accuracies of 93.0\% on ModelNet40 and 88.34\% on the highly challenging ScanObjectNN dataset, substantially outperforming existing Transformer and Mamba-based baselines.

## 🌟 Key Innovations

* **Elastic Structural Scanning (ESS):** Dynamically optimizes the scanning sequence offsets to capture local geometric priors.
* **Dynamic Gating Mechanism:** Intelligently fuses features from different scanning directions based on structural salience.
* **Adaptive Regularization Loss ($\mathcal{L}_{reg}$):** A novel constraint applied directly to the ESS offset predictions, preventing sequence degeneration during early training phases.
* **Golden Sample Hunter:** A built-in evaluation tool to automatically capture and save hard-to-classify point clouds where ElaMamba succeeds over traditional baselines.

## 🏛️ Architecture Overview

<div align="center">
  <img src="./figure/visio_elamamba.png" width="888" />
</div>

<p align="justify">
  Figure 1: The overall architecture of the proposed <b>ElaMamba</b>. Given a raw 3D point cloud, initial geometric features and center coordinates are extracted via FPS and KNN tokenization. The rigidly indexed sequence is then processed by four core micro-modules: <b>(A) Elastic Structural Scanning (ESS)</b> adaptively predicts physical micro-deformation offsets ($\Delta \mathbf{P}$) to bypass geometric voids while preserving the macroscopic Hilbert prior; <b>(B) Offset-Aware Dynamic Gate</b> modulates feature propagation by autonomously suppressing irrelevant background noise based on deformation costs; <b>(C) Spatial Aware Indicator</b> injects deformed positional cues into the purified sequence; and <b>(D) State Space Model (SSM)</b> performs linear-time global context modeling via MixerModel stacks. The entire framework is governed by an <b>Adaptive Regularization Loss</b> ($\mathcal{L}_{reg}$) to dynamically balance spatial elasticity and structural rigidity.
</p>

## 📊 Main Results

| Task | Dataset | Config | Overall Acc. | Download (ckpt/log) |
| :---- | :---- | :---- |:-------:|:---:|
| Pre-training | ShapeNet | [pretrain.yaml](./cfgs/pretrain.yaml) | N.A. | [ckpt](https://github.com/lzq3367785771/Elamamba/releases/download/v1.0/pretrain.pth) |
| Classification | ModelNet40 | [finetune_modelnet.yaml](./cfgs/finetune_modelnet.yaml) | **93.0%** | [ckpt](https://github.com/lzq3367785771/Elamamba/releases/download/v1.0/modelnet_best.pth)|
| Classification | ScanObjectNN | [finetune_scan_hardest.yaml](./cfgs/finetune_scan_hardest.yaml) | **88.34%** | [ckpt](https://github.com/lzq3367785771/Elamamba/releases/download/v1.0/scanobjectnn_best.pth)|

*(Note: Click the links above to download the pre-trained weights. Links will be activated upon paper release.)*

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

* [PointMamba](https://github.com/LMD0311/PointMamba)
* [Point-MAE](https://github.com/Pang-Yatian/Point-MAE)
* [Mamba](https://github.com/state-spaces/mamba)

## 📖 Citation

If you find our method, code useful in your research, please consider giving us a star ⭐ and citing our paper:

```bibtex
@article{ElaMamba2026,
  title={ElaMamba: Empowering State Space Models with Elastic Structural Scanning for 3D Point Clouds},
  author={Li, Zhiqiang and Liu, Qi and Li, Yufei and Feng, Qiang and Li, Shiyang and Wang, Xiyue and others},
  journal={Under review},
  year={2026}
}