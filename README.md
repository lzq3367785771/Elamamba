# ElaMamba: Enhancing 3D Point Cloud Analysis with Elastic Structural Scanning

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](#)

This is the official PyTorch implementation of **ElaMamba**. 

In this repository, we introduce **ElaMamba**, a novel architecture tailored for 3D point cloud classification. By integrating an **Elastic Structural Scanning (ESS)** module, Dynamic Gating, and an Adaptive Regularization Loss, ElaMamba effectively captures both local geometries and global contexts using State Space Models (Mamba), achieving state-of-the-art performance on multiple benchmarks.

> **Insert your architecture diagram here:** > `![ElaMamba Architecture](figure/architecture.png)`

## 🚀 Key Features
* **Elastic Structural Scanning (ESS):** Dynamically optimizes the scanning sequence of 3D point clouds.
* **Adaptive Regularization Loss:** Constrains the predicted structural offsets to ensure robust convergence.
* **Efficient Mamba Backbone:** Leverages State Space Models for linear-complexity sequence modeling on 3D data.

## 📊 Main Results

| Dataset | Split | Overall Accuracy (OA) | Checkpoint |
| :--- | :---: | :---: | :---: |
| **ModelNet40** | Test | 93.6% | [Download](#) |
| **ScanObjectNN** | Hardest | 88.X% | [Download](#) |

*(Note: Pre-trained weights will be uploaded to the `ckpts` folder soon.)*

## 🛠️ Installation

**1. Clone the repository:**
```bash
git clone [https://github.com/lzq3367785771/Elamamba.git](https://github.com/lzq3367785771/Elamamba.git)
cd Elamamba