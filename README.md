# Enhanced noise2noise-based multitemporal progressive interaction learning for real dual-polarization SAR imagery despeckling
This repository is an official implementation of the paper "Enhanced noise2noise-based multitemporal progressive interaction learning for real dual-polarization SAR imagery despeckling".

By [Jiangong Xu](https://jiangong-xu.github.io/), Yang Yang, Weibao Xue, Yingdong Pi, Junli Li, Jun Pana, and Mi Wang

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![Pytorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)]()

## 🚀 Abstract
Effective speckle suppression is essential for the reliable utilization of synthetic aperture radar (SAR) data. Nevertheless, despeckling remains a challenging task due to the absence of clean reference data and the inherently complex statistical characteristics of speckle noise. Existing unsupervised deep learning methods partially mitigate this issue but still struggle to exploit temporal correlations among multitemporal observations and often overlook physical constraints inherent in polarimetric information, leading to incomplete structural recovery and scattering distortion. To overcome these limitations, we propose a multitemporal progressive interaction network (MTPI-Net) built upon an enhanced Noise2Noise paradigm, which introduces a hierarchical progressive learning strategy that jointly models spatial, frequency-domain, and temporal dependencies. Its core consists of stacked dual-domain collaboration and refinement units with a recursive residual-in-recursive-attention mechanism, enabling fine-grained cross-domain interaction through perception, gating, and aggregation. A collaborative optimization loss further enforces numerical fidelity, structural preservation, and temporal coherence, guided by polarimetric priors derived from covariance statistics and decomposition features. Extensive experiments on real dual-polarization Sentinel-1 time-series data demonstrate that MTPI-Net consistently surpasses existing approaches in both quantitative and visual evaluations, effectively preserving spatial details, polarimetric consistency, and semantic integrity.

![MTPI-Net Architecture]([figures/structure_of_MTPI-Net.png](https://github.com/JianGong-XU/MTPI-Net/blob/main/figurs/structure_of_MTPI-Net.png))




## 🚀 Features
- Dual-domain collaboration and refinement units
- Recursive residual-in-recursive-attention mechanism
- Multitemporal progressive learning
- Polarimetric prior guidance

## 📦 Installation

```bash
git clone https://github.com/JianGong-XU/MTPI-Net.git
cd MTPI-Net
pip install -r requirements.txt
