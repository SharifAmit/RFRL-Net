# MICCAI-OMIA2022 RFRL-Network

This code is for our paper "Feature Representation Learning for Robust Retinal Disease Detection from Optical Coherence Tomography Images" which is part of the supplementary materials for MICCAI 2022 9th Ophthalmic Medical Image Analysis (OMIA) Workshop. The paper has since been accepted at MICCAI-OMIA 2022.


![](img.png?)

### Springer
```
https://link.springer.com/chapter/10.1007/978-3-031-16525-2_3
```

### Arxiv Pre-print
```
https://arxiv.org/pdf/2101.00535v2
```

# Citation
```
@InProceedings{10.1007/978-3-031-16525-2_3,
author="Kamran, Sharif Amit
and Hossain, Khondker Fariha
and Tavakkoli, Alireza
and Zuckerbrod, Stewart Lee
and Baker, Salah A.",
editor="Antony, Bhavna
and Fu, Huazhu
and Lee, Cecilia S.
and MacGillivray, Tom
and Xu, Yanwu
and Zheng, Yalin",
title="Feature Representation Learning for Robust Retinal Disease Detection from Optical Coherence Tomography Images",
booktitle="Ophthalmic Medical Image Analysis",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="22--32",
abstract="Ophthalmic images may contain identical-looking pathologies that can cause failure in automated techniques to distinguish different retinal degenerative diseases. Additionally, reliance on large annotated datasets and lack of knowledge distillation can restrict ML-based clinical support systems' deployment in real-world environments. To improve the robustness and transferability of knowledge, an enhanced feature-learning module is required to extract meaningful spatial representations from the retinal subspace. Such a module, if used effectively, can detect unique disease traits and differentiate the severity of such retinal degenerative pathologies. In this work, we propose a robust disease detection architecture with three learning heads, i) A supervised encoder for retinal disease classification, ii) An unsupervised decoder for the reconstruction of disease-specific spatial information, and iii) A novel representation learning module for learning the similarity between encoder-decoder feature and enhancing the accuracy of the model. Our experimental results on two publicly available OCT datasets illustrate that the proposed model outperforms existing state-of-the-art models in terms of accuracy, interpretability, and robustness for out-of-distribution retinal disease detection.",
isbn="978-3-031-16525-2"
}

```

## Pre-requisite
- Ubuntu 18.04 / Windows 7 or later
- NVIDIA Graphics card

## Installation Instruction for Ubuntu
- Download and Install [Nvidia Drivers](https://www.nvidia.com/Download/driverResults.aspx/142567/en-us)
- Download and Install via Runfile [Nvidia Cuda Toolkit 10.0](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal)
- Download and Install [Nvidia CuDNN 7.6.5 or later](https://developer.nvidia.com/rdp/cudnn-archive)
- Install Pip3 and Python3 enviornment
```
sudo apt-get install pip3 python3-dev
```
- Install Tensorflow-Gpu version-2.0.0 and Keras version-2.3.1
```
sudo pip3 install tensorflow-gpu==2.0.0
sudo pip3 install keras==2.3.1
```
- Install packages from requirements.txt
```
sudo pip3 install -r requirements.txt
```
