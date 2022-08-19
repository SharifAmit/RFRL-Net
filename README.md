# MICCAI-OMAI2022 RFRL-Network

This code is for our paper "Feature Representation Learning for Robust Retinal Disease Detection from Optical Coherence Tomography Images" which is part of the supplementary materials for MICCAI 2022 Ophthalmic Medical Image Analysis (OMIA) Workshop. The paper has since been accpeted and presented at MICCAI-OMAI 2022.


![](img.png?)

### Arxiv Pre-print
```
https://arxiv.org/pdf/2101.00535v2
```

# Citation
```
@article{kamran2022feature,
  title={Feature Representation Learning for Robust Retinal Disease Detection from Optical Coherence Tomography Images},
  author={Kamran, Sharif Amit and Hossain, Khondker Fariha and Tavakkoli, Alireza and Zuckerbrod, Stewart Lee and Baker, Salah A},
  journal={arXiv preprint arXiv:2206.12136},
  year={2022}
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
