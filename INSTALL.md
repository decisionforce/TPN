# Installation

```shell
git clone https://github.com/decisionforce/TPN.git
```

## Requirements

- Linux
- Python 3.5+
- PyTorch 1.0+
- CUDA 9.0+
- NVCC 2+
- GCC 4.9+
- [mmcv](https://github.com/open-mmlab/mmcv).
  Note that you are strongly recommended to clone the master branch and build from scratch since some of the features have not been added in the latest release.

## Install MMAction
(a) Install Cython
```shell
pip install cython
```
(b) Install mmaction
```shell
python setup.py develop
```

