# Model Zoo

## Pretrained Models
All pretrained models could be downloaded from [Google Drive](https://drive.google.com/drive/folders/1UnqZ48doF0UTYjH6iZCXQW3HlDocbBxl). After downloading, put them into `ckpt/`.

## Main Results
We report our methods on Kinetics-400, Something-Something V1 and V2. All the numbers including baseline and TPN are obtained via fully-convolutional testing. 

### Kinetics-400
Since the number of Kinetics-400 videos are slightly different (might lead to a performance drop), we report all results on own dataset. Our data contains 240403 training videos and 19769 validation videos which are rescaled to 240*320 resolution. Note that the trimmed time of [Non-Local](https://github.com/facebookresearch/video-nonlocal-net/blob/master/DATASET.md) data and the resolution of [MMAction](https://github.com/open-mmlab/mmaction/blob/master/MODEL_ZOO.md) data are different from ours. But the improvement of TPN are consistent. In order to ensure the reproduction, we will find a proper way to release our validation set. All the following results on Kinetics-400 also take flip augmentation testing (~0.1% fluctuation). We sample F frames with stride of S frames (denote FxS). 


| Model | Frames | TPN | Top-1 | Weights | Config | 
| :---: | :------: | :--------: | :------: | :------: | :------ |
|R50    | 8 x 8    | -   | 74.9 | [link](https://drive.google.com/open?id=1uKHvZsY_heFHTBl6RXo02I7-W_aLBhFI) | config_files/kinetics400/baseline/r50f8s8.py |
|R50    | 8 x 8    | Yes | 76.1 | [link](https://drive.google.com/open?id=1KoISwdKDlfzZdEsLItygcvPGkKNwWyR-) | config_files/kinetics400/tpn/r50f8s8.py |
|R50    | 16 x 4   | -   | 76.1 | [link](https://drive.google.com/open?id=1Qgck89mUVs9gyUzalbYJPfJPwQEPbyI9) | config_files/kinetics400/baseline/r50f16s4.py |
|R50    | 16 x 4   | Yes | 77.3 | [link](https://drive.google.com/open?id=1TY39uBR-ckUw3aiabeFLNpR9uPSxt--H) | config_files/kinetics400/tpn/r50f16s4.py |
|R50    | 32 x 2   | -   | 75.7 | [link](https://drive.google.com/open?id=1oJ1sTzMeLPXHtnutJAAD8gWfm0b3NYpi) | config_files/kinetics400/baseline/r50f32s2.py |
|R50    | 32 x 2   | Yes | 77.7 | [link](https://drive.google.com/open?id=1TjeqcTJ2tReDz4VnLR8ajSHySre9sZDd) | config_files/kinetics400/tpn/r50f32s2.py |
|R101   | 8 x 8    | -   | 76.0 | [link](https://drive.google.com/open?id=1dqLWiI3DFHAPIzGtEY_jfI66nthw2GEX) | config_files/kinetics400/baseline/r101f8s8.py |
|R101   | 8 x 8    | Yes | 77.2 | [link](https://drive.google.com/open?id=1B4Vsld-JzQe4QmXeZHd0TolMPNyZypXI) | config_files/kinetics400/tpn/r101f8s8.py |
|R101   | 16 x 4   | -   | 77.0 | [link](https://drive.google.com/open?id=1tj2Y0OChKW7RoElXXmBeU63dph40kEyJ) | config_files/kinetics400/baseline/r101f16s4.py |
|R101   | 16 x 4   | Yes | 78.1 | [link](https://drive.google.com/open?id=1mT4kuaYuAGA-Zjagc56vByMQdvx0bE-H) | config_files/kinetics400/tpn/r101f16s4.py |
|R101   | 32 x 2   | -   | 77.4 | [link](https://drive.google.com/open?id=1IAobiYS3PhXC1sA_MCdudGCdHRWcWc9J) | config_files/kinetics400/baseline/r101f32s2.py |
|R101   | 32 x 2   | Yes | 78.9 | [link](https://drive.google.com/open?id=1OPudI7CzJzpdeI0YpwLgZB59VCzcoidp) | config_files/kinetics400/tpn/r101f32s2.py |

We also train our TPN on [MMAction](https://github.com/open-mmlab/mmaction/blob/master/MODEL_ZOO.md) data, the performance will increase due to the raw resolution and ratio.

| Model | Frames | TPN | Top-1 | Weights | Config |
| :---: | :------: | :--------: | :------: | :------: | :------ |
|R50    | 8 x 8    | Yes  | 76.7 | [link](https://drive.google.com/drive/folders/1UnqZ48doF0UTYjH6iZCXQW3HlDocbBxl) | config_files/kinetics400/baseline/r50f8s8.py |
|R101   | 8 x 8    | Yes  | 78.2 | [link](https://drive.google.com/drive/folders/1UnqZ48doF0UTYjH6iZCXQW3HlDocbBxl) | config_files/kinetics400/baseline/r101f8s8.py |

All models are trained on 32 GPUs with 150 epochs. More details could be found in `config_files`.

### Something-Something
Something-Something is a more stable benchmark and the whole data could be download from their [website](https://20bn.com/datasets/something-something). We report our results on both V1 and V2. All numbers are obtained by following the standard protocol i.e., 3 crops * 2 clips. [TSM](https://github.com/mit-han-lab/temporal-shift-module) serves as our backbone network. 
Different from original [repo](https://github.com/mit-han-lab/temporal-shift-module) of TSM which takes Kinetics-pretrain, our implementation is initialized by imagenet-pretrain and trained with longer schedule. We use **the same** hyper-parameters of training for both baseline and TPN. Therefore, the improvements come from TPN design instead of other training tricks. We take the uniform sampling for training and validation.

| Model | Dataset Version | Frames | TPN | Top-1 | Weights | Config |
| :---: | :------: |    :------: | :--------: | :------: | :------: | :------ |
|TSM50  | V1       | 8  | -   | 48.2 | [link](https://drive.google.com/open?id=1x7iwL2Op0qxaUluyQCPOVVEEH53cavhL) | config_files/sthv1/tsm_baseline.py |  
|TSM50  | V1       | 8  | Yes | 50.7 | [link](https://drive.google.com/open?id=1NVjsCYgNXKUKAn33XCxV2YEIaWXlEnLS) | config_files/sthv1/tsm_tpn.py      |
|TSM50  | V2       | 8  | -   | 62.3 | [link](https://drive.google.com/open?id=1fU1b9WySld5knJ8E2bMXfuyRenoViSEX) | config_files/sthv2/tsm_baseline.py |
|TSM50  | V2       | 8  | Yes | 64.7 | [link](https://drive.google.com/open?id=15HHKGIhksTf0dSmgxrTsoHzZxF6n7eRa) | config_files/sthv2/tsm_tpn.py      |

If you have any problem about how to reproduce our results, please contact Ceyuan Yang (yc019@ie.cuhk.edu.hk) or Yinghao Xu (xy119@ie.cuhk.edu.hk).

