# Temporal Pyramid Network for Action Recognition 

![image](./docs/figures/framework.png)
[[Paper](https://arxiv.org/pdf/2004.03548.pdf)]
[[Project Page](https://decisionforce.github.io/TPN/)]


## License
The project is release under the [Apache 2.0 license](./LICENSE).

## Model zoo
Results and reference models are available in the [model zoo](./MODELZOO.md).

## Installation and Data preparation
Please refer to [INSTALL](INSTALL.md) for installation and [DATA](./data/README.md) for data preparation.

## Get started
Please refer to [GETTING_STARTED](./tools/README.md) for detailed usage.

## Quick demo
We provide `test_video.py` to inference a single video.
Download the checkpoints and put them to the `ckpt/.` and run:
```
python ./test_video.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --video_file ${VIDOE_NAME} --label_file ${LABLE_FILE} --rendered_output ${RENDERED_NAME}
```
Arguments:
- `--video_file`: Path for demo video, default is `./demo/demo.mp4` 
- `--label_file`: The label file for pretrained model, default is `demo/category.txt`
- `--redndered_output`: The output file name. If specified, the script will render output video with label name, default is `demo/demo_pred.webm`. 

For example, we could predict for demo video (download [here](https://drive.google.com/open?id=14VYS8hGA5i1J70qBqrUqLiDxJq_FgXiW) and put it under `demo/.`)  by running:
```
python ./test_video.py config_files/sthv2/tsm_tpn.py ckpt/sthv2_tpn.pth
```
The rendered output video:

![image](./demo/demo_pred.gif)

## Acknowledgement
We really appreciate developers of [MMAction](https://github.com/open-mmlab/mmaction) for such wonderful codebase. We also thank Yue Zhao for the insightful discussion.

## Contact
This repo is currently maintained by Ceyuan Yang ([@limbo0000](https://github.com/limbo0000)) and Yinghao Xu ([@justimyhxu](https://github.com/justimyhxu)).

## Bibtex
```
@inproceedings{yang2020tpn,
  title={Temporal Pyramid Network for Action Recognition},
  author={Yang, Ceyuan and Xu, Yinghao and Shi, Jianping and Dai, Bo and Zhou, Bolei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020},
}
```
