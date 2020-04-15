## Data Preparation

### Notes on Video Data format
Since the original VideoDataloader of MMAction requires [decord](https://github.com/zhreshold/decord) for efficient video loading which is non-trivial to compile, this repo only supports **raw frame** format of videos. Therefore, you have to extract frames from raw videos. We will find another libaries and support VideoLoader soon.

### Supported datasets
The `rawframe_dataset` loads data in a general manner by preparing a `.txt` file which contains the directory path of frames, total number of a certain video, and the groundtruth label. After that, specify the `data_root` and `image_tmpl` of config files. See the sample below:

```bash
shot_put/c5-PBp04AQI 299 298
marching/5OEnoefcO1Y 299 192
dancing_ballet/pR1jxLvjcgU 249 84
motorcycling/0dC3o90WYHs 299 199
hoverboarding/RVkof6bxvg0 278 157
playing_piano/H3JzOkvTrJk 297 241
```
Such general loader might help your experiment with other dataset e.g. UCF101 or custom dataset.

### Prepare annotations

- [Kinetics400](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) contains ~240k training videos and ~19k validation videos. See the [guide](https://github.com/open-mmlab/mmaction/tree/master/data_tools/kinetics400/PREPARING_KINETICS400.md) of original MMAction to generate annotations.
- [Something-Someting](https://github.com/TwentyBN) has 2 versions which you have to apply on their [website](https://20bn.com/datasets/something-something). See the [guide](https://github.com/mit-han-lab/temporal-shift-module/tree/master/tools) of TSM to generate annotations.

Thank original [MMAction](https://github.com/open-mmlab/mmaction) and [TSM](https://github.com/mit-han-lab/temporal-shift-module) repo for kindly providing preprocessing scripts.
