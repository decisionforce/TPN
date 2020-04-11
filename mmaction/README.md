# mmaction

This code is based on [MMAction](https://github.com/open-mmlab/mmaction) which supports modular design and high efficiency. Our TPN would be merged into the latest MMAction in the future.

Here we briefly introduce the structure of this codebase:

- `apis`: contains the launcher of the whole codebase and intializer of distributed training environment.
- `core`: contains multiple hooks for evaluation e.g. calculating the Top-1/Top-5 accuracy.
- `datasets`: contains `rawframes_dataset` and transform for training.
- `losses`: contains kinds of CrossEntropy loss.
- `models`: contains recognizers and various submodules of network e.g. *backbone*, *neck*,and *head* under `models/tenons` 

Such modular design helps us quickly and easily conduct experiments with different modules.
