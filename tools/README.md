# Getting Started

This directory provides basic tutorials for the usage of MMAction.

After installation of codebase and preparation of data, you could use the given scripts for training/evaluating your models.

### Test a reference model
Our codebase supports distributed and non-distributed evaluation mode for reference model. Actually, distributed testing is a little faster than non-distributed testing.  
```
# non-distributed testing
python tools/test_recognizer.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] {--gpus ${GPU_NUM}} --ignore_cache --fcn_testing

# distributed testing
./tools/dist_test_recognizer.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] --ignore_cache --fcn_testing
```
Optional arguments:
- `--ignore_cache`: If specified, the results cache will be ignored.
- `--fcn_testing`: If specified, spatially fully-convolutional testing is performed via 3 crops approximation.
- `--flip`: If specified, all frames would be flipped firstly and then fed into models.

**Important**: some of our models might requires machine with more than 24G memory.

Examples:
Assume that you have already downloaded the checkpoints to the directory `ckpt/`.

1. Test tpn_f8s8 model with non-distributed evaluation mode on 8 GPUs
```
python ./tools/test_recognizer.py config_files/kinetics400/tpn/r50f8s8.py ckpt/kinetics400_tpn_r50f8s8 --gpus  8  --out ckpt/kinetics400_tpn_r50f8s8.pkl --fcn_testing --ignore_cache
```
2. Test tpn_f8s8 model with distributed evaluation mode on 8 GPUs
```shell
./tools/dist_test_recognizer.sh config_files/kinetics400/tpn/r50f8s8.py ckpt/kinetics400_tpn_r50f8s8 8  --out ckpt/kinetics400_tpn_r50f8s8.pkl --fcn_testing --ignore_cache
```

### Train a model
 
Our codebase also supports distributed training and non-distributed training.

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

By default we evaluate the model on the validation set after each epoch, you can change the evaluation interval by adding the interval argument in the training config.
```python
evaluation = dict(interval=10)  # This evaluate the model per 10 epoch.
```

#### Train with a single GPU
```
python tools/train_recognizer.py ${CONFIG_FILE}
```
If you want to specify the working directory in the command, you can add an argument `--work_dir ${YOUR_WORK_DIR}`.

#### Train with multiple GPUs
```shell
./tools/dist_train_recognizer.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments:
- `--validate`: Perform evaluation at every 1 epoch during the training.
- `--work_dir`: All outputs (log files and checkpoints) will be saved to the working directory. 
- `--resume_from`: Resume from a previous checkpoint file.
 
Difference between `resume_from` and `load_from`: `resume_from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally. `load_from` only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

**Important**: The default learning rate in config files is for 8 GPUs and 8 video/gpu (batch size = 8*8 = 64). According to the Linear Scaling Rule, you need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU, e.g., lr=0.01 for 8 GPUs * 8 video/gpu and lr=0.04 for 32 GPUs * 8 video/gpu.

Here is the example of using 8 GPUs to train Kinetics400_r50_f8s8:
```shell
./tools/dist_train_recognizer.sh config_files/kinetics400/tpn/r50f8s8.py 8 --validate 
```
