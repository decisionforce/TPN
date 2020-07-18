import argparse

import torch
import mmcv
import tempfile
import os.path as osp
import torch.distributed as dist
import shutil
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict, get_dist_info
from mmcv.parallel import scatter, collate, MMDataParallel, MMDistributedDataParallel
from mmaction.apis import init_dist
from mmaction import datasets
from mmaction.datasets import build_dataloader
from mmaction.models import build_recognizer, recognizers
from mmaction.core.evaluation.accuracy import (softmax, top_k_accuracy, non_mean_class_accuracy,
                                               mean_class_accuracy)


def single_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data['get_logit'] = True
            result = model(return_loss=False, **data)
        results.append(result)

        batch_size = data['img_group_0'].data[0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # data['get_logit'] = True
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img_group_0'].data[0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full(
            (MAX_LEN,), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            print('temp_dir', tmpdir)
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoinls'
                                           't file')
    parser.add_argument(
        '--gpus', default=8, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--log', help='output log file')
    parser.add_argument('--fcn_testing', action='store_true', default=False,
                        help='whether to use fcn testing')
    parser.add_argument('--flip', action='store_true', default=False,
                        help='whether to flip videos')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--ignore_cache', action='store_true', help='whether to ignore cache')
    args = parser.parse_args()
    print('args==>>', args)
    return args


def main():
    args = parse_args()

    assert args.out, ('Please specify the output path for results')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if cfg.model.get('necks', None) is not None:
        cfg.model.necks.aux_head_config = None

    if cfg.data.test.oversample == 'three_crop':
        cfg.model.spatial_temporal_module.spatial_size = 8
    if args.fcn_testing:
        cfg.model['cls_head'].update({'fcn_testing': True})
        cfg.model.update({'fcn_testing': True})
    if args.flip:
        cfg.model.update({'flip': True})

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))

    if args.ignore_cache and args.out is not None:
        if not distributed:
            if args.gpus == 1:
                model = build_recognizer(
                    cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
                load_checkpoint(model, args.checkpoint, strict=False, map_location='cpu')
                model = MMDataParallel(model, device_ids=[0])

                data_loader = build_dataloader(
                    dataset,
                    imgs_per_gpu=1,
                    workers_per_gpu=cfg.data.workers_per_gpu,
                    num_gpus=1,
                    dist=False,
                    shuffle=False)
                outputs = single_test(model, data_loader)
            else:
                model_args = cfg.model.copy()
                model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
                model_type = getattr(recognizers, model_args.pop('type'))

                outputs = parallel_test(
                    model_type,
                    model_args,
                    args.checkpoint,
                    dataset,
                    _data_func,
                    range(args.gpus),
                    workers_per_gpu=args.proc_per_gpu)
        else:
            data_loader = build_dataloader(
                dataset,
                imgs_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False)
            model = build_recognizer(
                cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
            load_checkpoint(model, args.checkpoint, strict=False, map_location='cpu')
            model = MMDistributedDataParallel(model.cuda())
            outputs = multi_gpu_test(model, data_loader, args.tmpdir)
    else:
        try:
            if distributed:
                rank, _ = get_dist_info()
                if rank == 0:
                    outputs = mmcv.load(args.out)
            else:
                outputs = mmcv.load(args.out)
        except:
            raise FileNotFoundError

    rank, _ = get_dist_info()
    if args.out:
        if rank == 0:
            print('writing results to {}'.format(args.out))
            mmcv.dump(outputs, args.out)
            gt_labels = []
            for i in range(len(dataset)):
                ann = dataset.get_ann_info(i)
                gt_labels.append(ann['label'])

            results = []
            for res in outputs:
                res_list = [res[i] for i in range(res.shape[0])]
                results += res_list
            results = results[:len(gt_labels)]
            print('results_length', len(results))
            top1, top5 = top_k_accuracy(results, gt_labels, k=(1, 5))
            mean_acc = mean_class_accuracy(results, gt_labels)
            non_mean_acc = non_mean_class_accuracy(results, gt_labels)
            if args.log:
                f = open(args.log, 'w')
                f.write(f'Testing ckpt from {args.checkpoint}\n')
                f.write(f'Testing config from {args.config}\n')
                f.write("Mean Class Accuracy = {:.04f}\n".format(mean_acc * 100))
                f.write("Top-1 Accuracy = {:.04f}\n".format(top1 * 100))
                f.write("Top-5 Accuracy = {:.04f}\n".format(top5 * 100))
                f.close()
            else:
                print("Mean Class Accuracy = {:.02f}".format(mean_acc * 100))
                print("Top-1 Accuracy = {:.02f}".format(top1 * 100))
                print("Top-5 Accuracy = {:.02f}".format(top5 * 100))
                print("Non mean Class Accuracy", non_mean_acc)
                print('saving non_mean acc')


if __name__ == '__main__':
    main()
