import os
import re
import cv2
import argparse
import functools
import subprocess
import warnings
from scipy.special import softmax
import moviepy.editor as mpy
import numpy as np
import torch

import mmcv
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter

from mmaction.models import build_recognizer
from mmaction.datasets.transforms import GroupImageTransform


def init_recognizer(config, checkpoint=None, label_file=None, device='cuda:0'):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.backbone.pretrained = None
    config.model.spatial_temporal_module.spatial_size = 8
    model = build_recognizer(
        config.model, train_cfg=None, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if label_file is not None:
            classes = [line.rstrip() for line in open(label_file, 'r').readlines()]
            model.CLASSES = classes
        else:
            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                warnings.warn('Class names are not saved in the checkpoint\'s '
                              'meta data, use something-something-v2 classes by default.')
                model.CLASSES = get_classes('something=something-v2')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_recognizer(model, frames):
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_transform = GroupImageTransform(
        crop_size=cfg.data.test.input_size,
        oversample=None,
        resize_crop=False,
        **dict(mean=[123.675, 116.28, 103.53],
               std=[58.395, 57.12, 57.375], to_rgb=True))
    # prepare data
    frames, *l = test_transform(
        frames, (cfg.data.test.img_scale, cfg.data.test.img_scale),
        crop_history=None,
        flip=False,
        keep_ratio=False,
        div_255=False,
        is_flow=False)
    data = dict(img_group_0=frames,
                num_modalities=1,
                img_meta={})
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def extract_frames(video_file, num_frames=8):
    try:
        os.makedirs(os.path.join(os.getcwd(), 'frames'))
    except OSError:
        pass
    fps = subprocess.check_output(['ffprobe', '-v', 'error',
                                   '-select_streams',
                                   'v', '-of', 'default=noprint_wrappers=1:nokey=1',
                                   '-show_entries',
                                   ' stream=r_frame_rate',
                                   video_file]).decode('utf-8').strip().split('/')[0]
    fps = int(fps)

    output = subprocess.Popen(['ffmpeg', '-i', video_file,
                               '-loglevel', 'panic',
                               'frames/%d.jpg']).communicate()
    frame_paths = [os.path.join('frames', frame)
                   for frame in sorted(os.listdir('frames'), key=lambda x: int(x.split('.')[0]))]

    seg_frames, raw_frames = load_frames(frame_paths)
    subprocess.call(['rm', '-rf', 'frames'])

    return seg_frames, raw_frames, fps


def load_frames(frame_paths, num_frames=8):
    frames = [mmcv.imread(frame) for frame in frame_paths]
    if len(frames) >= num_frames:
        return frames[::int(np.floor(len(frames) / float(num_frames)))][:num_frames].copy(), frames.copy()
    else:
        raise ValueError('Video must have at least {} frames'.format(num_frames))


def render_frames(frames, prediction):
    rendered_frames = []
    for frame in frames:
        img = np.array(frame[:, :, ::-1])
        height, width, _ = img.shape
        cv2.putText(img=img, text=prediction, org=(1, int(height / 8)), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=0.6, color=(255, 255, 255), lineType=cv2.LINE_8, bottomLeftOrigin=False)
        rendered_frames.append(img)
    return rendered_frames


# options
parser = argparse.ArgumentParser(description="test TPN on a single video")
parser.add_argument('config', type=str, default=None, help='model init config')
parser.add_argument('checkpoint', type=str, default=None)
parser.add_argument('--label_file', type=str, default='demo/category.txt')
parser.add_argument('--video_file', type=str, default='demo/demo.mp4')
parser.add_argument('--frame_folder', type=str, default=None)
parser.add_argument('--rendered_output', type=str, default='demo/demo_pred.mp4')
args = parser.parse_args()

# Obtain video frames
if args.frame_folder is not None:
    print('Loading frames in {}'.format(args.frame_folder))
    import glob

    # Here, make sure after sorting the frame paths have the correct temporal order
    frame_paths = sorted(glob.glob(os.path.join(args.frame_folder, '*.jpg')))
    seg_frames, raw_frames = load_frames(frame_paths)
    fps = 4
else:
    print('Extracting frames using ffmpeg...')
    seg_frames, raw_frames, fps = extract_frames(args.video_file, 8)

model = init_recognizer(args.config, checkpoint=args.checkpoint, label_file=args.label_file)
results = inference_recognizer(model, seg_frames)
prob = softmax(results.squeeze())
idx = np.argsort(-prob)
# Output the prediction.
video_name = args.frame_folder if args.frame_folder is not None else args.video_file
print('RESULT ON ' + video_name)
for i in range(0, 5):
    print('{:.3f} -> {}'.format(prob[idx[i]], model.CLASSES[idx[i]]))

# Render output frames with prediction text.
if args.rendered_output is not None:
    prediction = model.CLASSES[idx[0]]
    rendered_frames = render_frames(raw_frames, prediction)
    clip = mpy.ImageSequenceClip(rendered_frames, fps=fps)
    clip.write_videofile(args.rendered_output)
