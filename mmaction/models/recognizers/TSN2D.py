import torch.nn as nn
from .base import BaseRecognizer
from .. import builder
from ..registry import RECOGNIZERS
import torch
import numpy as np


@RECOGNIZERS.register_module
class TSN2D(BaseRecognizer):

    def __init__(self,
                 backbone,
                 necks=None,
                 modality='RGB',
                 in_channels=3,
                 spatial_temporal_module=None,
                 segmental_consensus=None,
                 fcn_testing=False,
                 flip=False,
                 cls_head=None,
                 train_cfg=None,
                 test_cfg=None):

        super(TSN2D, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.modality = modality
        self.in_channels = in_channels
        if necks is not None:
            self.necks = builder.build_neck(necks)
        else:
            self.necks = None

        if spatial_temporal_module is not None:
            self.spatial_temporal_module = builder.build_spatial_temporal_module(
                spatial_temporal_module)
        else:
            raise NotImplementedError

        if segmental_consensus is not None:
            self.segmental_consensus = builder.build_segmental_consensus(
                segmental_consensus)
        else:
            raise NotImplementedError

        if cls_head is not None:
            self.cls_head = builder.build_head(cls_head)
        else:
            raise NotImplementedError

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fcn_testing = fcn_testing
        self.flip = flip
        assert modality in ['RGB', 'Flow', 'RGBDiff']

        self.init_weights()

    @property
    def with_spatial_temporal_module(self):
        return hasattr(self, 'spatial_temporal_module') and self.spatial_temporal_module is not None

    @property
    def with_segmental_consensus(self):
        return hasattr(self, 'segmental_consensus') and self.segmental_consensus is not None

    @property
    def with_cls_head(self):
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def init_weights(self):
        super(TSN2D, self).init_weights()
        self.backbone.init_weights()

        if self.with_spatial_temporal_module:
            self.spatial_temporal_module.init_weights()

        if self.with_segmental_consensus:
            self.segmental_consensus.init_weights()

        if self.with_cls_head:
            self.cls_head.init_weights()

        if self.necks is not None:
            self.necks.init_weights()

    def extract_feat(self, img_group):
        x = self.backbone(img_group)
        return x

    def forward_train(self,
                      num_modalities,
                      img_meta,
                      gt_label,
                      **kwargs):
        assert num_modalities == 1
        img_group = kwargs['img_group_0']

        bs = img_group.shape[0]
        img_group = img_group.reshape(
            (-1, self.in_channels) + img_group.shape[3:])
        num_seg = img_group.shape[0] // bs

        x = self.extract_feat(img_group)
        if self.necks is not None:
            x = [each.reshape((-1, num_seg) + each.shape[1:]).transpose(1, 2) for each in x]
            x, aux_losses = self.necks(x, gt_label.squeeze())
            x = x.squeeze(2)
            num_seg = 1

        if self.with_spatial_temporal_module:
            x = self.spatial_temporal_module(x)
        x = x.reshape((-1, num_seg) + x.shape[1:])
        if self.with_segmental_consensus:
            x = self.segmental_consensus(x)
            x = x.squeeze(1)
        losses = dict()
        if self.with_cls_head:
            cls_score = self.cls_head(x)
            gt_label = gt_label.squeeze()
            loss_cls = self.cls_head.loss(cls_score, gt_label)
            losses.update(loss_cls)
        if self.necks is not None:
            if aux_losses is not None:
                losses.update(aux_losses)
        return losses

    def forward_test(self,
                     num_modalities,
                     img_meta,
                     **kwargs):
        if not self.fcn_testing:
            # 1crop * 1clip 
            assert num_modalities == 1
            img_group = kwargs['img_group_0']

            bs = img_group.shape[0]
            img_group = img_group.reshape(
                (-1, self.in_channels) + img_group.shape[3:])
            num_seg = img_group.shape[0] // bs

            x = self.extract_feat(img_group)

            if self.necks is not None:
                x = [each.reshape((-1, num_seg) + each.shape[1:]).transpose(1, 2) for each in x]
                x, _ = self.necks(x)
                x = x.squeeze(2)
                num_seg = 1

            if self.with_spatial_temporal_module:
                x = self.spatial_temporal_module(x)
            x = x.reshape((-1, num_seg) + x.shape[1:])
            if self.with_segmental_consensus:
                x = self.segmental_consensus(x)
                x = x.squeeze(1)
            if self.with_cls_head:
                x = self.cls_head(x)

            return x.cpu().numpy()
        else:
            # fcn testing
            assert num_modalities == 1
            img_group = kwargs['img_group_0']

            bs = img_group.shape[0]
            img_group = img_group.reshape(
                (-1, self.in_channels) + img_group.shape[3:])
            # standard protocol i.e. 3 crops * 2 clips
            num_seg = self.backbone.nsegments * 2
            # 3 crops to cover full resolution
            num_crops = 3
            img_group = img_group.reshape((num_crops, num_seg) + img_group.shape[1:])

            x1 = img_group[:, ::2, :, :, :]
            x2 = img_group[:, 1::2, :, :, :]
            img_group = torch.cat([x1, x2], 0)
            num_seg = num_seg // 2
            num_clips = img_group.shape[0]
            img_group = img_group.view(num_clips * num_seg, img_group.shape[2], img_group.shape[3], img_group.shape[4])

            if self.flip:
                img_group = self.extract_feat(torch.flip(img_group, [-1]))
            x = self.extract_feat(img_group)
            if self.necks is not None:
                x = [each.reshape((-1, num_seg) + each.shape[1:]).transpose(1, 2) for each in x]
                x, _ = self.necks(x)
            else:
                x = x.reshape((-1, num_seg) + x.shape[1:]).transpose(1, 2)
            x = self.cls_head(x)

            prob = torch.nn.functional.softmax(x.mean([2, 3, 4]), 1).mean(0, keepdim=True).detach().cpu().numpy()
            return prob
