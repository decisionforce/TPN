from .base import BaseRecognizer
from .. import builder
from ..registry import RECOGNIZERS

import torch


@RECOGNIZERS.register_module
class TSN3D(BaseRecognizer):

    def __init__(self,
                 backbone,
                 necks=None,
                 spatial_temporal_module=None,
                 segmental_consensus=None,
                 fcn_testing=False,
                 flip=False,
                 cls_head=None,
                 train_cfg=None,
                 test_cfg=None):

        super(TSN3D, self).__init__()
        self.backbone = builder.build_backbone(backbone)

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
        super(TSN3D, self).init_weights()
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
        img_group = img_group.reshape((-1,) + img_group.shape[2:])
        num_seg = img_group.shape[0] // bs

        x = self.extract_feat(img_group)

        if self.necks is not None:
            x, aux_losses = self.necks(x, gt_label.squeeze())

        if self.with_spatial_temporal_module:
            x = self.spatial_temporal_module(x)
        if self.with_segmental_consensus:
            x = x.reshape((-1, num_seg) + x.shape[1:])
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
        assert num_modalities == 1
        img_group = kwargs['img_group_0']

        bs = img_group.shape[0]
        img_group = img_group.reshape((-1,) + img_group.shape[2:])
        num_seg = img_group.shape[0] // bs

        if self.flip:
            img_group = self.extract_feat(torch.flip(img_group, [-1]))
        x = self.extract_feat(img_group)
        if self.necks is not None:
            x, _ = self.necks(x)
        if self.fcn_testing:
            if self.with_cls_head:
                x = self.cls_head(x)
                prob1 = torch.nn.functional.softmax(x.mean([2, 3, 4]), 1).mean(0, keepdim=True).detach().cpu().numpy()
                return prob1

        if self.with_spatial_temporal_module:
            x = self.spatial_temporal_module(x)
        if self.with_segmental_consensus:
            x = x.reshape((-1, num_seg) + x.shape[1:])
            x = self.segmental_consensus(x)
            x = x.squeeze(1)
        if self.with_cls_head:
            x = self.cls_head(x)

        return x.cpu().numpy()
