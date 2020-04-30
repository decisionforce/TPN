import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmcv import Config
import numpy as np

from ...registry import NECKS


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ConvModule(nn.Module):
    def __init__(
            self,
            inplanes,
            planes,
            kernel_size,
            stride,
            padding,
            bias=False,
            groups=1,
    ):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv3d(inplanes, planes, kernel_size, stride, padding, bias=bias, groups=groups)
        self.bn = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class AuxHead(nn.Module):
    def __init__(
            self,
            inplanes,
            planes,
            loss_weight=0.5
    ):
        super(AuxHead, self).__init__()
        self.convs = \
            ConvModule(inplanes, inplanes * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.loss_weight = loss_weight
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(inplanes * 2, planes)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)

    def forward(self, x, target=None):
        if target is None:
            return None
        loss = dict()
        x = self.convs(x)
        x = F.adaptive_avg_pool3d(x, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)

        loss['loss_aux'] = self.loss_weight * F.cross_entropy(x, target)
        return loss


class TemporalModulation(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 downsample_scale=8,
                 ):
        super(TemporalModulation, self).__init__()

        self.conv = nn.Conv3d(inplanes, planes, (3, 1, 1), (1, 1, 1), (1, 0, 0), bias=False, groups=32)
        self.pool = nn.MaxPool3d((downsample_scale, 1, 1), (downsample_scale, 1, 1), (0, 0, 0), ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class Upsampling(nn.Module):
    def __init__(self,
                 scale=(2, 1, 1),
                 ):
        super(Upsampling, self).__init__()
        self.scale = scale

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        return x


class Downampling(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=(3, 1, 1),
                 stride=(1, 1, 1),
                 padding=(1, 0, 0),
                 bias=False,
                 groups=1,
                 norm=False,
                 activation=False,
                 downsample_position='after',
                 downsample_scale=(1, 2, 2),
                 ):
        super(Downampling, self).__init__()

        self.conv = nn.Conv3d(inplanes, planes, kernel_size, stride, padding, bias=bias, groups=groups)
        self.norm = nn.BatchNorm3d(planes) if norm else None
        self.relu = nn.ReLU(inplace=True) if activation else None
        assert (downsample_position in ['before', 'after'])
        self.downsample_position = downsample_position
        self.pool = nn.MaxPool3d(downsample_scale, downsample_scale, (0, 0, 0), ceil_mode=True)

    def forward(self, x):
        if self.downsample_position == 'before':
            x = self.pool(x)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.downsample_position == 'after':
            x = self.pool(x)

        return x


class LevelFusion(nn.Module):
    def __init__(self,
                 in_channels=[1024, 1024],
                 mid_channels=[1024, 1024],
                 out_channels=2048,
                 ds_scales=[(1, 1, 1), (1, 1, 1)],
                 ):
        super(LevelFusion, self).__init__()
        self.ops = nn.ModuleList()
        num_ins = len(in_channels)
        for i in range(num_ins):
            op = Downampling(in_channels[i], mid_channels[i], kernel_size=(1, 1, 1), stride=(1, 1, 1),
                             padding=(0, 0, 0), bias=False, groups=32, norm=True, activation=True,
                             downsample_position='before', downsample_scale=ds_scales[i])
            self.ops.append(op)

        in_dims = np.sum(mid_channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(in_dims, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        out = [self.ops[i](feature) for i, feature in enumerate(inputs)]
        out = torch.cat(out, 1)
        out = self.fusion_conv(out)
        return out


class SpatialModulation(nn.Module):
    def __init__(
            self,
            inplanes=[1024, 2048],
            planes=2048,
    ):
        super(SpatialModulation, self).__init__()

        self.spatial_modulation = nn.ModuleList()
        for i, dim in enumerate(inplanes):
            op = nn.ModuleList()
            ds_factor = planes // dim
            ds_num = int(np.log2(ds_factor))
            if ds_num < 1:
                op = Identity()
            else:
                for dsi in range(ds_num):
                    in_factor = 2 ** dsi
                    out_factor = 2 ** (dsi + 1)
                    op.append(ConvModule(dim * in_factor, dim * out_factor, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                         padding=(0, 1, 1), bias=False))
            self.spatial_modulation.append(op)

    def forward(self, inputs):
        out = []
        for i, feature in enumerate(inputs):
            if isinstance(self.spatial_modulation[i], nn.ModuleList):
                out_ = inputs[i]
                for III, op in enumerate(self.spatial_modulation[i]):
                    out_ = op(out_)
                out.append(out_)
            else:
                out.append(self.spatial_modulation[i](inputs[i]))
        return out


@NECKS.register_module
class TPN(nn.Module):

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=256,
                 spatial_modulation_config=None,
                 temporal_modulation_config=None,
                 upsampling_config=None,
                 downsampling_config=None,
                 level_fusion_config=None,
                 aux_head_config=None,
                 ):
        super(TPN, self).__init__()
        assert isinstance(in_channels, list)
        assert isinstance(out_channels, int)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)

        spatial_modulation_config = Config(spatial_modulation_config) if isinstance(spatial_modulation_config,
                                                                                    dict) else spatial_modulation_config
        temporal_modulation_config = Config(temporal_modulation_config) if isinstance(temporal_modulation_config,
                                                                                      dict) else temporal_modulation_config
        upsampling_config = Config(upsampling_config) if isinstance(upsampling_config, dict) else upsampling_config
        downsampling_config = Config(downsampling_config) if isinstance(downsampling_config,
                                                                        dict) else downsampling_config
        aux_head_config = Config(aux_head_config) if isinstance(aux_head_config, dict) else aux_head_config
        level_fusion_config = Config(level_fusion_config) if isinstance(level_fusion_config,
                                                                        dict) else level_fusion_config

        self.temporal_modulation_ops = nn.ModuleList()
        self.upsampling_ops = nn.ModuleList()
        self.downsampling_ops = nn.ModuleList()
        self.level_fusion_op = LevelFusion(**level_fusion_config)
        self.spatial_modulation = SpatialModulation(**spatial_modulation_config)
        for i in range(0, self.num_ins, 1):
            inplanes = in_channels[-1]
            planes = out_channels

            if temporal_modulation_config is not None:
                # overwrite the temporal_modulation_config
                temporal_modulation_config.param.downsample_scale = temporal_modulation_config.scales[i]
                temporal_modulation_config.param.inplanes = inplanes
                temporal_modulation_config.param.planes = planes
                temporal_modulation = TemporalModulation(**temporal_modulation_config.param)
                self.temporal_modulation_ops.append(temporal_modulation)

            if i < self.num_ins - 1:
                if upsampling_config is not None:
                    # overwrite the upsampling_config
                    upsampling = Upsampling(**upsampling_config)
                    self.upsampling_ops.append(upsampling)

                if downsampling_config is not None:
                    # overwrite the downsampling_config
                    downsampling_config.param.inplanes = planes
                    downsampling_config.param.planes = planes
                    downsampling_config.param.downsample_scale = downsampling_config.scales
                    downsampling = Downampling(**downsampling_config.param)
                    self.downsampling_ops.append(downsampling)

        out_dims = level_fusion_config.out_channels

        # Two pyramids
        self.level_fusion_op2 = LevelFusion(**level_fusion_config)

        self.pyramid_fusion_op = nn.Sequential(
            nn.Conv3d(out_dims * 2, 2048, 1, 1, 0, bias=False),
            nn.BatchNorm3d(2048),
            nn.ReLU(inplace=True)
        )

        # overwrite aux_head_config
        if aux_head_config is not None:
            aux_head_config.inplanes = self.in_channels[-2]
            self.aux_head = AuxHead(**aux_head_config)
        else:
            self.aux_head = None

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)

        if self.aux_head is not None:
            self.aux_head.init_weights()

    def forward(self, inputs, target=None):
        loss = None

        # Auxiliary loss
        if self.aux_head is not None:
            loss = self.aux_head(inputs[-2], target)

            # Spatial Modulation
        outs = self.spatial_modulation(inputs)

        # Temporal Modulation
        outs = [temporal_modulation(outs[i]) for i, temporal_modulation in enumerate(self.temporal_modulation_ops)]

        temporal_modulation_outs = outs

        # Build top-down flow - upsampling operation
        if self.upsampling_ops is not None:
            for i in range(self.num_ins - 1, 0, -1):
                outs[i - 1] = outs[i - 1] + self.upsampling_ops[i - 1](outs[i])

        # Get top-down outs
        topdownouts = self.level_fusion_op2(outs)
        outs = temporal_modulation_outs

        # Build bottom-up flow - downsampling operation
        if self.downsampling_ops is not None:
            for i in range(0, self.num_ins - 1, 1):
                outs[i + 1] = outs[i + 1] + self.downsampling_ops[i](outs[i])

                # Get bottom-up outs
        outs = self.level_fusion_op(outs)

        # fuse two pyramid outs
        outs = self.pyramid_fusion_op(torch.cat([topdownouts, outs], 1))

        return outs, loss
