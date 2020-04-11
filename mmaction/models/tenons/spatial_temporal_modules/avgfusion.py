import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registry import SPATIAL_TEMPORAL_MODULES


@SPATIAL_TEMPORAL_MODULES.register_module
class AvgFusion(nn.Module):
    def __init__(self, fusion_type='concat'):
        super(AvgFusion, self).__init__()
        assert fusion_type in ['add', 'avg', 'concat', 'concatadd', 'concatavg']
        self.fusion_type = fusion_type

    def init_weights(self):
        pass

    def forward(self, input):
        assert (isinstance(input, tuple))
        after_avgpool = [F.adaptive_avg_pool3d(each, 1) for each in input]

        if self.fusion_type == 'add':
            out = torch.sum(torch.cat(after_avgpool, -1), -1, keepdim=True)

        elif self.fusion_type == 'avg':
            out = torch.mean(torch.cat(after_avgpool, -1), -1, keepdim=True)

        elif self.fusion_type == 'concat':
            out = torch.cat(after_avgpool, 1)

        elif self.fusion_type == 'concatadd':
            out_first = torch.cat(after_avgpool[:-1], 1)
            out = torch.sum(torch.cat([out_first, after_avgpool[-1]], -1), -1, keepdim=True)
        elif self.fusion_type == 'concatavg':
            out_first = torch.cat(after_avgpool[:-1], 1)
            out = torch.mean(torch.cat([out_first, after_avgpool[-1]], -1), -1, keepdim=True)
        else:
            raise ValueError

        return out


def main():
    res2 = torch.FloatTensor(8, 512, 8, 56, 56).cuda()
    res3 = torch.FloatTensor(8, 512, 8, 28, 28).cuda()
    res4 = torch.FloatTensor(8, 512, 8, 14, 14).cuda()
    res5 = torch.FloatTensor(8, 512, 8, 7, 7).cuda()
    feature = tuple([res2, res3, res4, res5])
    model = AvgFusion(fusion_type='add').cuda()
    out = model(feature)
    print(out.shape)


if __name__ == '__main__':
    main()
