import torch
import torch.nn as nn

from ...registry import SEGMENTAL_CONSENSUSES


class _SimpleConsensus(torch.autograd.Function):
    """Simplest segmental consensus module"""

    @staticmethod
    def forward(ctx,x,dim,consensus_type):
        ctx.dim = dim
        ctx.consensus_type=consensus_type
        ctx.save_for_backward(x)
        if consensus_type == 'avg':
            output = x.mean(dim=dim, keepdim=True)
        else:
            output = None
        return output


    @staticmethod
    def backward( ctx,grad_output):
        x, = ctx.saved_tensors
        dim = ctx.dim
        consensus_type=ctx.consensus_type
        shape = x.size()
        if consensus_type == 'avg':
            grad_in = grad_output.expand(shape) / float(shape[dim])
        else:
            grad_in = None
        return grad_in, None , None


@SEGMENTAL_CONSENSUSES.register_module
class SimpleConsensus(nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(SimpleConsensus, self).__init__()

        assert consensus_type in ['avg']
        self.consensus_type = consensus_type
        self.dim = dim

    def init_weights(self):
        pass

    def forward(self, input):
        return _SimpleConsensus.apply(input,
                                    self.dim,
                                    self.consensus_type)
