import torch
import sys
import collections

model = torch.load(sys.argv[1])

weight = model['state_dict']

out = collections.OrderedDict()
for k, v in weight.items():
    name = k.replace('backbone.', '').replace('cls_head.', '')
    out[name] = v.cpu()
    print(name)

torch.save(out, sys.argv[2])
