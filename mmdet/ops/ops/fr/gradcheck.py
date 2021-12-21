import os.path as osp
import sys
import math
import torch
from torch.autograd import gradcheck

# sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from . import FR  # noqa: E402, isort:skip
feat_size_x = 32
feat_size_y = 32
stride = 8.0
spatial_scale = 1.0 / stride
base_size = 4.0 * stride
num_imgs = 2
num_chns = 16

feat = torch.randn(num_imgs, num_chns, feat_size_x, feat_size_y, requires_grad=True).cuda()

xc, yc = torch.meshgrid(stride * torch.arange(feat_size_x), stride * torch.arange(feat_size_y))
xc = xc[None,:,:]
yc = yc[None,:,:]
xc = xc + base_size * torch.randn(num_imgs, feat_size_x, feat_size_y)
yc = yc + base_size * torch.randn(num_imgs, feat_size_x, feat_size_y)
w = base_size * torch.randn(num_imgs, feat_size_x, feat_size_y).exp()
h = base_size * torch.randn(num_imgs, feat_size_x, feat_size_y).exp()
a = -math.pi / 2 * torch.rand(num_imgs, feat_size_x, feat_size_y)
bbbox = torch.stack([xc, yc, w, h, a], dim=-1).cuda()
inputs = (feat, bbbox)

print('Gradcheck for FR...')
test = gradcheck(FR(spatial_scale, points=1), inputs, atol=1e-3, eps=1e-3)
print(test)
test = gradcheck(FR(spatial_scale, points=5), inputs, atol=1e-3, eps=1e-3)
print(test)
