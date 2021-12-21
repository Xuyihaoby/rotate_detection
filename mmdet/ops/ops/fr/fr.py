import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from . import feature_refine_cuda

class FeatureRefineFunction(Function):

    @staticmethod
    def forward(ctx, features, best_rbboxes, spatial_scale, points=1):
        ctx.spatial_scale = spatial_scale
        ctx.points = points
        ctx.save_for_backward(best_rbboxes)
        assert points in [1, 5] 
        assert features.is_cuda
        output = torch.zeros_like(features)
        feature_refine_cuda.forward(features, best_rbboxes, spatial_scale, points, output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        best_rbboxes = ctx.saved_tensors[0]
        points = ctx.points
        spatial_scale = ctx.spatial_scale
        assert grad_output.is_cuda
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.zeros_like(grad_output)
            feature_refine_cuda.backward(grad_output.contiguous(), best_rbboxes, spatial_scale, points, grad_input)
        return grad_input, None, None, None


feature_refine = FeatureRefineFunction.apply


class FR(nn.Module):
    def __init__(self, spatial_scale, points=1):
        super(FR, self).__init__()
        self.spatial_scale = float(spatial_scale)
        self.points = points

    def forward(self, features, best_rbboxes):
        return feature_refine(features, best_rbboxes, self.spatial_scale, self.points)

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += '(spatial_scale={}, points={})'.format(self.spatial_scale, self.points)
        return format_str
