import torch
import torch.nn as nn
from . import FR
from mmcv.cnn import normal_init


class FeatureRefineModule(nn.Module):
    def __init__(
            self,
            in_channels,
            featmap_strides,
            conv_cfg=None,
            norm_cfg=None):
        super(FeatureRefineModule, self).__init__()
        self.in_channels = in_channels
        self.featmap_strides = featmap_strides
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self):
        self.fr = nn.ModuleList([FR(spatial_scale=1 / s)
                                 for s in self.featmap_strides])
        self.conv_5_1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=(5, 1),
            stride=1,
            padding=(2, 0))
        self.conv_1_5 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=(1, 5),
            stride=1,
            padding=(0, 2))
        self.conv_1_1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1)

    def init_weights(self):
        normal_init(self.conv_5_1, std=0.01)
        normal_init(self.conv_1_5, std=0.01)
        normal_init(self.conv_1_1, std=0.01)

    def forward(self, x, best_rbboxes):
        """
        Args:
            x (list[Tensor]):
                feature maps of multiple scales
            best_rbboxes (list[list[Tensor]]):
                best rbboxes of multiple scales of multiple images
        """
        mlvl_rbboxes = [torch.cat(best_rbbox) for best_rbbox in zip(*best_rbboxes)]
        out = []
        for x_scale, best_rbboxes_scale, fr_scale in zip(x, mlvl_rbboxes, self.fr):
            feat_scale_1 = self.conv_5_1(self.conv_1_5(x_scale))
            feat_scale_2 = self.conv_1_1(x_scale)
            feat_scale = feat_scale_1 + feat_scale_2
            feat_refined_scale = fr_scale(feat_scale, best_rbboxes_scale)
            out.append(x_scale + feat_refined_scale)
        return out
