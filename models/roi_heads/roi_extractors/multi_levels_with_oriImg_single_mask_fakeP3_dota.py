from mmcv.cnn.bricks import build_plugin_layer
from mmcv.runner import force_fp32

from mmdet.models.builder import ROI_EXTRACTORS
from .base_roi_extractor import BaseRoIExtractor

import torch
import torch.nn as nn
from mmcv import ops


@ROI_EXTRACTORS.register_module()
class MultiRoIWithOriginalImageSingleMaskFakeP3ExtractorDOTA(BaseRoIExtractor):
    """Extract RoI features from all level feature maps levels.

    This is the implementation of `A novel Region of Interest Extraction Layer
    for Instance Segmentation <https://arxiv.org/abs/2004.13665>`_.

    Args:
        aggregation (str): The method to aggregate multiple feature maps.
            Options are 'sum', 'concat'. Default: 'sum'.
        pre_cfg (dict | None): Specify pre-processing modules. Default: None.
        post_cfg (dict | None): Specify post-processing modules. Default: None.
        kwargs (keyword arguments): Arguments that are the same
            as :class:`BaseRoIExtractor`.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56):
        super(MultiRoIWithOriginalImageSingleMaskFakeP3ExtractorDOTA, self).__init__(roi_layer, out_channels,
                                                                               featmap_strides)
        # TODO: check
        self.finest_scale = finest_scale

    def build_roi_layers(self, layer_cfg, featmap_strides):
        """Build RoI operator to extract feature from each level feature map.

        Args:
            layer_cfg (dict): Dictionary to construct and config RoI layer
                operation. Options are modules under ``mmcv/ops`` such as
                ``RoIAlign``.
            featmap_strides (int): The stride of input feature map w.r.t to the
                original image size, which would be used to scale RoI
                coordinate (original image coordinate system) to feature
                coordinate system.

        Returns:
            nn.ModuleList: The RoI extractor modules for each level feature
                map.
        """

        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        # 不仅是将feature map上的进行roialign 还要在原图和mask上进行同样的操作
        # 只需要加上这一句，难道就有很好的效果了吗
        roi_layers.append(layer_cls(spatial_scale=1 / 8, **cfg))  ## for single mask like P2-style
        roi_layers.append(layer_cls(spatial_scale=1, **cfg))  ## for original image
        return roi_layers

    @force_fp32(apply_to=('feats',), out_fp16=True)
    def forward(self, feats, rois, img, seg_fea, mask_lvls=None, roi_scale_factor=None):
        """Forward function."""
        # feats: (P2, P3, P4, P5)
        # norm_img: normalized original image
        # seg_fea: [B,C,H,W]
        # masks: None or list(P2~P5, each is [B,1,H,W])
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        # 这里添加了新的
        if mask_lvls is not None:
            mask_lvls = mask_lvls + [None, None]
        else:
            mask_lvls = [None] * 6

        out_size = self.roi_layers[0].output_size
        roi_feats = [feats[0].new_zeros(rois.size(0), self.out_channels, *out_size)
                     for _ in range(len(feats) + 1)]

        #TOD: 查看结果
        # feats is tuple
        feats = feats + (seg_fea, img)
        roi_feats.append(feats[0].new_zeros(rois.size(0), 3, *out_size))

        assert len(feats) == len(mask_lvls) == len(self.featmap_strides) + 2
        num_levels = len(feats)  # len 6

        # some times rois is an empty tensor
        if len(roi_feats) == 0:
            return roi_feats

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)


        for i in range(num_levels):
            if mask_lvls[i] is not None:
                roi_feats_t = self.roi_layers[i](feats[i].mul(mask_lvls[i].sigmoid()), rois)
            else:
                roi_feats_t = self.roi_layers[i](feats[i], rois)
            roi_feats[i] += roi_feats_t

        return roi_feats
