import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
import warnings

from mmcv.ops import DeformConv2d, ModulatedDeformConv2d
from mmcv.cnn import ConvModule
from ..builder import NECKS


@NECKS.register_module()
class DcnIncpLatFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 dcn_lat_modulated=False,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(DcnIncpLatFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.dcn_lat_modulated = dcn_lat_modulated

        if end_level == -1:
            self.backbone_end_level = self.num_ins  ## 4
            assert num_outs >= self.num_ins - start_level  ## 4
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_shorcut = nn.ModuleList()

        self.lateral_1x1conv1 = nn.ModuleList()
        self.lateral_dconvs1 = nn.ModuleList()
        self.lateral_offsets1 = nn.ModuleList()

        self.lateral_1x1conv2 = nn.ModuleList()
        self.lateral_dconvs2 = nn.ModuleList()
        self.lateral_offsets2 = nn.ModuleList()

        self.lateral_dconvs3 = nn.ModuleList()
        self.lateral_offsets3 = nn.ModuleList()

        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            deformable_groups = 1
            if self.dcn_lat_modulated:
                conv_op = ModulatedDeformConv2d
                offset_channels = 3
            else:
                conv_op = DeformConv2d
                offset_channels = 2

            shortcut = nn.Conv2d(
                in_channels[i],
                out_channels // 4,
                kernel_size=1
            )

            conv1x1_1 = nn.Conv2d(
                in_channels[i],
                out_channels // 4,
                kernel_size=1
            )
            l_offset1 = nn.Conv2d(
                out_channels // 4,
                deformable_groups * offset_channels,  ## deformable_groups * offset_channels
                kernel_size=1
            )
            l_dconv1 = conv_op(
                out_channels // 4,
                out_channels // 4,
                kernel_size=1,
                deformable_groups=deformable_groups,
                bias=False
            )

            conv1x1_2 = nn.Conv2d(
                in_channels[i],
                out_channels // 4,
                kernel_size=1
            )
            l_offset2 = nn.Conv2d(
                out_channels // 4,
                deformable_groups * offset_channels,  ## deformable_groups * offset_channels
                kernel_size=1
            )
            l_dconv2 = conv_op(
                out_channels // 4,
                out_channels // 4,
                kernel_size=1,
                deformable_groups=deformable_groups,
                bias=False
            )

            l_offset3 = nn.Conv2d(
                in_channels[i],
                deformable_groups * offset_channels,  ## deformable_groups * offset_channels
                kernel_size=1
            )
            l_dconv3 = conv_op(
                in_channels[i],
                out_channels // 4,
                kernel_size=1,
                deformable_groups=deformable_groups,
                bias=False
            )

            # l_conv = ConvModule(
            # 	in_channels[i],
            # 	out_channels,
            # 	1,
            # 	normalize=normalize,
            # 	bias=self.with_bias,
            # 	activation=self.activation, ## No activation.
            # 	inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                conv_cfg=conv_cfg,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_shorcut.append(shortcut)

            self.lateral_1x1conv1.append(conv1x1_1)
            self.lateral_dconvs1.append(l_dconv1)
            self.lateral_offsets1.append(l_offset1)

            self.lateral_1x1conv2.append(conv1x1_2)
            self.lateral_dconvs2.append(l_dconv2)
            self.lateral_offsets2.append(l_offset2)

            self.lateral_dconvs3.append(l_dconv3)
            self.lateral_offsets3.append(l_offset3)

            self.fpn_convs.append(fpn_conv)

        # lvl_id = i - self.start_level
        # setattr(self, 'lateral_conv{}'.format(lvl_id), l_conv)
        # setattr(self, 'fpn_conv{}'.format(lvl_id), fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    conv_cfg=conv_cfg,
                    stride=2,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.fpn_convs.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

        for modules in [self.lateral_shorcut, self.lateral_1x1conv1, self.lateral_1x1conv2]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')

        for modules in [self.lateral_offsets1, self.lateral_offsets2, self.lateral_offsets3]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = []
        # for i in range(len(self.lateral_dconvs)):
        # 	x = inputs[i + self.start_level]
        # 	if self.dcn_lat_modulated:
        # 		offset = self.lateral_offsets[i](x)
        # 		x = self.lateral_dconvs[i](x, offset[:, :18, :, :], offset[:, -9:, :, :].sigmoid())
        # 	else:
        # 		offset = self.lateral_offsets[i](x)
        # 		x = self.lateral_dconvs[i](x, offset)
        # 	laterals.append(x)

        for i in range(len(self.lateral_shorcut)):
            # print(i)
            x = inputs[i + self.start_level]
            x0 = self.lateral_shorcut[i](x)  ## [B,C//4,H,W]

            x1 = self.lateral_1x1conv1[i](x)
            if self.dcn_lat_modulated:
                offset = self.lateral_offsets1[i](x1)
                x1 = self.lateral_dconvs1[i](x1, offset[:, :18, :, :], offset[:, -9:, :, :].sigmoid())
            else:
                offset = self.lateral_offsets1[i](x1)
                x1 = self.lateral_dconvs1[i](x1, offset)

            x2 = self.lateral_1x1conv2[i](x)
            if self.dcn_lat_modulated:
                offset = self.lateral_offsets2[i](x2)
                x2 = self.lateral_dconvs2[i](x2, offset[:, :18, :, :], offset[:, -9:, :, :].sigmoid())
            else:
                offset = self.lateral_offsets2[i](x2)
                x2 = self.lateral_dconvs2[i](x2, offset)

            if self.dcn_lat_modulated:
                offset = self.lateral_offsets3[i](x)
                x3 = self.lateral_dconvs3[i](x, offset[:, :18, :, :], offset[:, -9:, :, :].sigmoid())
            else:
                offset = self.lateral_offsets3[i](x)
                x3 = self.lateral_dconvs3[i](x, offset)

            out = torch.cat([x0, x1, x2, x3], dim=1)  ## [B, 256, H, W]
            laterals.append(out)
        # laterals = [
        # 	lateral_conv(inputs[i + self.start_level])
        # 	for i, lateral_conv in enumerate(self.lateral_dconvs)
        # ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
