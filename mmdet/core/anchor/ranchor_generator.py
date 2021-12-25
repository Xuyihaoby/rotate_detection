import torch
from .builder import ANCHOR_GENERATORS
from . import AnchorGenerator
from torch.nn.modules.utils import _pair
import numpy as np


# adapted from https://github.com/SJTU-Thinklab-Det/r3det-on-mmdetection

@ANCHOR_GENERATORS.register_module()
class RAnchorGenerator(AnchorGenerator):
    """Non-Standard XYWHA anchor generator for rotated anchor-based detectors

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels.
        ratios (list[float]): The list of ratios between the height and width
            of anchors in a single level.
        scales (list[int] | None): Anchor scales for anchors in a single level.
            It cannot be set at the same time if `octave_base_scale` and
            `scales_per_octave` are set.
        angles (list[float] | None): Anchor angles for anchors in a single level.
            If None is given, angles will be set to zero.
        base_sizes (list[int] | None): The basic sizes
            of anchors in multiple levels.
            If None is given, strides will be used as base_sizes.
            (If strides are non square, the shortest stride is taken.)
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales. By default it is True in V2.0
        octave_base_scale (int): The base scale of octave.
        scales_per_octave (int): Number of scales for each octave.
            `octave_base_scale` and `scales_per_octave` are usually used in
            retinanet and the `scales` should be None when they are set.
        centers (list[tuple[float, float]] | None): The centers of the anchor
            relative to the feature grid center in multiple feature levels.
            By default it is set to be None and not used. If a list of tuple of
            float is given, they will be used to shift the centers of anchors.
        center_offset (float): The offset of center in propotion to anchors'
            width and height. By default it is 0 in V2.0.

    Examples:
        >>> from mmdet.core import RAnchorGenerator
        >>> self = RAnchorGenerator([8.], [1.], [4.], [0.])
        >>> all_anchors = self.grid_anchors([(2, 2)], device='cpu')
        >>> print(all_anchors)
        [tensor([[ 0.,  0., 32., 32.,  0.],
                [ 8.,  0., 32., 32.,  0.],
                [ 0.,  8., 32., 32.,  0.],
                [ 8.,  8., 32., 32.,  0.]])]
    """

    def __init__(self,
                 strides,
                 ratios,
                 scales=None,
                 angles=None,
                 base_sizes=None,
                 scale_major=True,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 centers=None,
                 center_offset=0.,
                 version='v1'):
        self.format = version
        if angles is None:
            angles = [0.]
        elif self.format == 'v1':
            angles = self._checkOpencvformat(angles)
        self.angles = torch.Tensor(angles)
        super(RAnchorGenerator, self).__init__(
            strides,
            ratios,
            scales,
            base_sizes,
            scale_major,
            octave_base_scale,
            scales_per_octave,
            centers,
            center_offset
        )

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios

        assert self.scale_major, "RAnchorGenerator only support scale-major anchors!"
        ws = (w * w_ratios[:, None, None] * self.scales[None, :, None] *
              torch.ones_like(self.angles)[None, None, :]).view(-1)
        hs = (h * h_ratios[:, None, None] * self.scales[None, :, None] *
              torch.ones_like(self.angles)[None, None, :]).view(-1)
        angles = self.angles.repeat(len(self.scales) * len(self.ratios))

        # use float anchor and the anchor's center is aligned with the
        # pixel center

        x_center += torch.zeros_like(ws)
        y_center += torch.zeros_like(ws)
        base_anchors = torch.stack(
            [x_center, y_center, ws, hs, angles], dim=-1)

        return base_anchors

    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=(16, 16),
                                  device='cuda'):
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride[0]
        shift_y = torch.arange(0, feat_h, device=device) * stride[1]
        # try to make shiftxx and shiftyy do component
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        # the following is new added
        shift_others = torch.zeros_like(shift_xx)
        shifts = torch.stack(
            [shift_xx, shift_yy, shift_others, shift_others, shift_others], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 5) to K shifts (K, 1, 5) to get
        # shifted anchors (K, A, 5), reshape to (K*A, 5)

        # base anchors (0, 0, w, h, angle) shape(base_num, 5) and shifts shape (65536, 5)
        # all anchors shape (shiftxx*shiftyy, baseanchor_num, 5)
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 5)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def _checkOpencvformat(self, angles):
        new_angles = []
        for angle in angles:
            while not 0 > angle >= -90:
                if angle >= 0:
                    angle -= 90
                else:
                    angle += 90
            angle = angle / 180 * np.pi
            assert 0 > angle >= -np.pi / 2
            new_angles.append(angle)
        return new_angles

    def __repr__(self):
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides},\n'
        repr_str += f'{indent_str}ratios={self.ratios},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}angles={self.angles},\n'
        repr_str += f'{indent_str}base_sizes={self.base_sizes},\n'
        repr_str += f'{indent_str}scale_major={self.scale_major},\n'
        repr_str += f'{indent_str}octave_base_scale='
        repr_str += f'{self.octave_base_scale},\n'
        repr_str += f'{indent_str}scales_per_octave='
        repr_str += f'{self.scales_per_octave},\n'
        repr_str += f'{indent_str}num_levels={self.num_levels}\n'
        repr_str += f'{indent_str}centers={self.centers},\n'
        repr_str += f'{indent_str}center_offset={self.center_offset})'
        return repr_str


@ANCHOR_GENERATORS.register_module()
class PseudoAnchorGenerator(AnchorGenerator):
    """Non-Standard pseudo anchor generator that is used to generate valid flags only!
       Calling its grid_anchors() method will raise NotImplementedError!
    """

    def __init__(self,
                 strides):
        self.strides = [_pair(stride) for stride in strides]

    @property
    def num_base_anchors(self):
        return [1 for _ in self.strides]

    def single_level_grid_anchors(self, featmap_sizes, device='cuda'):
        raise NotImplementedError

    def __repr__(self):
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides})'
        return repr_str

