from timm.models.layers import trunc_normal_
import torch
from torch import nn
from timm.models.layers import DropPath
import math
import torch.nn.functional as F
from functools import partial
from mmdet.core.custom import load_checkpoint
from mmdet.utils import get_root_logger
from ..builder import BACKBONES


class ConvEncoder(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvEncoderBNHS(nn.Module):
    """
        Conv. Encoder with Batch Norm and Hard-Swish Activation
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.Hardswish()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class SDTAEncoder(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4,
                 use_pos_emb=True, num_heads=8, qkv_bias=True, attn_drop=0., drop=0., scales=1):
        super().__init__()
        width = max(int(math.ceil(dim / scales)), int(math.floor(dim // scales)))
        self.width = width
        if scales == 1:
            self.nums = 1
        else:
            self.nums = scales - 1
        convs = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width))
        self.convs = nn.ModuleList(convs)

        self.pos_embd = None
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=dim)
        self.norm_xca = LayerNorm(dim, eps=1e-6)
        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()  # TODO: MobileViT is using 'swish'
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        x = torch.cat((out, spx[self.nums]), 1)
        # XCA
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        if self.pos_embd:
            pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding
        x = x + self.drop_path(self.gamma_xca * self.xca(self.norm_xca(x)))
        x = x.reshape(B, H, W, C)

        # Inverted Bottleneck
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x


class SDTAEncoderBNHS(nn.Module):
    """
        SDTA Encoder with Batch Norm and Hard-Swish Activation
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4,
                 use_pos_emb=True, num_heads=8, qkv_bias=True, attn_drop=0., drop=0., scales=1):
        super().__init__()
        width = max(int(math.ceil(dim / scales)), int(math.floor(dim // scales)))
        self.width = width
        if scales == 1:
            self.nums = 1
        else:
            self.nums = scales - 1
        convs = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width))
        self.convs = nn.ModuleList(convs)

        self.pos_embd = None
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=dim)
        self.norm_xca = nn.BatchNorm2d(dim)
        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.Hardswish()  # TODO: MobileViT is using 'swish'
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        x = torch.cat((out, spx[self.nums]), 1)
        # XCA
        x = self.norm_xca(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        if self.pos_embd:
            pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding
        x = x + self.drop_path(self.gamma_xca * self.xca(x))
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Inverted Bottleneck
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x


class XCA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class PositionalEncodingFourier(nn.Module):
    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)

        return pos


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


@BACKBONES.register_module()
class EdgeNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[24, 48, 88, 168],
                 global_block=[0, 0, 0, 3], global_block_type=['None', 'None', 'None', 'SDTA'],
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1., expan_ratio=4,
                 kernel_sizes=[7, 7, 7, 7], heads=[8, 8, 8, 8], use_pos_embd_xca=[False, False, False, False],
                 use_pos_embd_global=False, d2_scales=[2, 3, 4, 5], out_indices=[0, 1, 2, 3], **kwargs):
        super().__init__()
        for g in global_block_type:
            assert g in ['None', 'SDTA']
        if use_pos_embd_global:
            self.pos_embd = PositionalEncodingFourier(dim=dims[0])
        else:
            self.pos_embd = None
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                if j > depths[i] - global_block[i] - 1:
                    if global_block_type[i] == 'SDTA':
                        stage_blocks.append(SDTAEncoder(dim=dims[i], drop_path=dp_rates[cur + j],
                                                        expan_ratio=expan_ratio, scales=d2_scales[i],
                                                        use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i]))
                    else:
                        raise NotImplementedError
                else:
                    stage_blocks.append(ConvEncoder(dim=dims[i], drop_path=dp_rates[cur + j],
                                                    layer_scale_init_value=layer_scale_init_value,
                                                    expan_ratio=expan_ratio, kernel_size=kernel_sizes[i]))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += depths[i]
        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # Final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)
        self.out_indices = out_indices
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

        # self.head_dropout = nn.Dropout(kwargs["classifier_dropout"])
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):  # TODO: MobileViT is using 'kaiming_normal' for initializing conv layers
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        if isinstance(pretrained, str):
            # self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x):
        outs = []
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)  # 四倍下采样
        if self.pos_embd:
            B, C, H, W = x.shape
            x = x + self.pos_embd(B, H, W)
        if 0 in self.out_indices:
            norm_layer = getattr(self, f'norm{0}')
            x_out = norm_layer(x)
            outs.append(x_out)
        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)  # Global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        return x

# @register_model
# def edgenext_xx_small(pretrained=False, **kwargs):
#     # 1.33M & 260.58M @ 256 resolution
#     # 71.23% Top-1 accuracy
#     # No AA, Color Jitter=0.4, No Mixup & Cutmix, DropPath=0.0, BS=4096, lr=0.006, multi-scale-sampler
#     # Jetson FPS=51.66 versus 47.67 for MobileViT_XXS
#     # For A100: FPS @ BS=1: 212.13 & @ BS=256: 7042.06 versus FPS @ BS=1: 96.68 & @ BS=256: 4624.71 for MobileViT_XXS
#     model = EdgeNeXt(depths=[2, 2, 6, 2], dims=[24, 48, 88, 168], expan_ratio=4,
#                      global_block=[0, 1, 1, 1],
#                      global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
#                      use_pos_embd_xca=[False, True, False, False],
#                      kernel_sizes=[3, 5, 7, 9],
#                      heads=[4, 4, 4, 4],
#                      d2_scales=[2, 2, 3, 4],
#                      **kwargs)
#
#     return model


# @register_model
# def edgenext_x_small(pretrained=False, **kwargs):
#     # 2.34M & 538.0M @ 256 resolution
#     # 75.00% Top-1 accuracy
#     # No AA, No Mixup & Cutmix, DropPath=0.0, BS=4096, lr=0.006, multi-scale-sampler
#     # Jetson FPS=31.61 versus 28.49 for MobileViT_XS
#     # For A100: FPS @ BS=1: 179.55 & @ BS=256: 4404.95 versus FPS @ BS=1: 94.55 & @ BS=256: 2361.53 for MobileViT_XS
#     model = EdgeNeXt(depths=[3, 3, 9, 3], dims=[32, 64, 100, 192], expan_ratio=4,
#                      global_block=[0, 1, 1, 1],
#                      global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
#                      use_pos_embd_xca=[False, True, False, False],
#                      kernel_sizes=[3, 5, 7, 9],
#                      heads=[4, 4, 4, 4],
#                      d2_scales=[2, 2, 3, 4],
#                      **kwargs)
#
#     return model
#
#
# @register_model
# def edgenext_small(pretrained=False, **kwargs):
#     # 5.59M & 1260.59M @ 256 resolution
#     # 79.43% Top-1 accuracy
#     # AA=True, No Mixup & Cutmix, DropPath=0.1, BS=4096, lr=0.006, multi-scale-sampler
#     # Jetson FPS=20.47 versus 18.86 for MobileViT_S
#     # For A100: FPS @ BS=1: 172.33 & @ BS=256: 3010.25 versus FPS @ BS=1: 93.84 & @ BS=256: 1785.92 for MobileViT_S
#     model = EdgeNeXt(depths=[3, 3, 9, 3], dims=[48, 96, 160, 304], expan_ratio=4,
#                      global_block=[0, 1, 1, 1],
#                      global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
#                      use_pos_embd_xca=[False, True, False, False],
#                      kernel_sizes=[3, 5, 7, 9],
#                      d2_scales=[2, 2, 3, 4],
#                      **kwargs)
#
#     return model
#
#
# @register_model
# def edgenext_base(pretrained=False, **kwargs):
#     # 18.51M & 3840.93M @ 256 resolution
#     # 82.5% (normal) 83.7% (USI) Top-1 accuracy
#     # AA=True, Mixup & Cutmix, DropPath=0.1, BS=4096, lr=0.006, multi-scale-sampler
#     # Jetson FPS=xx.xx versus xx.xx for MobileViT_S
#     # For A100: FPS @ BS=1: xxx.xx & @ BS=256: xxxx.xx
#     model = EdgeNeXt(depths=[3, 3, 9, 3], dims=[80, 160, 288, 584], expan_ratio=4,
#                      global_block=[0, 1, 1, 1],
#                      global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
#                      use_pos_embd_xca=[False, True, False, False],
#                      kernel_sizes=[3, 5, 7, 9],
#                      d2_scales=[2, 2, 3, 4],
#                      **kwargs)
#
#     return model
#
#
# """
#     Using BN & HSwish instead of LN & GeLU
# """
#
#
# @register_model
# def edgenext_xx_small_bn_hs(pretrained=False, **kwargs):
#     # 1.33M & 259.53M @ 256 resolution
#     # 70.33% Top-1 accuracy
#     # For A100: FPS @ BS=1: 219.66 & @ BS=256: 10359.98
#     model = EdgeNeXtBNHS(depths=[2, 2, 6, 2], dims=[24, 48, 88, 168], expan_ratio=4,
#                          global_block=[0, 1, 1, 1],
#                          global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS', 'SDTA_BN_HS'],
#                          use_pos_embd_xca=[False, True, False, False],
#                          kernel_sizes=[3, 5, 7, 9],
#                          heads=[4, 4, 4, 4],
#                          d2_scales=[2, 2, 3, 4],
#                          **kwargs)
#
#     return model
#
#
# @register_model
# def edgenext_x_small_bn_hs(pretrained=False, **kwargs):
#     # 2.34M & 535.84M @ 256 resolution
#     # 74.87% Top-1 accuracy
#     # For A100: FPS @ BS=1: 179.25 & @ BS=256: 6059.59
#     model = EdgeNeXtBNHS(depths=[3, 3, 9, 3], dims=[32, 64, 100, 192], expan_ratio=4,
#                          global_block=[0, 1, 1, 1],
#                          global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS', 'SDTA_BN_HS'],
#                          use_pos_embd_xca=[False, True, False, False],
#                          kernel_sizes=[3, 5, 7, 9],
#                          heads=[4, 4, 4, 4],
#                          d2_scales=[2, 2, 3, 4],
#                          **kwargs)
#
#     return model
#
#
# @register_model
# def edgenext_small_bn_hs(pretrained=False, **kwargs):
#     # 5.58M & 1257.28M @ 256 resolution
#     # 78.39% Top-1 accuracy
#     # For A100: FPS @ BS=1: 174.68 & @ BS=256: 3808.19
#     model = EdgeNeXtBNHS(depths=[3, 3, 9, 3], dims=[48, 96, 160, 304], expan_ratio=4,
#                          global_block=[0, 1, 1, 1],
#                          global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS', 'SDTA_BN_HS'],
#                          use_pos_embd_xca=[False, True, False, False],
#                          kernel_sizes=[3, 5, 7, 9],
#                          d2_scales=[2, 2, 3, 4],
#                          **kwargs)
#
#     return model
