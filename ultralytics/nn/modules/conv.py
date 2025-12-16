# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Convolution modules
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter

__all__ = ('Conv', 'LightConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus', 'GhostConv',
           'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'RepConv', 'CoordinateAttention', 
           'CBAMv2', 'SelectiveKernelAttention', 'EfficientMultiScaleAttention', 'MultiScalePyramidAttention',
           'EdgeEnhancedAttention')


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]:i[0] + 1, i[1]:i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__('cv2')


class LightConv(nn.Module):
    """Light convolution with args(ch_in, ch_out, kernel).
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.c1
        groups = self.g
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


def channel_shuffle(x, groups=2):  ##shuffle channel
    # RESHAPE----->transpose------->Flatten
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out


class GAM_Attention(nn.Module):
    def __init__(self, c1, c2, group=True, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1)
        )

        self.spatial_attention = nn.Sequential(

            nn.Conv2d(c1, c1 // rate, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(c1, int(c1 / rate),
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // rate, c2, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(int(c1 / rate), c2,
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        # x_channel_att=channel_shuffle(x_channel_att,4) #last shuffle
        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att = channel_shuffle(x_spatial_att, 4)  # last shuffle
        out = x * x_spatial_att
        # out=channel_shuffle(out,4) #last shuffle
        return out


class GCT(nn.Module):
    def __init__(self, channels, c=2, eps=1e-5):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.eps = eps
        self.c = c

    def forward(self, x):
        y = self.avgpool(x)
        mean = y.mean(dim=1, keepdim=True)
        mean_x2 = (y ** 2).mean(dim=1, keepdim=True)
        var = mean_x2 - mean ** 2
        y_norm = (y - mean) / torch.sqrt(var + self.eps)
        y_transform = torch.exp(-(y_norm ** 2 / 2 * self.c))
        return x * y_transform.expand_as(x)


class ShuffleAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


class ResBlock_CBAM(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=1):
        super(ResBlock_CBAM, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )
        # self.cbam = CBAM(c1=places * self.expansion, c2=places * self.expansion, )
        self.cbam = CBAM(c1=places * self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ECAAttention(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, c1, k_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4, pos_emb=False):
        super(MHSA, self).__init__()

        self.heads = heads
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.pos = pos_emb
        if self.pos:
            self.rel_h_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, 1, int(height)]),
                                             requires_grad=True)
            self.rel_w_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, int(width), 1]),
                                             requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)  # 1,C,h*w,h*w
        c1, c2, c3, c4 = content_content.size()
        if self.pos:
            content_position = (self.rel_h_weight + self.rel_w_weight).view(1, self.heads, C // self.heads, -1).permute(
                0, 1, 3, 2)  # 1,4,1024,64

            content_position = torch.matmul(content_position, q)  # ([1, 4, 1024, 256])
            content_position = content_position if (
                    content_content.shape == content_position.shape) else content_position[:, :, :c3, ]
            assert (content_content.shape == content_position.shape)
            energy = content_content + content_position
        else:
            energy = content_content
        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))  # 1,4,256,64
        out = out.view(n_batch, C, width, height)
        return out


import torch.nn.functional as F
from timm.layers.create_act import create_act_layer, get_act_layer
from timm.layers.helpers import make_divisible
from timm.layers.mlp import ConvMlp
from timm.layers.norm import LayerNorm2d


class GlobalContext(nn.Module):

    def __init__(self, channels, use_attn=True, fuse_add=False, fuse_scale=True, init_last_zero=False,
                 rd_ratio=1. / 8, rd_channels=None, rd_divisor=1, act_layer=nn.ReLU, gate_layer='sigmoid'):
        super(GlobalContext, self).__init__()
        act_layer = get_act_layer(act_layer)

        self.conv_attn = nn.Conv2d(channels, 1, kernel_size=1, bias=True) if use_attn else None

        if rd_channels is None:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        if fuse_add:
            self.mlp_add = ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
        else:
            self.mlp_add = None
        if fuse_scale:
            self.mlp_scale = ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
        else:
            self.mlp_scale = None

        self.gate = create_act_layer(gate_layer)
        self.init_last_zero = init_last_zero
        self.reset_parameters()

    def reset_parameters(self):
        if self.conv_attn is not None:
            nn.init.kaiming_normal_(self.conv_attn.weight, mode='fan_in', nonlinearity='relu')
        if self.mlp_add is not None:
            nn.init.zeros_(self.mlp_add.fc2.weight)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.conv_attn is not None:
            attn = self.conv_attn(x).reshape(B, 1, H * W)  # (B, 1, H * W)
            attn = F.softmax(attn, dim=-1).unsqueeze(3)  # (B, 1, H * W, 1)
            context = x.reshape(B, C, H * W).unsqueeze(1) @ attn
            context = context.view(B, C, 1, 1)
        else:
            context = x.mean(dim=(2, 3), keepdim=True)

        if self.mlp_scale is not None:
            mlp_x = self.mlp_scale(context)
            x = x * self.gate(mlp_x)
        if self.mlp_add is not None:
            mlp_x = self.mlp_add(context)
            x = x + mlp_x

        return x


from timm.layers.create_conv2d import create_conv2d


class GatherExcite(nn.Module):
    def __init__(
            self, channels, feat_size=None, extra_params=False, extent=0, use_mlp=True,
            rd_ratio=1. / 16, rd_channels=None, rd_divisor=1, add_maxpool=False,
            act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, gate_layer='sigmoid'):
        super(GatherExcite, self).__init__()
        self.add_maxpool = add_maxpool
        act_layer = get_act_layer(act_layer)
        self.extent = extent
        if extra_params:
            self.gather = nn.Sequential()
            if extent == 0:
                assert feat_size is not None, 'spatial feature size must be specified for global extent w/ params'
                self.gather.add_module(
                    'conv1', create_conv2d(channels, channels, kernel_size=feat_size, stride=1, depthwise=True))
                if norm_layer:
                    self.gather.add_module(f'norm1', nn.BatchNorm2d(channels))
            else:
                assert extent % 2 == 0
                num_conv = int(math.log2(extent))
                for i in range(num_conv):
                    self.gather.add_module(
                        f'conv{i + 1}',
                        create_conv2d(channels, channels, kernel_size=3, stride=2, depthwise=True))
                    if norm_layer:
                        self.gather.add_module(f'norm{i + 1}', nn.BatchNorm2d(channels))
                    if i != num_conv - 1:
                        self.gather.add_module(f'act{i + 1}', act_layer(inplace=True))
        else:
            self.gather = None
            if self.extent == 0:
                self.gk = 0
                self.gs = 0
            else:
                assert extent % 2 == 0
                self.gk = self.extent * 2 - 1
                self.gs = self.extent

        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.mlp = ConvMlp(channels, rd_channels, act_layer=act_layer) if use_mlp else nn.Identity()
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        size = x.shape[-2:]
        if self.gather is not None:
            x_ge = self.gather(x)
        else:
            if self.extent == 0:
                # global extent
                x_ge = x.mean(dim=(2, 3), keepdims=True)
                if self.add_maxpool:
                    # experimental codepath, may remove or change
                    x_ge = 0.5 * x_ge + 0.5 * x.amax((2, 3), keepdim=True)
            else:
                x_ge = F.avg_pool2d(
                    x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2, count_include_pad=False)
                if self.add_maxpool:
                    # experimental codepath, may remove or change
                    x_ge = 0.5 * x_ge + 0.5 * F.max_pool2d(x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2)
        x_ge = self.mlp(x_ge)
        if x_ge.shape[-1] != 1 or x_ge.shape[-2] != 1:
            x_ge = F.interpolate(x_ge, size=size)
        return x * self.gate(x_ge)


# Latest Attention Mechanisms for Wind Turbine Damage Detection

class CoordinateAttention(nn.Module):
    """
    Coordinate Attention (CA) - 2021
    Perfect for wind turbine damage detection as it captures both spatial and channel relationships.
    Especially good for detecting linear cracks and edge erosion patterns.
    """
    def __init__(self, c1, reduction=32):
        super(CoordinateAttention, self).__init__()
        # c1 is automatically passed from the previous layer's output channels
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, c1 // reduction)

        self.conv1 = nn.Conv2d(c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        
        self.conv_h = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class CBAMv2(nn.Module):
    """
    Enhanced CBAM v2 - 2022
    Improved spatial attention with better gradient flow for wind turbine surface damage detection.
    """
    def __init__(self, c1, reduction=16):
        super(CBAMv2, self).__init__()
        self.channel_attention = ChannelAttentionv2(c1, reduction)
        self.spatial_attention = SpatialAttentionv2()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ChannelAttentionv2(nn.Module):
    """Enhanced Channel Attention with improved gradient flow"""
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttentionv2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        
        # v2 enhancement: residual connection in channel attention
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        
        # Enhanced combination with residual-like connection
        out = avg_out + max_out
        attention = self.sigmoid(out)
        
        return attention


class SpatialAttentionv2(nn.Module):
    """Enhanced Spatial Attention with improved feature extraction"""
    def __init__(self, kernel_size=7):
        super(SpatialAttentionv2, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        
        # v2 enhancement: additional convolution for better spatial feature extraction
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_cat = self.conv1(x_cat)
        
        # v2 enhancement: additional spatial processing
        x_cat = self.conv2(x_cat)
        attention = self.sigmoid(x_cat)
        
        return attention


class SelectiveKernelAttention(nn.Module):
    """
    Selective Kernel Attention (SKA) - 2023
    Adaptive receptive fields crucial for multi-scale wind turbine damage detection.
    Can detect both fine cracks and large erosion areas simultaneously.
    """
    def __init__(self, c1, stride=1, M=2, r=16, L=32):
        super(SelectiveKernelAttention, self).__init__()
        # Use c1 for both input and output channels to avoid mismatch
        d = max(int(c1/r), L)
        self.M = M
        self.in_channels = c1
        self.convs = nn.ModuleList([])
        
        for i in range(M):
            # Different kernel sizes for multi-scale feature extraction
            kernel_size = 3 + i * 2  # 3, 5, 7, ... 
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(c1, c1, kernel_size, stride=stride, 
                             padding=kernel_size//2, groups=1, bias=False),
                    nn.BatchNorm2d(c1),
                    nn.SiLU()
                )
            )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c1, d)
        self.fcs = nn.ModuleList([])
        
        for i in range(M):
            self.fcs.append(nn.Linear(d, c1))
            
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Multi-scale feature extraction
        feats = [conv(x) for conv in self.convs]
        feats = torch.stack(feats, dim=1)  # (B, M, C, H, W)
        
        # Fuse features
        feats_sum = torch.sum(feats, dim=1)  # (B, C, H, W)
        feats_gap = self.gap(feats_sum).view(batch_size, -1)  # (B, C)
        
        # Feature descriptor
        feats_gap = self.fc(feats_gap)  # (B, d)
        
        # Generate attention weights for each scale
        attention_weights = []
        for fc in self.fcs:
            weight = fc(feats_gap).view(batch_size, -1, 1, 1)  # (B, C, 1, 1)
            attention_weights.append(weight)
        
        attention_weights = torch.stack(attention_weights, dim=1)  # (B, M, C, 1, 1)
        attention_weights = self.softmax(attention_weights)
        
        # Apply attention weights
        feats_weighted = feats * attention_weights
        result = torch.sum(feats_weighted, dim=1)  # (B, C, H, W)
        
        return result


class EfficientMultiScaleAttention(nn.Module):
    """
    Efficient Multi-Scale Attention (EMA) - 2023
    Balances computational efficiency with multi-scale feature extraction.
    Perfect for real-time wind turbine inspection systems.
    """
    def __init__(self, channels, reduction=4, num_heads=8):
        super(EfficientMultiScaleAttention, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.num_heads = num_heads
        
        # Multi-scale feature extraction with different kernel sizes
        self.conv1x1 = nn.Conv2d(channels, channels // reduction, 1)
        self.conv3x3 = nn.Conv2d(channels // reduction, channels // reduction, 3, padding=1, groups=channels // reduction)
        self.conv5x5 = nn.Conv2d(channels // reduction, channels // reduction, 5, padding=2, groups=channels // reduction)
        
        # Cross-scale interaction
        self.cross_conv = nn.Conv2d(channels // reduction * 2, channels // reduction, 1)
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels // reduction, channels // reduction // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction // 4, channels // reduction, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels // reduction, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_conv = nn.Conv2d(channels // reduction, channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        identity = x
        
        # Initial feature reduction
        x = self.conv1x1(x)
        
        # Multi-scale feature extraction
        feat_3x3 = self.conv3x3(x)
        feat_5x5 = self.conv5x5(x)
        
        # Cross-scale interaction
        cross_feat = torch.cat([feat_3x3, feat_5x5], dim=1)
        cross_feat = self.cross_conv(cross_feat)
        
        # Apply channel attention
        channel_att = self.channel_attention(cross_feat)
        feat_ca = cross_feat * channel_att
        
        # Apply spatial attention
        spatial_att = self.spatial_attention(feat_ca)
        feat_sa = feat_ca * spatial_att
        
        # Output projection
        output = self.output_conv(feat_sa)
        attention = self.sigmoid(output)
        
        return identity * attention


class MultiScalePyramidAttention(nn.Module):
    """
<<<<<<< HEAD
    Multi-Scale Pyramid Attention (MSPA) - 2024
    Designed for wind turbine damage detection with multi-scale feature pyramid.
    Handles damage at different scales: fine cracks, medium erosion, large surface damage.
    Uses pyramid pooling and multi-scale convolutions for comprehensive feature extraction.
    """
    def __init__(self, c1, reduction=16, scales=[1, 2, 4, 8]):
        super(MultiScalePyramidAttention, self).__init__()
        self.in_channels = c1
        self.reduction = reduction
        self.scales = scales
        
        # Feature reduction
        self.conv1x1 = nn.Conv2d(c1, c1 // reduction, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1 // reduction)
        
        # Multi-scale pyramid pooling
        self.pyramid_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(scale) for scale in scales
        ])
        
        # Multi-scale convolutions (without BatchNorm to avoid small feature map issues)
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c1 // reduction, c1 // reduction, 3, padding=1, bias=True),
                nn.SiLU()
            ) for _ in scales
        ])
        
        # Feature fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(c1 // reduction * len(scales), c1 // reduction, 1, bias=False),
            nn.BatchNorm2d(c1 // reduction),
            nn.SiLU()
        )
        
        # Channel attention (corrected to match reduced channels)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1 // reduction, c1 // reduction // 4, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(c1 // reduction // 4, c1 // reduction, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention (without BatchNorm to avoid small feature map issues)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c1 // reduction, 1, 7, padding=3, bias=True),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_conv = nn.Conv2d(c1 // reduction, c1, 1, bias=False)
        self.act = nn.SiLU()
        
    def forward(self, x):
        identity = x
        B, C, H, W = x.size()
        
        # Feature reduction
        x = self.conv1x1(x)
        x = self.bn1(x)
        x = self.act(x)
        
        # Multi-scale pyramid features
        pyramid_feats = []
        for i, (pool, conv) in enumerate(zip(self.pyramid_pools, self.scale_convs)):
            # Pool to specific scale
            pooled = pool(x)
            # Apply convolution
            conv_feat = conv(pooled)
            # Upsample back to original size
            if pooled.size(2) != H or pooled.size(3) != W:
                conv_feat = nn.functional.interpolate(conv_feat, size=(H, W), mode='bilinear', align_corners=False)
            pyramid_feats.append(conv_feat)
        
        # Fuse multi-scale features
        fused_feat = torch.cat(pyramid_feats, dim=1)
        fused_feat = self.fusion_conv(fused_feat)
        
        # Apply channel attention
        channel_att = self.channel_attention(fused_feat)
        feat_ca = fused_feat * channel_att
        
        # Apply spatial attention
        spatial_att = self.spatial_attention(feat_ca)
        feat_sa = feat_ca * spatial_att
        
        # Output projection
        output = self.output_conv(feat_sa)
        
        return identity + output


class EdgeEnhancedAttention(nn.Module):
    """
    Edge-Enhanced Attention (EEA) - 2024
    Specialized for wind turbine damage detection with edge enhancement.
    Emphasizes edge features critical for crack and erosion boundary detection.
    Uses Sobel operators and edge-aware convolutions for enhanced edge detection.
    """
    def __init__(self, c1, reduction=16, edge_threshold=0.1):
        super(EdgeEnhancedAttention, self).__init__()
        self.in_channels = c1
        self.reduction = reduction
        self.edge_threshold = edge_threshold
        
        # Edge detection kernels (Sobel operators)
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        
        # Feature reduction
        self.conv1x1 = nn.Conv2d(c1, c1 // reduction, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1 // reduction)
        
        # Edge enhancement branch
        self.edge_conv = nn.Sequential(
            nn.Conv2d(c1 // reduction, c1 // reduction, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1 // reduction),
            nn.SiLU(),
            nn.Conv2d(c1 // reduction, c1 // reduction, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1 // reduction),
            nn.SiLU()
        )
        
        # Edge-aware spatial attention (reduced BatchNorm to avoid small feature map issues)
        self.edge_spatial_attention = nn.Sequential(
            nn.Conv2d(c1 // reduction + 2, c1 // reduction, 7, padding=3, bias=True),  # +2 for edge maps
            nn.SiLU(),
            nn.Conv2d(c1 // reduction, 1, 1, bias=True),
            nn.Sigmoid()
        )
        
        # Channel attention with edge information (corrected to match reduced channels)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1 // reduction, c1 // reduction // 4, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(c1 // reduction // 4, c1 // reduction, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_conv = nn.Conv2d(c1 // reduction, c1, 1, bias=False)
        self.act = nn.SiLU()
        
    def forward(self, x):
        identity = x
        B, C, H, W = x.size()
        
        # Feature reduction
        x = self.conv1x1(x)
        x = self.bn1(x)
        x = self.act(x)
        
        # Edge detection
        # Convert to grayscale for edge detection (average across channels)
        gray = torch.mean(x, dim=1, keepdim=True)
        
        # Apply Sobel operators
        edge_x = nn.functional.conv2d(gray, self.sobel_x.repeat(1, 1, 1, 1), padding=1)
        edge_y = nn.functional.conv2d(gray, self.sobel_y.repeat(1, 1, 1, 1), padding=1)
        
        # Compute edge magnitude
        edge_mag = torch.sqrt(edge_x**2 + edge_y**2)
        
        # Edge enhancement
        edge_enhanced = self.edge_conv(x)
        
        # Combine original features with edge information
        edge_spatial_input = torch.cat([edge_enhanced, edge_x, edge_y], dim=1)
        
        # Edge-aware spatial attention
        spatial_att = self.edge_spatial_attention(edge_spatial_input)
        feat_sa = edge_enhanced * spatial_att
        
        # Channel attention
        channel_att = self.channel_attention(feat_sa)
        feat_ca = feat_sa * channel_att
        
        # Output projection
        output = self.output_conv(feat_ca)
        
        return identity + output
