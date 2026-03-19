
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from timm.models.layers import DropPath
from typing import List, Tuple
import antialiased_cnns
from fvcore.nn import FlopCountAnalysis


# ------------------------- MDFA-NET model -------------------------

class FRA1(nn.Module):   #FRA1
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.p_conv = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, bias=False),
            norm_layer(dim * 4), #or BN
            act_layer(),
            nn.Conv2d(dim * 4, dim, 1, bias=False)
        )
        self.gate_fn = nn.Sigmoid()

    def forward(self, x):
        att = self.p_conv(x)
        x = x * self.gate_fn(att)
        return x

class FRA2(nn.Module):   #FRA2
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            norm_layer(dim),
            act_layer()
        )

    def forward(self, x):
        return self.conv(x)



class EEA(nn.Module):
    def __init__(self, channel, att_kernel, norm_layer):
        super().__init__()
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.channel = channel
        self.lg_attention = LG_attention()
        self.max_m2 = antialiased_cnns.BlurPool(channel, stride=3)
        self.max_m3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.H_att1 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1),
                                groups=channel, bias=False)
        self.V_att1 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding),
                                groups=channel, bias=False)

        self.H_att2 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1),
                                groups=channel, bias=False)
        self.V_att2 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding),
                                groups=channel, bias=False)

        self.norm = norm_layer(channel)

    def forward(self, x):
        x_tem = self.lg_attention(x_tem)
        x_tem = self.max_m2(x_tem)
        x_h1 = self.H_att1(x_tem)
        x_w1 = self.V_att1(x_tem)
        x_h2 = self.inv_h_transform(self.H_att2(self.h_transform(x_tem)))
        x_w2 = self.inv_v_transform(self.V_att2(self.v_transform(x_tem)))
        att = self.norm(x_h1 + x_w1 + x_h2 + x_w2)
        att = self.max_m3(att)
        out = x[:, :self.channel, :, :] * F.interpolate(
            self.gate_fn(att), size=(x.shape[-2], x.shape[-1]), mode='nearest')
        return out

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)

class ESA3(nn.Module):   #ESA1
    def __init__(self, dim, head_dim=4, num_heads=None,
                 qkv_bias=False, attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 3, 1, 2)
        return x

class ESA1(nn.Module):  #ESA2
    def __init__(self, dim, act_layer):
        super().__init__()
        self.lw_attention = lw_attention()
        self.uppool = nn.MaxUnpool2d((2, 2), 2, padding=0)

        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = act_layer()

        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9,
                                      groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x_, idx = self.lw_attention(x)
        x_ = self.proj_1(x_)
        x_ = self.activation(x_)
        attn1 = self.conv0(x_)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0:1, :, :] + attn2 * sig[:, 1:2, :, :]
        attn = self.conv(attn)
        x_ = x_ * attn
        x_ = self.proj_2(x_)
        x = x + self.uppool(x_, indices=idx)
        return x

class ESA2(nn.Module): 
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = ESA3(dim)
        self.downpool = nn.MaxPool2d(kernel_size=3, stride=3, return_indices=True)
        self.uppool = nn.MaxUnpool2d((3, 3), 3, padding=0)

    def forward(self, x):
        x_, idx = self.downpool(x)
        x = x_ * self.norm(self.attn(x_))
        x = self.uppool(x, indices=idx)
        return x

class DynamicChannelSplit(nn.Module):
    def __init__(self, channels, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = max(4, channels // 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, 2, kernel_size=1, bias=True)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        s = self.pool(x)
        s = self.fc(s)
        return self.softmax(s)

class MLPFusion(nn.Module):
    def __init__(self, channels, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = max(4, channels // 4)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.fuse(x)


# ------------------------- Test Script -------------------------
if __name__ == '__main__':
    {}
