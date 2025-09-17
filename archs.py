import torch
from torch import nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
# 假设utils里是常用工具函数，按实际补充
from utils import *  
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
from pdb import set_trace as st

from kan import KANLinear, KAN
from torch.nn import init


class AttentionGatedFusion(nn.Module):
    """注意力增强的门控融合"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 特征重标定
        self.feature_recalibration = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x1, x2):
        concat_feat = torch.cat([x1, x2], dim=1)
        
        # 通道注意力
        channel_att = self.channel_attention(concat_feat)
        
        # 空间注意力  
        spatial_att = self.spatial_attention(concat_feat)
        
        # 门控权重
        gate_weight = self.gate_network(concat_feat)
        
        # 综合注意力加权
        enhanced_x1 = x1 * channel_att * spatial_att
        enhanced_x2 = x2 * channel_att * spatial_att
        
        # 门控融合
        gated_fusion = gate_weight * enhanced_x1 + (1 - gate_weight) * enhanced_x2
        
        # 特征重标定
        output = self.feature_recalibration(gated_fusion)
        
        return output


# 🔥 新增CBAM模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# 🔥 MSCB模块
class MSCB(nn.Module):
    """多尺度卷积块"""
    def __init__(self, in_channels, out_channels, stride=1, kernel_sizes=[1, 3, 5], 
                 expansion_factor=1, dw_parallel=True, add=True, activation='relu6'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.dw_parallel = dw_parallel
        self.add = add
        
        # 中间通道数
        mid_channels = in_channels * expansion_factor
        
        # 1x1 扩展卷积
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            self._get_activation(activation)
        )
        
        # 多尺度深度卷积
        self.dw_convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            dw_conv = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, 
                         groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                self._get_activation(activation)
            )
            self.dw_convs.append(dw_conv)
        
        # 1x1 压缩卷积
        self.project_conv = nn.Sequential(
            nn.Conv2d(mid_channels * len(kernel_sizes), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 残差连接
        self.use_residual = self.add and stride == 1 and in_channels == out_channels
    
    def _get_activation(self, activation):
        """获取激活函数"""
        if activation == 'relu6':
            return nn.ReLU6(inplace=True)
        elif activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        # 扩展
        x = self.expand_conv(x)
        
        # 多尺度深度卷积
        if self.dw_parallel:
            # 并行处理
            dw_outputs = []
            for dw_conv in self.dw_convs:
                dw_outputs.append(dw_conv(x))
            x = torch.cat(dw_outputs, dim=1)
        else:
            # 串行处理
            for dw_conv in self.dw_convs:
                x = dw_conv(x)
        
        # 压缩
        x = self.project_conv(x)
        
        # 残差连接
        if self.use_residual:
            x = x + identity
            
        return x


# 🔥 修改后的KANLayer - 只用一个MSCB
class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., 
                 no_kan=False, use_msdc=True, msdc_config=None, initial_lw=0.05, cbam_ratio=16):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.use_msdc = use_msdc

        # KAN层参数
        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        # MSDC分支配置
        default_msdc_config = {
            'kernel_sizes': [1, 3, 5],
            'expansion_factor': 1,
            'dw_parallel': True,
            'add': True,
            'activation': 'relu6'
        }
        self.msdc_config = {**default_msdc_config, **(msdc_config or {})}

        # 🔥 分支1：完整的KAN路径
        if not no_kan:
            self.fc1 = KANLinear(
                in_features, hidden_features,
                grid_size=grid_size, spline_order=spline_order,
                scale_noise=scale_noise, scale_base=scale_base,
                scale_spline=scale_spline, base_activation=base_activation,
                grid_eps=grid_eps, grid_range=grid_range,
            )
            self.fc2 = KANLinear(
                hidden_features, out_features,
                grid_size=grid_size, spline_order=spline_order,
                scale_noise=scale_noise, scale_base=scale_base,
                scale_spline=scale_spline, base_activation=base_activation,
                grid_eps=grid_eps, grid_range=grid_range,
            )
            self.fc3 = KANLinear(
                hidden_features, out_features,
                grid_size=grid_size, spline_order=spline_order,
                scale_noise=scale_noise, scale_base=scale_base,
                scale_spline=scale_spline, base_activation=base_activation,
                grid_eps=grid_eps, grid_range=grid_range,
            )
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        # KAN分支的DWConv
        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(out_features)

        # 🔥 分支2：CBAM + 单个MSCB路径
        if self.use_msdc:
            # 输入投影：将输入特征投影到合适维度
            self.input_proj = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.GELU(),
                nn.Dropout(drop)
            )
            
            # 🔥 CBAM注意力模块
            self.cbam = CBAM(hidden_features, ratio=cbam_ratio)
            
            # 🔥 单个MSCB处理：直接输出到out_features维度
            self.mscb = MSCB(
                in_channels=hidden_features,
                out_channels=out_features,
                stride=1,
                kernel_sizes=self.msdc_config['kernel_sizes'],
                expansion_factor=self.msdc_config['expansion_factor'],
                dw_parallel=self.msdc_config['dw_parallel'],
                add=self.msdc_config['add'],
                activation=self.msdc_config['activation']
            )

            # 🔥 可学习权重参数 lw，初始值0.05
            self.lw = nn.Parameter(torch.tensor(initial_lw, dtype=torch.float32))

        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _token_to_conv(self, x, H, W):
        """将token格式转换为卷积格式"""
        B, N, C = x.shape
        return x.transpose(1, 2).view(B, C, H, W)

    def _conv_to_token(self, x):
        """将卷积格式转换为token格式"""
        return x.flatten(2).transpose(1, 2)

    def forward(self, x, H, W):
        B, N, C = x.shape

        # 🔥 分支1：完整的KAN处理路径（KAN→DW→KAN→DW→KAN→DW）
        x1 = self.fc1(x.reshape(B*N, C))           # KAN
        x1 = x1.reshape(B, N, C).contiguous()
        x1 = self.dwconv_1(x1, H, W)              # DW
        
        x2 = self.fc2(x1.reshape(B*N, C))         # KAN  
        x2 = x2.reshape(B, N, C).contiguous()
        x2 = self.dwconv_2(x2, H, W)              # DW
        
        x3 = self.fc3(x2.reshape(B*N, C))         # KAN
        x3 = x3.reshape(B, N, C).contiguous()
        kan_output = self.dwconv_3(x3, H, W)      # DW → kan_output

        # 如果不使用MSDC，直接返回KAN输出
        if not self.use_msdc:
            return kan_output

        # 🔥 分支2：CBAM + 单个MSCB处理路径
        # 从原始输入开始独立处理
        msdc_x = self.input_proj(x.reshape(B*N, C))  # 投影到hidden_features维度
        msdc_x = msdc_x.reshape(B, N, -1).contiguous()
        
        # 转换为卷积格式
        msdc_conv = self._token_to_conv(msdc_x, H, W)
        
        # 🔥 通过CBAM注意力模块
        msdc_conv = self.cbam(msdc_conv)
        
        # 🔥 通过单个MSCB处理
        msdc_conv = self.mscb(msdc_conv)           # 单个MSCB
        
        # 转回token格式
        msdc_output = self._conv_to_token(msdc_conv)

        # 🔥 最终融合：kan_output + msdc_output * lw
        final_output = kan_output + msdc_output * self.lw

        return final_output


# 🔥 修改后的KANBlock
class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 no_kan=False, use_msdc=True, msdc_config=None, initial_lw=0.05, cbam_ratio=16):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = KANLayer(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, 
            drop=drop, 
            no_kan=no_kan,
            use_msdc=use_msdc,
            msdc_config=msdc_config,
            initial_lw=initial_lw,
            cbam_ratio=cbam_ratio
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


# 新增：定义中间循环的KAN块组，通过depth控制循环次数
class TokKANLoop(nn.Module):
    def __init__(self, dim, depth, drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 use_msdc=True, msdc_config=None, initial_lw=0.05, cbam_ratio=16):
        super().__init__()
        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            self.blocks.append(
                KANBlock(
                    dim=dim,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    use_msdc=use_msdc,
                    msdc_config=msdc_config,
                    initial_lw=initial_lw,
                    cbam_ratio=cbam_ratio
                )
            )

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        return x


class UKAN(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3, embed_dims=[256, 320, 512], no_kan=False,
    drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 3, 3], reduction=16, use_msdc=True, msdc_config=None, initial_lw=0.05, cbam_ratio=16, **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]

        self.encoder1 = ConvLayer(3, kan_input_dim//8)  
        self.encoder2 = ConvLayer(kan_input_dim//8, kan_input_dim//4)  
        self.encoder3 = ConvLayer(kan_input_dim//4, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 使用可配置深度的TokKANLoop替代原来的单层KANBlock
        self.tok_kan_loop1 = TokKANLoop(
            dim=embed_dims[1], 
            depth=depths[0],
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            use_msdc=use_msdc,
            msdc_config=msdc_config,
            initial_lw=initial_lw,
            cbam_ratio=cbam_ratio
        )

        self.tok_kan_loop2 = TokKANLoop(
            dim=embed_dims[2],
            depth=depths[1],
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            use_msdc=use_msdc,
            msdc_config=msdc_config,
            initial_lw=initial_lw,
            cbam_ratio=cbam_ratio
        )

        # 解码器部分的KANBlock
        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], 
            drop=drop_rate, 
            drop_path=drop_path_rate, 
            norm_layer=norm_layer,
            use_msdc=use_msdc,
            msdc_config=msdc_config,
            initial_lw=initial_lw,
            cbam_ratio=cbam_ratio
        )])

        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0], 
            drop=drop_rate, 
            drop_path=drop_path_rate, 
            norm_layer=norm_layer,
            use_msdc=use_msdc,
            msdc_config=msdc_config,
            initial_lw=initial_lw,
            cbam_ratio=cbam_ratio
        )])

        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])  
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])  
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0]//4) 
        self.decoder4 = D_ConvLayer(embed_dims[0]//4, embed_dims[0]//8)
        self.decoder5 = D_ConvLayer(embed_dims[0]//8, embed_dims[0]//8)

        # 初始化注意力门控融合模块
        self.attention_fusion1 = AttentionGatedFusion(embed_dims[1], reduction)  # 融合decoder1输出和t4
        self.attention_fusion2 = AttentionGatedFusion(embed_dims[0], reduction)  # 融合decoder2输出和t3
        self.attention_fusion3 = AttentionGatedFusion(kan_input_dim//4, reduction)  # 融合decoder3输出和t2
        self.attention_fusion4 = AttentionGatedFusion(kan_input_dim//8, reduction)  # 融合decoder4输出和t1

        self.final = nn.Conv2d(embed_dims[0]//8, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out  # 保存用于后续融合
        ### Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out  # 保存用于后续融合
        ### Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out  # 保存用于后续融合

        ### Tokenized KAN Stage
        ### Stage 4
        out, H, W = self.patch_embed3(out)
        out = self.tok_kan_loop1(out, H, W)  # 使用增强的KAN循环
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out  # 保存用于后续融合

        ### Bottleneck
        out, H, W = self.patch_embed4(out)
        out = self.tok_kan_loop2(out, H, W)  # 使用增强的KAN循环
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### 解码器部分
        ### Stage 4 - 使用注意力门控融合替代简单相加
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2,2), mode='bilinear', align_corners=True))
        out = self.attention_fusion1(out, t4)  # 注意力门控融合
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3 - 使用注意力门控融合替代简单相加
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2,2), mode='bilinear', align_corners=True))
        out = self.attention_fusion2(out, t3)  # 注意力门控融合
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        ### 后续解码器阶段 - 全部替换为注意力门控融合
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2,2), mode='bilinear', align_corners=True))
        out = self.attention_fusion3(out, t2)  # 注意力门控融合
        
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2,2), mode='bilinear', align_corners=True))
        out = self.attention_fusion4(out, t1)  # 注意力门控融合
        
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2,2), mode='bilinear', align_corners=True))

        return self.final(out)