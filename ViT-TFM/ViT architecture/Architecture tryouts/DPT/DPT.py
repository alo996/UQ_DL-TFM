import torch
import torch.nn as nn
import numpy as np


# Basic ViT classes
class PatchEmbed(nn.Module):
    def __init__(self, device, img_size=104, patch_size=8, in_chans=2, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, device=device)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, device, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, device, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(self, device, embed_dim=128, num_heads=8, mlp_ratio=4., qkv_bias=True, drop=0.05, attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, device=device)

        self.norm2 = norm_layer(embed_dim)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), act_layer=act_layer, drop=drop, device=device)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        return x


class ResidualConvUnit(nn.Module):
    def __init__(self, features, device):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, device=device)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, device=device)
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.gelu(x)
        out = self.conv1(out)
        out = self.gelu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, device):
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features, device=device)
        self.resConfUnit2 = ResidualConvUnit(features, device=device)

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)  # [batchsize, channels, height, width]
        return output


class Encoder(nn.Module):
    def __init__(self, device, embed_dim=128, norm_layer=nn.LayerNorm):
        super(Encoder, self).__init__()

        self.patch_embed = PatchEmbed(device=device)
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.)
        self.encoder_blocks = nn.ModuleList(
            [
                Block(
                    device, embed_dim=128, num_heads=8, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm
                )
                for i in range(6)
            ]
        )
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        layer1 = None
        layer2 = None

        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            if i == 1:
                layer1 = x
            elif i == 3:
                layer2 = x
        layer3 = self.norm(x)

        return layer1, layer2, layer3


class RecTracHead(nn.Module):
    def __init__(self, in_dim, device):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim // 2, kernel_size=1, stride=1, padding=0, bias=False, device=device)
        self.conv2 = nn.Conv2d(in_dim // 2, in_dim // 2, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.conv3 = nn.Conv2d(in_dim // 2, 2, kernel_size=1, stride=1, padding=0, bias=False, device=device)
        self.norm1 = nn.BatchNorm2d(in_dim // 2)
        self.batchnorm2 = nn.BatchNorm2d(in_dim // 2)
        self.gelu = nn.GELU()

    def forward(self, x):
        # print(f"forward call RecTracHead 1: x.shape is {x.shape}")
        x = self.conv1(x)
        x = self.batchnorm1(x)
        # print(f"forward call RecTracHead 2: x.shape is {x.shape}")
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        # print(f"forward call RecTracHead 3: x.shape is {x.shape}")
        x = self.gelu(x)
        x_rec = self.conv3(x)
        # print(f"forward call RecTracHead 4: x_rec.shape is {x_rec.shape}")

        return x_rec


class RecTracHead2(nn.Module):
    def __init__(self, in_dim, device, patch_size=8):
        super().__init__()
        layers = [nn.Linear(in_dim, in_dim, device=device), nn.GELU()]
        self.mlp = nn.Sequential(*layers)
        self.conv = nn.Conv2d(
            in_dim,
            2,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            device=device
        )

    def forward(self, x):
        print(x.shape)
        x = x.transpose(1, 3)
        print(x.shape)
        x = self.mlp(x)
        print(x.shape)
        x = x.transpose(1, 3)
        print(x.shape)
        x = x.unflatten(2, (13, 13))
        print(x.shape)
        x_rec = self.conv()
        print(x_rec.shape)

        return x_rec


class DPT(nn.Module):
    def __init__(self, device, img_size=104, features=169, embed_dim=128, in_shape=[128, 2*128, 4*128]):
        super(DPT, self).__init__()
        self.encoder = Encoder(embed_dim=embed_dim, device=device)

        self.act_postprocess1 = nn.Sequential(self.readout_oper[0], Transpose(1, 2), nn.Unflatten(2, torch.Size([img_size // 8, img_size // 8])),
                                              nn.Conv2d(in_channels=embed_dim, out_channels=in_shape[0], kernel_size=1, stride=1, padding=0, device=device),
                                              nn.ConvTranspose2d(in_channels=in_shape[0], out_channels=in_shape[0], kernel_size=4, stride=4, padding=0, bias=True, dilation=1, groups=1, device=device))
        self.act_postprocess2 = nn.Sequential(self.readout_oper[1], Transpose(1, 2), nn.Unflatten(2, torch.Size([img_size // 8, img_size // 8])),
                                              nn.Conv2d(in_channels=embed_dim, out_channels=in_shape[1], kernel_size=1,stride=1,padding=0, device=device),
                                              nn.ConvTranspose2d(in_channels=in_shape[1], out_channels=in_shape[1], kernel_size=2, stride=2, padding=0, bias=True, dilation=1, groups=1, device=device))
        self.act_postprocess3 = nn.Sequential(self.readout_oper[3], Transpose(1, 2), nn.Unflatten(2, torch.Size([img_size // 8, img_size // 8])),
                                              nn.Conv2d(in_channels=embed_dim, out_channels=in_shape[2], kernel_size=1, stride=1, padding=0, device=device))

        self.layer1_rn = nn.Conv2d(in_shape[0], features, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.layer2_rn = nn.Conv2d(in_shape[1], features, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.layer3_rn = nn.Conv2d(in_shape[2], features, kernel_size=3, stride=1, padding=1, bias=False, device=device)

        self.refinenet1 = FeatureFusionBlock(features, device=device)
        self.refinenet2 = FeatureFusionBlock(features, device=device)
        self.refinenet3 = FeatureFusionBlock(features, device=device)

        self.head = RecTracHead2(in_dim=169, device=device)

    def forward(self, x):

        layer_1, layer_2, layer_3 = self.encoder(x)  # [1, 169, 128]

        layer_1 = self.act_postprocess1(layer_1)  # [1, 128, 52, 52]
        layer_2 = self.act_postprocess2(layer_2)  # [1, 256, 26, 26]
        layer_3 = self.act_postprocess3(layer_3)  # [1, 512, 13, 13]

        layer_1_rn = self.layer1_rn(layer_1)  # [1, 169, 52, 52]
        layer_2_rn = self.layer2_rn(layer_2)  # [1, 169, 26, 26]
        layer_3_rn = self.layer3_rn(layer_3)  # [1, 169, 13, 13]

        path_3 = self.refinenet3(layer_3_rn)  # [1, 169, 26, 26]
        path_2 = self.refinenet2(path_3, layer_2_rn)  # [1, 169, 52, 52]
        path_1 = self.refinenet1(path_2, layer_1_rn)  # [1, 169, 104, 104]

        out = self.head(path_1)  # [1, 2, 104, 104]

        return out
