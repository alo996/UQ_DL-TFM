import torch
import torch.nn as nn


# Basic ViT classes
class PatchEmbed(nn.Module):
    def __init__(self, img_size=104, patch_size=8, in_chans=2, embed_dim=8*169):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        print(f"dim is {dim}")
        print(f"num_heads is {num_heads}")
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
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
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(self, embed_dim=8*169, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(embed_dim)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# DPT specific stuff
def get_readout_oper(features, use_readout, start_index=0):
    if use_readout is False:
        readout_oper = [Slice(start_index)] * features
    else:
        assert (
            False
        ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper


class Slice(nn.Module):
    def __init__(self, start_index=0):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index:]


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


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
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        print(f"forward pass ResidualConvUnit: shape of x is {x.shape}")
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        print(f"forward call of FeatureFusionBlock: xs[0].shape is {xs[0].shape}")
        output = xs[0]
        if len(xs) == 2:
            print(f"forward call of FeatureFusionBlock: xs[1].shape is {xs[1].shape}")
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        return output


class Encoder(nn.Module):
    def __init__(self, embed_dim=8*169, num_heads=8, norm_layer=nn.LayerNorm):
        super(Encoder, self).__init__()

        self.patch_embed = PatchEmbed()
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=0.1)
        self.encoder_block = Block(embed_dim=embed_dim, num_heads=num_heads)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i in range(6):
            x = self.encoder_block(x)

        layer1 = x

        for i in range(6):
            x = self.encoder_block(x)

        layer2 = x

        for i in range(6):
            x = self.encoder_block(x)

        layer3 = x

        for i in range(6):
            x = self.encoder_block(x)

        layer4 = self.norm(x)
        return layer1, layer2, layer3, layer4


class DPT(nn.Module):
    def __init__(self, img_size=104, features=169, embed_dim=8*169, in_shape=[169, 2*169, 4*169, 8*169]):
        super(DPT, self).__init__()

        self.encoder = Encoder(embed_dim=embed_dim)

        self.refinenet1 = FeatureFusionBlock(features)
        self.refinenet2 = FeatureFusionBlock(features)
        self.refinenet3 = FeatureFusionBlock(features)
        self.refinenet4 = FeatureFusionBlock(features)

        self.layer1_rn = nn.Conv2d(in_shape[0], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(in_shape[1], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(in_shape[2], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(in_shape[3], features, kernel_size=3, stride=1, padding=1, bias=False)

        self.readout_oper = get_readout_oper(features, use_readout=False, start_index=0)

        self.act_postprocess1 = nn.Sequential(self.readout_oper[0], Transpose(1, 2), nn.Unflatten(2, torch.Size([img_size // 8, img_size // 8])),
                                              nn.Conv2d(in_channels=embed_dim, out_channels=in_shape[0], kernel_size=1, stride=1, padding=0),
                                              nn.ConvTranspose2d(in_channels=in_shape[0], out_channels=in_shape[0], kernel_size=4, stride=4, padding=0, bias=True, dilation=1, groups=1))
        self.act_postprocess2 = nn.Sequential(self.readout_oper[1], Transpose(1, 2), nn.Unflatten(2, torch.Size([img_size // 8, img_size // 8])),
                                              nn.Conv2d(in_channels=embed_dim, out_channels=in_shape[1],kernel_size=1,stride=1,padding=0),
                                              nn.ConvTranspose2d(in_channels=in_shape[1], out_channels=in_shape[1], kernel_size=2, stride=2, padding=0, bias=True, dilation=1, groups=1))
        self.act_postprocess3 = nn.Sequential(self.readout_oper[3], Transpose(1, 2), nn.Unflatten(2, torch.Size([img_size // 8, img_size // 8])),
                                              nn.Conv2d(in_channels=embed_dim, out_channels=in_shape[2], kernel_size=1, stride=1, padding=0))
        self.act_postprocess4 = nn.Sequential(self.readout_oper[4], Transpose(1, 2), nn.Unflatten(2, torch.Size([img_size // 8, img_size // 8])),
                                              nn.Conv2d(in_channels=embed_dim, out_channels=in_shape[3], kernel_size=1, stride=1, padding=0),
                                              nn.Conv2d(in_channels=in_shape[3], out_channels=in_shape[3], kernel_size=3, stride=2, padding=1))

        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        )

    def forward(self, x):

        layer_1, layer_2, layer_3, layer_4 = self.encoder(x)
        print(f"forward call of DPT: layer_1.shape is {layer_1.shape}")
        print(f"forward call of DPT: layer_2.shape is {layer_2.shape}")
        print(f"forward call of DPT: layer_3.shape is {layer_3.shape}")
        print(f"forward call of DPT: layer_4.shape is {layer_4.shape}")

        layer_1 = self.act_postprocess1(layer_1)
        layer_2 = self.act_postprocess1(layer_2)
        layer_3 = self.act_postprocess1(layer_3)
        layer_4 = self.act_postprocess1(layer_4)


        layer_1_rn = self.layer1_rn(layer_1)
        print(f"forward call of DPT: layer_2.shape is {layer_2.shape}")
        layer_2_rn = self.layer2_rn(layer_2)
        print(f"forward call of DPT: layer_2_rn.shape is {layer_2_rn.shape}")
        layer_3_rn = self.layer3_rn(layer_3)
        layer_4_rn = self.layer4_rn(layer_4)

        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)

        out = self.head(path_1)

        return out


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = DPT().to(device)