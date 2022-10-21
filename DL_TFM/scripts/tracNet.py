import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock_1(nn.Module):
    """Conv3D -> BatchNorm -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock_1, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(2, 3, 3), padding='same', bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_block_1(x)


class ConvBlock_2(nn.Module):
    """Conv3D -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock_2, self).__init__()
        self.conv_block_2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(2, 3, 3), padding='same'),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_block_2(x)


class TracNet(nn.Module):
    """
    Define the actual neural network as class `TracNet`.
    It makes use of the above defined `ConvBlock_1` and `ConvBlock_2` as well as
        - Transposed 3D-Convolution
        - 3D-MaxPooling
        - Concatenating layers
    """
    def __init__(self, n_channels):
        super(TracNet, self).__init__()

        self.s1 = ConvBlock_1(n_channels, 32)
        self.s2 = ConvBlock_2(32, 64)
        self.s3 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.s4 = ConvBlock_1(64, 64)
        self.s5 = ConvBlock_2(64, 128)
        self.s6 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.s7 = ConvBlock_1(128, 128)
        self.s8 = ConvBlock_2(128, 256)
        self.s9 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.s10 = ConvBlock_1(256, 128)
        self.s11 = ConvBlock_1(128, 256)
        self.s12 = nn.ConvTranspose3d(256, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2))
        # fusion3
        self.s13 = ConvBlock_1(512, 64)
        self.s14 = ConvBlock_1(64, 128)
        self.s15 = nn.ConvTranspose3d(128, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2))
        # fusion2
        self.s16 = ConvBlock_1(256, 32)
        self.s17 = ConvBlock_1(32, 64)
        self.s18 = nn.ConvTranspose3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2))
        # fusion1
        self.s19 = ConvBlock_1(128, 1)
        self.s20 = ConvBlock_1(1, 32)
        self.s21 = nn.Conv3d(32, 1, kernel_size=(2, 3, 3), padding='same')

    def forward(self, x):
        x1 = self.s1(x)
        x2 = self.s2(x1)
        x3 = self.s3(x2)
        x4 = self.s4(x3)
        x5 = self.s5(x4)
        x6 = self.s6(x5)
        x7 = self.s7(x6)
        x8 = self.s8(x7)
        x9 = self.s9(x8)
        x10 = self.s10(x9)
        x11 = self.s11(x10)
        x12 = self.s12(x11)
        padded = torch.nn.functional.pad(x12, (0, -1, 0, -1), 'constant', 0)
        fusion3 = torch.cat((x8, padded), dim=1)
        x13 = self.s13(fusion3)
        x14 = self.s14(x13)
        x15 = self.s15(x14)
        padded = torch.nn.functional.pad(x15, (0, -1, 0, -1), 'constant', 0)
        fusion2 = torch.cat((x5, padded), dim=1)
        x16 = self.s16(fusion2)
        x17 = self.s17(x16)
        x18 = self.s18(x17)
        padded = torch.nn.functional.pad(x18, (0, -1, 0, -1), 'constant', 0)
        fusion1 = torch.cat((x2, padded), dim=1)
        x19 = self.s19(fusion1)
        x20 = self.s20(x19)
        logits = self.s21(x20)

        return logits
