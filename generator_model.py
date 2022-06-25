import torch
import torch.nn as nn

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, is_encoder=True, use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, padding_mode="reflect")
            if is_encoder
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2) if is_encoder else nn.ReLU(),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(1, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = GenBlock(features, features * 2)
        self.down2 = GenBlock(features * 2, features * 4)
        self.down3 = GenBlock(features * 4, features * 8)
        self.down4 = GenBlock(features * 8, features * 8)
        self.down5 = GenBlock(features * 8, features * 8)
        self.down6 = GenBlock(features * 8, features * 8)
        self.bottom = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        self.up1 = GenBlock(features * 8, features * 8, is_encoder=False, use_dropout=True)
        self.up2 = GenBlock(features * 8 * 2, features * 8, is_encoder=False, use_dropout=True)
        self.up3 = GenBlock(features * 8 * 2, features * 8, is_encoder=False, use_dropout=True)
        self.up4 = GenBlock(features * 8 * 2, features * 8, is_encoder=False)
        self.up5 = GenBlock(features * 8 * 2, features * 4, is_encoder=False)
        self.up6 = GenBlock(features * 4 * 2, features * 2, is_encoder=False)
        self.up7 = GenBlock(features * 2 * 2, features, is_encoder=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, 2, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottom = self.bottom(d7)
        up1 = self.up1(bottom)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))
