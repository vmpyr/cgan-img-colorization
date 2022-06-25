import torch
import torch.nn as nn

class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels , out_channels, kernel_size=kernel_size, stride=stride, bias=False, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                DiscBlock(in_channels, feature, kernel_size=4, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, bias=False, padding=1, padding_mode="reflect"),
                nn.InstanceNorm2d(1, affine=True),
                nn.Sigmoid()
            )
        )

        self.disc_model = nn.Sequential(*layers)

    def forward(self, x, y):
        x=torch.cat([x,y], dim=1)
        x=self.initial(x)
        return self.disc_model(x)
