from ..common import *

class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs, padding_mode="reflect") if down else
            nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )
    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            GBlock(channels, channels, use_act=True, kernel_size=3, stride=1, padding=1),
            GBlock(channels, channels, use_act=False, kernel_size=3, stride=1, padding=1)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, network_features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, network_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"), # (3, 256, 256) -> (64, 256, 256)
            nn.ReLU(inplace=True),
        )

        self.down_block = nn.Sequential(
            GBlock(network_features*1, network_features*2, kernel_size=3, stride=2, padding=1), # (128, 128, 128)
            GBlock(network_features*2, network_features*4, kernel_size=3, stride=2, padding=1), # (256, 64, 64)
        )

        self.residual_block = nn.Sequential(
            *[ResidualBlock(network_features*4) for _ in range(num_residuals)]
        )

        self.up_block = nn.Sequential(
            GBlock(network_features*4, network_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (128, 128, 128)
            GBlock(network_features*2, network_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (64, 256, 256)
        )

        self.last = nn.Sequential(
            nn.Conv2d(network_features, 3, kernel_size=7, stride=1, padding=3, padding_mode="reflect"), # (3, 256, 256)
            nn.Tanh()
        )
    def forward(self, x):
        x = self.initial(x)
        x = self.down_block(x)
        x = self.residual_block(x)
        x = self.up_block(x)
        x = self.last(x)
        return x