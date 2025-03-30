from ..common import *

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first_block=False, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs, padding_mode='reflect', bias=True),
            nn.InstanceNorm2d(out_channels) if not first_block else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module):
    def __init__(self, network_features=64):
        super().__init__()
        self.model = nn.Sequential(
            DBlock(3, network_features, first_block=True, kernel_size=4, stride=2, padding=1), # (3, 256, 256) -> (64, 128, 128)
            DBlock(network_features*1, network_features*2, kernel_size=4, stride=2, padding=1), # (128, 64, 64)
            DBlock(network_features*2, network_features*4, kernel_size=4, stride=2, padding=1), # (256, 32, 32)
            DBlock(network_features*4, network_features*8, kernel_size=4, stride=1, padding=1), # (512, 31, 31)
        )
        self.last_layer = nn.Sequential(
            nn.Conv2d(network_features*8, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"), # (1, 30, 30)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 3, 256, 256)
        x = self.model(x)
        x = self.last_layer(x)
        return x
