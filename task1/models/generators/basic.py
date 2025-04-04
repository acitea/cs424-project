from ..common import *

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            ############## Encoder ##############
            # Initial convolution block
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, 
                      padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),  
            # Downsampling
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, 
                      stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, 
                      stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            ############## Transformer ##############
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),

            ############## Decoder ##############
            nn.ConvTranspose2d(in_channels=256, out_channels=128, 
                               kernel_size=3, stride=2, padding=0, 
                               output_padding=1),
            nn.InstanceNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, 
                               kernel_size=3, stride=2, padding=0, 
                               output_padding=1),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            # Output layer
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)