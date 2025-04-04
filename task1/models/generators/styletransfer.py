from ..common import *

# https://arxiv.org/pdf/2207.01909v1
# Style aware normalisation module To
# be more specific, content is the information that we would like to preserve during the transformation (e.g. shape, semantic information), while style is what
# we need to change to make the source image ’similar’ to the target image (e.g.
# color, illumination, clarity). Content-fixed transfer means the content information before and after transformation should be retained.


class StyleTransferGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super().__init__()
        # Initial convolution
        self.init = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(6)]
        )
        
        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        
        # Output layer
        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, 7),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.init(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.out(x)


class PhotorealisticStyleTransferGenerator(nn.Module):
    def __init__(self):
        super(PhotorealisticStyleTransferGenerator, self).__init__()
        self.style_net = StyleTransferGenerator()  # Base style transfer network
        self.smoother = SmootherNetwork()
    
    def forward(self, content, style):
        # First stage: Style transfer
        stylized = self.style_net(content, style)
        
        # Second stage: Photorealistic smoothing
        smoothed = self.smoother(stylized, content)
        return smoothed
