from ..common import *

# -------------------------
# Model Components
# -------------------------
class _ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.1, use_spectral_norm=True):
        super(_ResidualBlock, self).__init__()
        
        # Apply spectral normalization if enabled
        norm_layer = lambda x: nn.utils.spectral_norm(x) if use_spectral_norm else x
        
        self.conv1 = norm_layer(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False))
        self.in1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.conv2 = norm_layer(nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False))
        self.in2 = nn.InstanceNorm2d(out_channels)
        
        # If the dimensions differ, use a 1x1 conv to match them.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                norm_layer(nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False)),
                nn.InstanceNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.in1(self.conv1(x)))
        out = self.dropout(out)
        out = self.in2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class UpBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, upsample=False):
        super(UpBasicBlock, self).__init__()
        self.upsample_flag = upsample
        
        if upsample:
            self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=3, stride=2,
                                           padding=1, output_padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1,
                                  padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(planes)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(planes)
        
        # If upsampling or channel change is needed, adjust the residual connection
        if upsample or in_planes != planes:
            if upsample:
                self.shortcut = nn.ConvTranspose2d(in_planes, planes, kernel_size=1, stride=2,
                                                 output_padding=1, bias=False)
            else:
                self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False)
        else:
            self.shortcut = None
            
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        
        if self.shortcut is not None:
            identity = self.shortcut(x)
            
        out += identity
        out = self.relu(out)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Input: 128×128×3 --> conv: kernel=7, stride=2, padding=3 => 64×64×64
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Encoder path
        self.res1_1 = _ResidualBlock(64, 128, stride=1, dropout_rate=0.0)
        self.res1_2 = _ResidualBlock(128, 128, stride=1, dropout_rate=0.0)
        
        # 32x32x128 --> 16×16×256 using a residual block with stride=2.
        self.res2_1 = _ResidualBlock(128, 256, stride=2, dropout_rate=0.0)
        self.res2_2 = _ResidualBlock(256, 256, stride=1, dropout_rate=0.0)
        
        # Additional encoder block
        self.res3_1 = _ResidualBlock(256, 512, stride=2, dropout_rate=0.0)
        self.res3_2 = _ResidualBlock(512, 512, stride=1, dropout_rate=0.0)
        
        # Decoder path with skip connections
        # Upsampling: 32×32×256 --> 64×64×128.
        self.up_block1_1 = UpBasicBlock(512 + 512, 256, upsample=True)
        self.up_block1_2 = UpBasicBlock(256, 256, upsample=False)
        
        # Upsampling: 64×64×128 --> 128×128×64.
        self.up_block2_1 = UpBasicBlock(256 + 256, 128, upsample=True)
        self.up_block2_2 = UpBasicBlock(128, 128, upsample=False)
        
        # Final upsampling block
        self.up_block3_1 = UpBasicBlock(128 + 128, 64, upsample=True)
        self.up_block3_2 = UpBasicBlock(64, 64, upsample=False)
        
        # Final layer: output 128×128×3.
        self.pad = nn.ReflectionPad2d(1)
        self.cov_final = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=0)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Encoder path with feature map storage
        x_initial = self.initial_conv(x)
        
        x_res1_1 = self.res1_1(x_initial)
        x_res1_2 = self.res1_2(x_res1_1)
        
        x_res2_1 = self.res2_1(x_res1_2)
        x_res2_2 = self.res2_2(x_res2_1)
        
        x_res3_1 = self.res3_1(x_res2_2)
        x_res3_2 = self.res3_2(x_res3_1)
        
        # Decoder path with skip connections
        x_up1_1 = self.up_block1_1(torch.cat([x_res3_2, x_res3_1], dim=1))
        x_up1_2 = self.up_block1_2(x_up1_1)
        
        x_up2_1 = self.up_block2_1(torch.cat([x_up1_2, x_res2_2], dim=1))
        x_up2_2 = self.up_block2_2(x_up2_1)
        
        x_up3_1 = self.up_block3_1(torch.cat([x_up2_2, x_res1_2], dim=1))
        x_up3_2 = self.up_block3_2(x_up3_1)
        
        x_up3_2 = self.pad(x_up3_2)
        x = self.cov_final(x_up3_2)
        x = self.tanh(x)
        return x