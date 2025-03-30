from ..common import *

class RegionalBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(True)
        )
        
    def forward(self, features, mask):
        # Apply region mask
        masked_features = features * mask
        # Process features
        x = self.conv1(masked_features)
        x = self.conv2(x)
        return x


class FaceStyleGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(FaceStyleGenerator, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )
        
        # Self-attention layer
        self.attention = SelfAttention(256)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(6)])
        
        # Region-specific branches
        self.eyes_branch = RegionalBranch(256)
        self.nose_branch = RegionalBranch(256)
        self.mouth_branch = RegionalBranch(256)
        self.skin_branch = RegionalBranch(256)
        
        # Feature fusion
        self.fusion = nn.Conv2d(256*5, 256, 1)  # Original + 4 regions
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, output_channels, 7, padding=3),
            nn.Tanh()
        )
        
        # Photorealistic smoothing module (optional)
        self.smoother = SmootherNetwork(output_channels)
        
    def forward(self, x, face_parsing=None, apply_smoothing=True):
        # Initial encoding
        features = self.encoder(x)
        
        # Apply attention
        attended = self.attention(features)
        
        # Process with residual blocks
        res_features = self.res_blocks(attended)
        
        # Region-specific processing if face parsing is available
        if face_parsing is not None:
            # Process facial regions
            eyes_feat = self.eyes_branch(res_features, face_parsing['eyes'])
            nose_feat = self.nose_branch(res_features, face_parsing['nose'])
            mouth_feat = self.mouth_branch(res_features, face_parsing['mouth'])
            skin_feat = self.skin_branch(res_features, face_parsing['skin'])
            
            # Combine original features with region-specific features
            combined = torch.cat([res_features, eyes_feat, nose_feat, mouth_feat, skin_feat], dim=1)
            fused = self.fusion(combined)
        else:
            fused = res_features
        
        # Generate output
        output = self.decoder(fused)
        
        # Apply photorealistic smoothing if requested
        if apply_smoothing:
            output = self.smoother(output, x)
        
        return output
