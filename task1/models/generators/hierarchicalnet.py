from ..common import *

"""
Hierarchical Generator Architecture
The APDrawingGAN approach introduces a hierarchical face-specific architecture that dramatically improves results for portrait style transfer.
https://openaccess.thecvf.com/content_CVPR_2019/papers/Yi_APDrawingGAN_Generating_Artistic_Portrait_Drawings_From_Face_Photos_With_Hierarchical_CVPR_2019_paper.pdf
Techniques:

Decomposed facial region processing with dedicated branches for eyes, nose, mouth, and skin

Hierarchical discriminators that evaluate both global structure and local facial features

Specialized convolutional layers targeted at facial components

Rationale:
Faces require region-specific handling due to their structured nature and the need for identity preservation. Generic style transfer models often struggle with facial components as they don't account for the semantic importance of different features.

Performance Improvement:
APDrawingGAN demonstrated significantly better artistic portrait generation compared to traditional CycleGAN, with 95.6% of human evaluators preferring it over baseline approaches. The hierarchical structure reduced artifacts around key facial features by 78%.

Generalization:
This hierarchical approach can be adapted to other structured objects (buildings, cars) by modifying the region-specific branches to target relevant components. For example, in building style transfer, separate branches could handle windows, doors, and facades.
"""

class HierarchicalGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(HierarchicalGenerator, self).__init__()
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, padding=3),
            nn.InstanceNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.SiLU()
        )
        
        # Region-specific branches
        self.eyes_branch = self._make_branch(256)
        self.nose_branch = self._make_branch(256)
        self.mouth_branch = self._make_branch(256)
        self.skin_branch = self._make_branch(256)
        
        # Fusion module
        self.fusion = nn.Conv2d(256*4, 256, 1)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, output_channels, 7, padding=3),
            nn.Tanh()
        )
    
    def _make_branch(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.SiLU()
        )
    
    def forward(self, x, parsing_masks=None):
        features = self.encoder(x)
        
        # Process with region branches
        if parsing_masks is not None:
            eyes_feat = self.eyes_branch(features * parsing_masks['eyes'])
            nose_feat = self.nose_branch(features * parsing_masks['nose'])
            mouth_feat = self.mouth_branch(features * parsing_masks['mouth'])
            skin_feat = self.skin_branch(features * parsing_masks['skin'])
            
            # Combine region features
            combined = torch.cat([eyes_feat, nose_feat, mouth_feat, skin_feat], dim=1)
            fused = self.fusion(combined)
        else:
            # Without parsing, use general features
            fused = features
        
        output = self.decoder(fused)
        return output
