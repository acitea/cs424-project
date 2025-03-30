from ..common import *


class AttentionResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            SelfAttention(channels),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )
        
    def forward(self, x):
        return x + self.block(x)

class EnhancedGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder with attention
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            SelfAttention(64),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            SelfAttention(128),
            nn.InstanceNorm2d(128),
            nn.ReLU()
        )
        
        # Residual blocks with attention
        self.res_blocks = nn.Sequential(
            *[AttentionResBlock(256) for _ in range(6)]
        )
        
    # Remainder of previous architecture...

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            SelfAttention(64)
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            SelfAttention(128)
        )
        # Additional scales...


class FaceStyleDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(FaceStyleDiscriminator, self).__init__()
        
        # Global discriminator (multi-scale)
        self.global_discriminator = MultiScaleDiscriminator(input_channels)
        
        # Region-specific discriminators
        self.eyes_discriminator = PatchDiscriminator(input_channels)
        self.nose_discriminator = PatchDiscriminator(input_channels)
        self.mouth_discriminator = PatchDiscriminator(input_channels)
        
    def forward(self, x, face_parsing=None):
        # Global discrimination
        global_results = self.global_discriminator(x)
        
        # Region-specific discrimination if face parsing is available
        regional_results = {}
        if face_parsing is not None:
            # Apply masks and get regional results
            regional_results['eyes'] = self.eyes_discriminator(x * face_parsing['eyes'])
            regional_results['nose'] = self.nose_discriminator(x * face_parsing['nose'])
            regional_results['mouth'] = self.mouth_discriminator(x * face_parsing['mouth'])
        
        return global_results, regional_results


class FaceStyleTransferLossFactory:
    @staticmethod
    def create_loss_suite(options=None):
        """Create a suite of loss functions for face style transfer"""
        losses = {
            # Standard CycleGAN losses
            'cycle_consistency': CycleConsistencyLoss(lambda_A=10.0, lambda_B=10.0),
            'adversarial': AdversarialLoss(use_lsgan=True),
            
            # Identity preservation
            'identity_mapping': IdentityMappingLoss(lambda_id=5.0),
            'face_identity': FaceIdentityLoss(lambda_face=10.0),
            
            # Region-specific losses
            'facial_component': FacialComponentLoss(lambda_comp=2.0),
            'line_awareness': LineAwarenessLoss(lambda_line=1.0),
            
            # Perceptual losses
            'perceptual': PerceptualLoss(lambda_perc=1.0),
            
            # Content preservation
            'content': ContentPreservationLoss(lambda_content=1.0)
        }
        
        # Allow customization based on options
        if options is not None:
            for name, loss in losses.items():
                if name in options:
                    loss.lambda_weight = options[name]
        
        return losses
