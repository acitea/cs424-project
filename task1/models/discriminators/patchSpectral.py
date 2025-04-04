from ..common import *

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super(PatchGANDiscriminator, self).__init__()
        
        # No normalization for first layer
        sequence = [
            nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                                               kernel_size=4, stride=2, padding=1)),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            
        # Add one more layer with stride 1
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                                           kernel_size=4, stride=1, padding=1)),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        # # Output 1-channel prediction map
        # sequence += [nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult, 1, 
        #                                             kernel_size=4, stride=1, padding=1))]
        
        # Added compatibility for BCE
        sequence += [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf * nf_mult, 1))
        ]
        
        self.model = nn.Sequential(*sequence)
        
    def forward(self, input):
        return self.model(input)
