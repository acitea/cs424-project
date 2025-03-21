from ..common import *

class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                        kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # Final layers for BCELoss compatibility
        sequence += [
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=1),
            nn.Sigmoid()  # Add sigmoid for probability output
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        x = self.model(input)
        return x.view(x.size(0), -1)  # Flatten to (batch_size, 1)
