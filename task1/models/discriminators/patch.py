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
    


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=4, norm_layer=nn.BatchNorm2d, use_sigmoid=True):
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=True),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=True),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

# Another implementation of the PatchGAN discriminator
# https://github.com/CycleGANS/CycleGANsImplementation/blob/master/simple_discriminator.py
class CLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, padding='valid', 
                 do_norm=True, do_relu=True, relu_alpha=0):
        super(CLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                              padding=1 if padding=='SAME' else 0, bias=False)
        self.do_norm = do_norm
        self.do_relu = do_relu
        self.relu_alpha = relu_alpha
        
        if do_norm:
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        
    def forward(self, x):
        x = self.conv(x)
        if self.do_norm:
            x = self.norm(x)
        if self.do_relu:
            if self.relu_alpha == 0:
                x = F.relu(x, inplace=True)
            else:
                x = F.leaky_relu(x, self.relu_alpha, inplace=True)
        return x

class GithubDiscriminator(nn.Module):
    def __init__(self, input_channels=3, num_filters=64):
        super(GithubDiscriminator, self).__init__()
        
        self.layer1 = CLayer(input_channels, num_filters, 4, 2, 'SAME', False, True, 0.2)
        self.layer2 = CLayer(num_filters, num_filters * 2, 4, 2, 'SAME', True, True, 0.2)
        self.layer3 = CLayer(num_filters * 2, num_filters * 4, 4, 2, 'SAME', True, True, 0.2)
        self.layer4 = CLayer(num_filters * 4, num_filters * 8, 4, 1, 'SAME', True, True, 0.2)
        self.layer5 = CLayer(num_filters * 8, 1, 4, 1, 'SAME', False, False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return self.sigmoid(x)
