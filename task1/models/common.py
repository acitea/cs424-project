import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, tv_tensors
from torchvision.transforms import v2 as transforms
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        proj_query = self.query_conv(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width*height)
        attention = F.softmax(torch.bmm(proj_query, proj_key), dim=-1)
        
        proj_value = self.value_conv(x).view(batch_size, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        return self.gamma*out + x
    

class ResidualBlock(nn.Module):
    def __init__(self, channels, stride=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, stride, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, stride, padding=1),
            nn.InstanceNorm2d(channels)
        )
        
    def forward(self, x):
        return x + self.block(x)
    


class EdgeDetection(nn.Module): # a.k.a. Sobel filter
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[[[-1., 0., 1.],
                                   [-2., 0., 2.],
                                   [-1., 0., 1.]]]])
        kernel_y = torch.tensor([[[[-1., -2., -1.],
                                   [0., 0., 0.],
                                   [1., 2., 1.]]]])
        
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, x):
        b, c, h, w = x.size()
        
        # Expand kernels for multi-channel input
        kernel_x = self.kernel_x.repeat(c, 1, 1, 1)
        kernel_y = self.kernel_y.repeat(c, 1, 1, 1)
        
        grad_x = F.conv2d(x, kernel_x, padding=1, groups=c)
        grad_y = F.conv2d(x, kernel_y, padding=1, groups=c)
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    

class SmootherNetwork(nn.Module):
    def __init__(self):
        super(SmootherNetwork, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=1)
        
    def forward(self, stylized, content):
        # Concatenate stylized result with original content for guidance
        x = torch.cat([stylized, content], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        residual = self.conv4(x)
        
        # Add residual for content preservation
        return torch.tanh(stylized + residual)


class StyleAwareNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.style_scale = nn.Linear(256, channels)
        self.style_bias = nn.Linear(256, channels)

    def forward(self, x, style_vector):
        normalized = self.norm(x)
        style_scale = self.style_scale(style_vector).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(style_vector).unsqueeze(2).unsqueeze(3)
        return normalized * style_scale + style_bias