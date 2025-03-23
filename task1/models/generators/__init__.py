# from .resnet import ResNetGenerator
# from .unet import UNetGenerator
# from .custom_generator import CustomGenerator
from .perplexity_resnet_unet import UNetGenerator
from .faststylegithub import FastStyleGenerator

__all__ = ["UNetGenerator", "FastStyleGenerator"]
