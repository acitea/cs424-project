# from .resnet import ResNetGenerator
# from .unet import UNetGenerator
# from .custom_generator import CustomGenerator
from .perplexity_resnet_unet import UNetGenerator
from .faststylegithub import FastStyleGenerator
from .default import Generator as DefaultGenerator
from .zhirong import Generator as ZhirongGenerator
from .basic import Generator as BasicGenerator

__all__ = ["UNetGenerator", "FastStyleGenerator", "BasicGenerator", "DefaultGenerator", "ZhirongGenerator"]
