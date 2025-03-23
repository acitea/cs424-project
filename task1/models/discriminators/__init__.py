from .patch import PatchDiscriminator
from .patchSpectral import PatchGANDiscriminator as PatchSpectralDiscriminator
from .random_kaggle import RandomKaggleDiscriminator
# from .resnet import CustomDiscriminator

__all__ = ["PatchDiscriminator", "PatchSpectralDiscriminator", "RandomKaggleDiscriminator"]
