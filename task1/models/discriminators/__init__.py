from .patch import PatchDiscriminator, NLayerDiscriminator, GithubDiscriminator
from .patchSpectral import PatchGANDiscriminator as PatchSpectralDiscriminator
from .random_kaggle import RandomKaggleDiscriminator
from .default import Discriminator as DefaultDiscriminator

__all__ = ["PatchDiscriminator", "PatchSpectralDiscriminator", "NLayerDiscriminator", "GithubDiscriminator", "RandomKaggleDiscriminator", "DefaultDiscriminator"]
