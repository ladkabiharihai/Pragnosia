"""
Pragnosia: A Brain-Inspired, Energy-Efficient Multimodal LLM

Core principle: Total parameters are large, but active parameters per forward pass are small.
"""

__version__ = "0.1.0"

from pragnosia.config import PragnosiaConfig
from pragnosia.model import Pragnosia

__all__ = ["Pragnosia", "PragnosiaConfig", "__version__"]
