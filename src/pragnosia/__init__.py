"""
Pragnosia: A Systems Architecture for Continual Representation Learning
Neuroscience-Inspired Design for Memory-Efficient Adaptive Language Models
"""

__version__ = "0.1.0"

from .models.pragnosia_model import PragnosiaModel
from .models.router import HebbianRouter
from .models.expert import ExpertModule
from .memory.hippocampus import Hippocampus
from .memory.neocortex import Neocortex
from .losses.intrinsic import IntrinsicObjective
from .training.trainer import PragnosiaTrainer
from .training.local_trainer import LocalLearningTrainer

__all__ = [
    "PragnosiaModel",
    "HebbianRouter",
    "ExpertModule",
    "Hippocampus",
    "Neocortex",
    "IntrinsicObjective",
    "PragnosiaTrainer",
    "LocalLearningTrainer",  # Constant-VRAM training with local learning
]
