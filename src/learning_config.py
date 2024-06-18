"""This module implements learning config class, used by some NN training functions"""

from dataclasses import dataclass

@dataclass(slots=True)
class LearningConfig:
    """Dataclass representing parameters of training scheme for some model."""
    learning_epochs: int
    learning_rate: float
    min_loss: float
