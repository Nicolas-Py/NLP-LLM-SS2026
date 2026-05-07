"""latinbench — evaluate dependency parsers on EvaLatin 2024."""
from .core import Bench, Model
from .models import MODELS

__all__ = ["Bench", "Model", "MODELS"]
