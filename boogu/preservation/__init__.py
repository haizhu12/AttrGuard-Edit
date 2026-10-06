"""Source-aligned, per-attribute conditioning for Boogu generation."""

from .adapters import SourceConditionedTransformer
from .conditioning import ATTRIBUTE_NAMES, SourceCondition, prepare_condition

__all__ = [
    "ATTRIBUTE_NAMES",
    "SourceCondition",
    "SourceConditionedTransformer",
    "prepare_condition",
]
