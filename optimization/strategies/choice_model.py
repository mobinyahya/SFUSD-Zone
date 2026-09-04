"""Compatibility imports for strategy choice models."""

from choice.models import (  # noqa: F401
    ChoiceModel,
    MNLChoiceModel,
    build_mnl_choice_model,
)

__all__ = [
    "ChoiceModel",
    "MNLChoiceModel",
    "build_mnl_choice_model",
]
