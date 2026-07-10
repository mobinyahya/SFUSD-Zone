"""Compatibility imports for strategy choice models."""

from choice.models import (  # noqa: F401
    ChoiceModel,
    DistanceChoiceModel,
    MNLChoiceModel,
    get_configured_choice_model,
    get_choice_model,
)

__all__ = [
    "ChoiceModel",
    "DistanceChoiceModel",
    "MNLChoiceModel",
    "get_configured_choice_model",
    "get_choice_model",
]
