"""modelduel: is model A really better than model B, or is it cross-validation noise?"""
from .duel import DuelResult, corrected_ttest, duel

__all__ = ["DuelResult", "corrected_ttest", "duel"]
__version__ = "0.1.0"
