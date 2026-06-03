"""outline_detection — rule-based Tibetan text boundary detection."""

from .api import detect_breakpoints
from .detector import RuleBasedDetector

__all__ = ["detect_breakpoints", "RuleBasedDetector"]
__version__ = "0.1.0"
