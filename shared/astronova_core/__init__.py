"""
AstroNova Core — Shared Library
================================
Production-grade shared utilities for all AstroNova microservices.
"""

from astronova_core.dynamic_logging import (
    DynamicLoggingManager,
    DynamicLoggingMiddleware,
    dynamic_logger_manager,
)

__version__ = "1.0.0"
__author__ = "AstroNova Team, ISRO"
__license__ = "MIT"

__all__ = [
    "DynamicLoggingManager",
    "DynamicLoggingMiddleware",
    "__author__",
    "__license__",
    "__version__",
    "dynamic_logger_manager",
]
