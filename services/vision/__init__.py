"""AstroNova Vision Service package.

Exposes the multimodal solar prediction pipeline and FastAPI router.
"""

from .inference import VisionInferencePipeline
from .api import router

__all__ = ["VisionInferencePipeline", "router"]
