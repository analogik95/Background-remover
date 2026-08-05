"""vidsticker - turn a video into a transparent animated sticker."""

from .matting import MatteConfig
from .pipeline import StickerOptions, StickerResult, create_sticker

__version__ = "1.0.0"
__all__ = ["MatteConfig", "StickerOptions", "StickerResult", "create_sticker", "__version__"]
