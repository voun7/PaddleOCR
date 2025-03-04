import importlib.metadata as importlib_metadata

from .paddleocr import PaddleOCR, parse_lang

try:
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["PaddleOCR", "parse_lang"]
