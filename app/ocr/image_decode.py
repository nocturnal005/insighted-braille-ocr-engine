"""Safe decoding of base64 image data URLs.

Never logs image content. Failures raise ImageDecodeError carrying a
contract-compatible uncertainty flag, so the API always returns controlled
JSON instead of crashing or fabricating text.
"""

from __future__ import annotations

import base64
import binascii
import io
import re

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.models.responses import Flag
from app.ocr.flags import CATEGORY_LOW_IMAGE_QUALITY, make_flag

SUPPORTED_MIME_TYPES = ("image/png", "image/jpeg", "image/jpg")

_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[a-zA-Z0-9.+-]+/[a-zA-Z0-9.+-]+);base64,(?P<payload>[A-Za-z0-9+/=\s]*)$"
)

_PDF_MESSAGE = (
    "PDF OCR is not yet supported in version 1. Please upload a PNG or JPEG "
    "image of the Braille page instead."
)


class ImageDecodeError(Exception):
    """Raised when an uploaded image cannot be safely decoded."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.flag: Flag = make_flag(
            text="",
            reason=reason,
            category=CATEGORY_LOW_IMAGE_QUALITY,
            severity="high",
        )


def _reject_unsupported_mime(mime: str) -> None:
    lowered = mime.strip().lower()
    if lowered == "application/pdf" or lowered.endswith("/pdf"):
        raise ImageDecodeError(_PDF_MESSAGE)
    if lowered not in SUPPORTED_MIME_TYPES:
        raise ImageDecodeError(
            f"Unsupported MIME type '{mime[:100]}'. Supported types: "
            "image/png, image/jpeg, image/jpg."
        )


def decode_data_url(
    data_url: str,
    declared_mime: str,
    *,
    max_bytes: int,
    max_pixels: int,
) -> tuple[np.ndarray, int]:
    """Decode a base64 image data URL into a grayscale uint8 array.

    Returns (grayscale_image, decoded_byte_count). Raises ImageDecodeError
    with a contract-compatible flag on any validation failure.
    """
    _reject_unsupported_mime(declared_mime)

    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        raise ImageDecodeError(
            "dataUrl is not a base64 data URL (expected 'data:<mime>;base64,...')."
        )

    match = _DATA_URL_RE.match(data_url)
    if match is None:
        raise ImageDecodeError("dataUrl could not be parsed as a base64 image data URL.")

    _reject_unsupported_mime(match.group("mime"))

    payload = re.sub(r"\s+", "", match.group("payload"))
    if not payload:
        raise ImageDecodeError("dataUrl contains no base64 image data.")

    approx_bytes = (len(payload) * 3) // 4
    if approx_bytes > max_bytes:
        raise ImageDecodeError(
            f"Image is too large (about {approx_bytes // (1024 * 1024)} MB). "
            f"Maximum allowed is {max_bytes // (1024 * 1024)} MB."
        )

    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise ImageDecodeError("dataUrl base64 payload could not be decoded.") from None

    if len(raw) > max_bytes:
        raise ImageDecodeError("Image exceeds the maximum allowed size.")

    try:
        image = Image.open(io.BytesIO(raw))
        width, height = image.size
        image_format = image.format
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        raise ImageDecodeError(
            "Uploaded data could not be read as a PNG or JPEG image."
        ) from None

    if image_format not in ("PNG", "JPEG"):
        raise ImageDecodeError(
            f"Image content is {image_format or 'unknown'}, not PNG or JPEG."
        )

    if width * height > max_pixels:
        raise ImageDecodeError("Image resolution is too large to process safely.")

    if width < 8 or height < 8:
        raise ImageDecodeError("Image is too small to contain readable Braille.")

    try:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    except Exception:
        raise ImageDecodeError("Image data appears to be corrupted or truncated.") from None

    return gray, len(raw)
