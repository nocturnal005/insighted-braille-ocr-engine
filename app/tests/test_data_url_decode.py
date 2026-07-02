"""Unit tests for safe data URL decoding."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from app.ocr.image_decode import ImageDecodeError, decode_data_url
from app.tests.helpers import make_data_url

LIMITS = {"max_bytes": 10 * 1024 * 1024, "max_pixels": 40_000_000}


def test_valid_png_decodes_to_grayscale_array():
    gray, byte_count = decode_data_url(make_data_url("hello"), "image/png", **LIMITS)
    assert isinstance(gray, np.ndarray)
    assert gray.dtype == np.uint8
    assert gray.ndim == 2
    assert gray.size > 0
    assert byte_count > 0


def test_non_data_url_rejected():
    with pytest.raises(ImageDecodeError) as exc_info:
        decode_data_url("https://example.com/image.png", "image/png", **LIMITS)
    assert exc_info.value.flag.category == "low_image_quality"
    assert exc_info.value.flag.severity == "high"


def test_invalid_base64_rejected():
    with pytest.raises(ImageDecodeError):
        decode_data_url("data:image/png;base64,QUJD===", "image/png", **LIMITS)


def test_garbage_characters_rejected():
    with pytest.raises(ImageDecodeError):
        decode_data_url("data:image/png;base64,@@@@", "image/png", **LIMITS)


def test_empty_payload_rejected():
    with pytest.raises(ImageDecodeError):
        decode_data_url("data:image/png;base64,", "image/png", **LIMITS)


def test_valid_base64_but_not_an_image_rejected():
    payload = base64.b64encode(b"this is not an image at all").decode()
    with pytest.raises(ImageDecodeError):
        decode_data_url(f"data:image/png;base64,{payload}", "image/png", **LIMITS)


def test_oversized_image_rejected():
    data_url = make_data_url("hello world")
    with pytest.raises(ImageDecodeError) as exc_info:
        decode_data_url(data_url, "image/png", max_bytes=64, max_pixels=40_000_000)
    assert "large" in exc_info.value.flag.reason.lower()


def test_pixel_limit_rejected():
    data_url = make_data_url("hello world")
    with pytest.raises(ImageDecodeError):
        decode_data_url(data_url, "image/png", max_bytes=10 * 1024 * 1024, max_pixels=100)


def test_pdf_declared_mime_rejected_with_pdf_message():
    with pytest.raises(ImageDecodeError) as exc_info:
        decode_data_url("data:image/png;base64,QUJE", "application/pdf", **LIMITS)
    assert "PDF" in exc_info.value.flag.reason


def test_pdf_data_url_mime_rejected_with_pdf_message():
    with pytest.raises(ImageDecodeError) as exc_info:
        decode_data_url("data:application/pdf;base64,QUJE", "image/png", **LIMITS)
    assert "PDF" in exc_info.value.flag.reason


def test_unsupported_declared_mime_rejected():
    with pytest.raises(ImageDecodeError) as exc_info:
        decode_data_url(make_data_url("hi"), "image/gif", **LIMITS)
    assert "Unsupported MIME type" in exc_info.value.flag.reason
