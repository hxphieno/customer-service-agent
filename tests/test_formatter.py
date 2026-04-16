# tests/test_formatter.py
import pytest
from src.utils.formatter import format_answer


def test_format_text_only():
    """Test formatting answer with text only (no images)."""
    answer = format_answer("这是一个纯文本回答。", [])
    assert answer == "这是一个纯文本回答。"


def test_format_with_images():
    """Test formatting answer with multiple images."""
    answer = format_answer(
        "第一部分 <PIC> 第二部分 <PIC> 第三部分",
        ["img_001", "img_002"]
    )
    expected = '"第一部分 <PIC> 第二部分 <PIC> 第三部分", ["img_001", "img_002"]'
    assert answer == expected


def test_format_single_image():
    """Test formatting answer with single image."""
    answer = format_answer("请参考下图 <PIC>", ["img_123"])
    expected = '"请参考下图 <PIC>", ["img_123"]'
    assert answer == expected
