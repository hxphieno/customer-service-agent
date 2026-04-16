# tests/test_parser.py
import json, tempfile
from pathlib import Path
from src.ingest.parser import parse_manual, extract_product_name

def test_extract_product_name():
    assert extract_product_name(Path("手册/冰箱手册.txt")) == "冰箱"
    assert extract_product_name(Path("手册/VR头显手册.txt")) == "VR头显"
    assert extract_product_name(Path("手册/空气净化器手册.txt")) == "空气净化器"

def test_parse_manual_basic():
    data = ["# 第一章\n一些文字<PIC>更多文字<PIC>\n# 第二章\n其他内容", ["img_01", "img_02"]]
    with tempfile.NamedTemporaryFile(mode="w", suffix="冰箱手册.txt", delete=False, encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        tmp_path = Path(f.name)
    result = parse_manual(tmp_path)
    assert result["product"] == "冰箱"
    assert result["text"] == data[0]
    assert result["image_ids"] == ["img_01", "img_02"]
    assert result["pic_count"] == 2
    tmp_path.unlink()

def test_parse_manual_no_pics():
    data = ["# 章节\n纯文字内容，没有图片", []]
    with tempfile.NamedTemporaryFile(mode="w", suffix="电钻手册.txt", delete=False, encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        tmp_path = Path(f.name)
    result = parse_manual(tmp_path)
    assert result["image_ids"] == []
    assert result["pic_count"] == 0
    tmp_path.unlink()

def test_parse_manual_pic_position_map():
    data = ["文字<PIC>中间文字<PIC>末尾文字", ["img_a", "img_b"]]
    with tempfile.NamedTemporaryFile(mode="w", suffix="烤箱手册.txt", delete=False, encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        tmp_path = Path(f.name)
    result = parse_manual(tmp_path)
    assert result["pic_position_map"][0] == "img_a"
    assert result["pic_position_map"][1] == "img_b"
    tmp_path.unlink()
