# tests/test_chunker.py
from src.ingest.chunker import split_into_parent_chunks, split_parent_into_children, chunk_manual

def test_split_by_heading():
    text = "# 第一章\n内容A\n# 第二章\n内容B"
    chunks = split_into_parent_chunks(text, "冰箱", {})
    assert len(chunks) == 2
    assert chunks[0]["chapter"] == "第一章"
    assert "内容A" in chunks[0]["text"]

def test_parent_chunk_carries_image_ids():
    text = "# 安装说明\n安装步骤<PIC>注意事项\n# 使用方法\n正常使用"
    pic_position_map = {0: "img_01"}
    chunks = split_into_parent_chunks(text, "冰箱", pic_position_map)
    assert chunks[0]["image_ids"] == ["img_01"]
    assert chunks[1]["image_ids"] == []

def test_child_chunks_inherit_parent_image_ids():
    parent = {
        "product": "冰箱", "chapter": "安装说明",
        "parent_id": "冰箱_安装说明",
        "text": "安装步骤说明。" * 100,
        "image_ids": ["img_01", "img_02"],
    }
    children = split_parent_into_children(parent, max_chars=100)
    assert len(children) > 1
    for child in children:
        assert child["image_ids"] == ["img_01", "img_02"]
        assert child["parent_id"] == "冰箱_安装说明"
        assert child["chunk_type"] == "child"

def test_chunk_manual_returns_both(tmp_path):
    import json
    from pathlib import Path
    from src.ingest.parser import parse_manual
    data = ["# 章节一\n内容文字。" * 60 + "<PIC>" + "\n# 章节二\n其他内容。" * 10, ["img_01"]]
    file = tmp_path / "冰箱手册.txt"
    file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    parsed = parse_manual(file)
    parents, children = chunk_manual(parsed)
    assert len(parents) >= 1
    assert len(children) >= len(parents)
    assert all(c["chunk_type"] == "child" for c in children)
