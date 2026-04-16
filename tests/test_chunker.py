# tests/test_chunker.py
from src.ingest.chunker import split_into_parent_chunks, split_parent_into_children, chunk_manual

def test_split_by_heading():
    """Test fixed-length chunking with sentence boundaries."""
    text = "这是第一句话。" * 100 + "这是第二句话。" * 100
    chunks = split_into_parent_chunks(text, "冰箱", {}, max_chars=500)
    assert len(chunks) >= 2
    assert chunks[0]["chapter"] == "part_0"
    assert chunks[1]["chapter"] == "part_1"
    assert all(len(c["text"]) <= 600 for c in chunks)  # Allow some overflow for sentence boundary

def test_parent_chunk_carries_image_ids():
    text = "安装步骤说明。" * 50 + "<PIC>" + "注意事项说明。" * 50 + "使用方法说明。" * 50
    pic_position_map = {0: "img_01"}
    chunks = split_into_parent_chunks(text, "冰箱", pic_position_map, max_chars=500)
    # Find chunk containing <PIC>
    pic_chunk = [c for c in chunks if "<PIC>" in c["text"]][0]
    assert pic_chunk["image_ids"] == ["img_01"]

def test_child_chunks_inherit_parent_image_ids():
    parent = {
        "product": "冰箱", "chapter": "part_0",
        "parent_id": "冰箱_p0",
        "text": "安装步骤说明。" * 100,
        "image_ids": ["img_01", "img_02"],
    }
    children = split_parent_into_children(parent, max_chars=100)
    assert len(children) > 1
    for child in children:
        assert child["image_ids"] == ["img_01", "img_02"]
        assert child["parent_id"] == "冰箱_p0"
        assert child["chunk_type"] == "child"

def test_chunk_manual_returns_both(tmp_path):
    import json
    from pathlib import Path
    from src.ingest.parser import parse_manual
    data = ["内容文字说明。" * 60 + "<PIC>" + "其他内容说明。" * 60, ["img_01"]]
    file = tmp_path / "冰箱手册.txt"
    file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    parsed = parse_manual(file)
    parents, children = chunk_manual(parsed)
    assert len(parents) >= 1
    assert len(children) >= len(parents)
    assert all(c["chunk_type"] == "child" for c in children)
