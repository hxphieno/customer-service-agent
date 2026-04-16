# src/ingest/parser.py
import json
from pathlib import Path


def extract_product_name(filepath: Path) -> str:
    """从文件名提取产品名，如 '冰箱手册.txt' -> '冰箱'"""
    stem = filepath.stem
    return stem.split("手册")[0].strip()


def parse_manual(filepath: Path) -> dict:
    """
    解析单本手册 .txt 文件（格式：[text, [image_ids]]）。
    Returns dict with keys: product, text, image_ids, pic_count, pic_position_map, filepath
    """
    import re
    raw = filepath.read_text(encoding="utf-8").strip()
    # Fix invalid escape sequences in the JSON files
    # Strategy: Match backslash followed by invalid escape char, and double it
    # But we need to handle both \X and \\X patterns
    # First, protect valid escapes by temporarily replacing them
    # Then fix all remaining backslashes, then restore valid escapes

    # Simpler approach: just replace all backslashes with double backslashes,
    # then fix the over-escaped valid ones
    raw = raw.replace('\\', '\\\\')
    # Now fix over-escaped valid escapes: \\\\" -> \\", \\\\\\\\ -> \\\\, etc.
    raw = raw.replace('\\\\\\\\', '\\\\')  # \\\\ -> \\
    raw = raw.replace('\\\\"', '\\"')      # \\" -> \"
    raw = raw.replace('\\\\/', '\\/')      # \\/ -> \/
    raw = raw.replace('\\\\b', '\\b')      # \\b -> \b
    raw = raw.replace('\\\\f', '\\f')      # \\f -> \f
    raw = raw.replace('\\\\n', '\\n')      # \\n -> \n
    raw = raw.replace('\\\\r', '\\r')      # \\r -> \r
    raw = raw.replace('\\\\t', '\\t')      # \\t -> \t
    raw = raw.replace('\\\\u', '\\u')      # \\u -> \u

    try:
        data = json.loads(raw)
        text: str = data[0]
        image_ids: list[str] = data[1]
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        raise ValueError(f"Invalid manual format in {filepath}: {e}")

    pic_count = text.count("<PIC>")
    pic_position_map = {i: image_ids[i] for i in range(min(pic_count, len(image_ids)))}

    return {
        "product": extract_product_name(filepath),
        "text": text,
        "image_ids": image_ids,
        "pic_count": pic_count,
        "pic_position_map": pic_position_map,
        "filepath": filepath,
    }


def parse_all_manuals(manuals_dir: Path) -> list[dict]:
    """解析目录下所有 *手册.txt 文件"""
    return [
        parse_manual(f)
        for f in sorted(manuals_dir.glob("*手册.txt"))
        if f.is_file() and "汇总" not in f.name
    ]
