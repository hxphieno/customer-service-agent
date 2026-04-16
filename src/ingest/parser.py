# src/ingest/parser.py
import json
from pathlib import Path


def extract_product_name(filepath: Path) -> str:
    """从文件名提取产品名，如 '冰箱手册.txt' -> '冰箱'"""
    stem = filepath.stem
    # Extract product name between any prefix and "手册"
    if "手册" in stem:
        idx = stem.find("手册")
        product = stem[:idx]
        # Find the first Chinese character or uppercase letter (for VR, etc.)
        start_idx = 0
        for i, char in enumerate(product):
            if '\u4e00' <= char <= '\u9fff' or (char.isupper() and char.isalpha()):
                start_idx = i
                break
        return product[start_idx:].strip()
    return stem.replace("手册", "").strip()


def parse_manual(filepath: Path) -> dict:
    """
    解析单本手册 .txt 文件（格式：[text, [image_ids]]）。
    Returns dict with keys: product, text, image_ids, pic_count, pic_position_map, filepath
    """
    raw = filepath.read_text(encoding="utf-8").strip()
    data = json.loads(raw)
    text: str = data[0]
    image_ids: list[str] = data[1]

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
    return [parse_manual(f) for f in sorted(manuals_dir.glob("*手册.txt")) if f.is_file()]
