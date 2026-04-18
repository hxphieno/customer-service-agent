# src/ingest/parser.py
import json
import re
from pathlib import Path

_VALID_ESCAPES = set(r'"\\/bfnrtu')


def _fix_escapes(raw: str) -> str:
    r"""Replace invalid JSON escape sequences like \* with \\*."""
    def replacer(m):
        char = m.group(1)
        return m.group(0) if char in _VALID_ESCAPES else '\\\\' + char
    return re.sub(r'\\(.)', replacer, raw)


def extract_product_name(filepath: Path) -> str:
    """从文件名提取产品名，如 '冰箱手册.txt' -> '冰箱'"""
    stem = filepath.stem
    return stem.split("手册")[0].strip()


def parse_manual(filepath: Path) -> dict:
    """
    解析单本手册 .txt 文件（格式：[text, [image_ids]]）。
    Returns dict with keys: product, text, image_ids, pic_count, pic_position_map, filepath
    """
    raw = filepath.read_text(encoding="utf-8").strip()
    raw = _fix_escapes(raw)

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


def parse_en_manuals(filepath: Path) -> list[dict]:
    """解析汇总英文手册（每行一本，格式同单本手册）"""
    results = []
    raw_lines = filepath.read_text(encoding="utf-8").splitlines()
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            text: str = data[0]
            image_ids: list[str] = data[1]
        except (json.JSONDecodeError, IndexError):
            continue
        # 用图片ID前缀作为产品名，如 Manual10_0 -> en_Manual10
        product = "en_unknown"
        if image_ids:
            prefix = "_".join(image_ids[0].split("_")[:-1])
            if prefix:
                product = f"en_{prefix}"
        pic_count = text.count("<PIC>")
        pic_position_map = {i: image_ids[i] for i in range(min(pic_count, len(image_ids)))}
        results.append({
            "product": product,
            "text": text,
            "image_ids": image_ids,
            "pic_count": pic_count,
            "pic_position_map": pic_position_map,
            "filepath": filepath,
        })
    return results


def parse_all_manuals(manuals_dir: Path) -> list[dict]:
    """解析目录下所有手册（中文单本 + 英文汇总）"""
    manuals = [
        parse_manual(f)
        for f in sorted(manuals_dir.glob("*手册.txt"))
        if f.is_file() and "汇总" not in f.name
    ]
    en_file = manuals_dir / "汇总英文手册.txt"
    if en_file.is_file():
        manuals.extend(parse_en_manuals(en_file))
    return manuals
