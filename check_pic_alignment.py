"""检查手册原文中 <PIC> 数量与 image_id_list 数量是否对齐"""
import json
from pathlib import Path

manuals_dir = Path("手册")
mismatches = []

for txt_file in sorted(manuals_dir.glob("*.txt")):
    if txt_file.name == "汇总英文手册.txt":
        continue
    try:
        data = json.loads(txt_file.read_text(encoding="utf-8"))
        text, image_ids = data[0], data[1]
        pic_count = text.count("<PIC>")
        img_count = len(image_ids)
        if pic_count != img_count:
            mismatches.append((txt_file.name, pic_count, img_count))
        else:
            print(f"OK  {txt_file.name}: {pic_count} <PIC> = {img_count} images")
    except Exception as e:
        print(f"ERR {txt_file.name}: {e}")

print()
if mismatches:
    print(f"=== 不对齐的手册 ({len(mismatches)}) ===")
    for name, p, i in mismatches:
        print(f"  {name}: <PIC>={p}, image_ids={i}")
else:
    print("所有手册 <PIC> 与 image_ids 数量完全对齐")
