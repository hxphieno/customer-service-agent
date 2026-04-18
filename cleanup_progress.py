"""清理 batch_progress_v3.csv：删除 ERROR 行、空 ret 行，修复 ```json 格式输出"""
import pandas as pd
import json

PROGRESS_FILE = "batch_progress_v3.csv"

df = pd.read_csv(PROGRESS_FILE, dtype=str)
original_len = len(df)

# 删除 ERROR 开头、空 ret、以及空文本+仅图片格式
df = df[df["ret"].notna()]
df = df[~df["ret"].str.startswith("ERROR:")]
df = df[~df["ret"].str.match(r'^\"?\s*\",\s*\[', na=False)]
after_filter = len(df)

def fix_ret(ret: str) -> str:
    ret = ret.strip()
    if not ("```" in ret or ret.startswith("{")):
        return ret
    start = ret.find("{")
    end = ret.rfind("}")
    if start == -1 or end == -1:
        return ret
    raw = ret[start:end+1]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # 中文引号等导致解析失败，用正则提取 answer 值
        import re
        m = re.search(r'"answer"\s*:\s*"(.*?)"\s*,\s*"image_ids"', raw, re.DOTALL)
        if not m:
            return ret
        answer = m.group(1)
        answer = " ".join(answer.replace("\n", " ").split())
        return answer
    answer = parsed.get("answer", "")
    image_ids = parsed.get("image_ids", [])
    if not answer:
        return ret
    answer = " ".join(answer.replace("\n", " ").split())
    if image_ids:
        return f'"{answer}", {str(image_ids).replace(chr(39), chr(34))}'
    return answer

df["ret"] = df["ret"].apply(fix_ret)

# 再次过滤修复后仍为空的
df = df[df["ret"].str.strip() != ""]

df.to_csv(PROGRESS_FILE, index=False, encoding="utf-8-sig")
print(f"原始: {original_len} 行")
print(f"过滤后: {after_filter} 行（删除 {original_len - after_filter} 行 ERROR/空）")
print(f"最终: {len(df)} 行")
