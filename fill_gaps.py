import csv

src = "batch_progress_v3.csv"
dst = "batch_progress_v3_filled.csv"

with open(src, newline="", encoding="utf-8-sig") as f:
    rows = {int(r["id"]): r["ret"] for r in csv.DictReader(f)}

ids = sorted(rows)
full_range = range(ids[0], ids[-1] + 1)

with open(dst, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "ret"])
    for i in full_range:
        w.writerow([i, rows.get(i, "")])

print(f"Written {len(full_range)} rows ({len(full_range) - len(rows)} gaps filled)")
