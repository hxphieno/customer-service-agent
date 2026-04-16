# run_batch.py
"""批量跑 question_public.csv，生成 submission_[timestamp].csv"""
import pandas as pd
from datetime import datetime
from src.agent.graph import run_agent
from src.utils.formatter import format_answer

INPUT_CSV = "question_public.csv"


def main():
    print("=== 批量答题 ===\n")
    df = pd.read_csv(INPUT_CSV)
    print(f"共 {len(df)} 道题\n")

    results = []
    for idx, row in df.iterrows():
        try:
            result = run_agent(str(row["question"]), None)
            final_answer = result.get("final_answer", "")
            used_images = result.get("used_images", [])
            ret = format_answer(final_answer, used_images)
            print(f"  [{row['id']}] done ({idx+1}/{len(df)}) | images: {used_images}")
            results.append({"id": row["id"], "ret": ret})
        except Exception as e:
            print(f"  [{row['id']}] FAILED: {e}")
            results.append({"id": row["id"], "ret": f"ERROR: {str(e)}"})

    output_df = pd.DataFrame(results).sort_values("id")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"submission_{timestamp}.csv"
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 已保存: {output_path}（{len(output_df)} 条）")


if __name__ == "__main__":
    main()
