# run_batch.py
"""批量跑 question_public.csv，生成 submission_[timestamp].csv"""
import asyncio
import pandas as pd
from datetime import datetime
from pathlib import Path
from src.agent.graph import run_agent
from src.utils.formatter import format_answer

INPUT_CSV = "question_public.csv"
MAX_CONCURRENT = 5  # 控制并发，避免触发 Anthropic rate limit


async def process_question(sem: asyncio.Semaphore, row: pd.Series) -> dict:
    async with sem:
        loop = asyncio.get_event_loop()
        # run_agent 是同步函数，在线程池中运行
        result = await loop.run_in_executor(
            None, run_agent, str(row["question"]), None
        )
        final_answer = result.get("final_answer", "")
        used_images = result.get("used_images", [])
        ret = format_answer(final_answer, used_images)
        print(f"  [{row['id']}] done | images: {used_images}")
        return {"id": row["id"], "ret": ret}


async def main():
    print("=== 批量答题 ===\n")
    df = pd.read_csv(INPUT_CSV)
    print(f"共 {len(df)} 道题\n")

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [process_question(sem, row) for _, row in df.iterrows()]
    results = await asyncio.gather(*tasks)

    output_df = pd.DataFrame(results).sort_values("id")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"submission_{timestamp}.csv"
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 已保存: {output_path}（{len(output_df)} 条）")


if __name__ == "__main__":
    asyncio.run(main())
