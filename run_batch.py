# run_batch.py
"""批量跑 question_public.csv，生成 submission_[timestamp].csv，支持断点续传"""
import pandas as pd
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from src.agent.graph import run_agent
from src.utils.formatter import format_answer

WORKERS = 4  # 并发线程数

INPUT_CSV = "question_public.csv"
PROGRESS_FILE = "batch_progress_v2.csv"
FAILURE_LOG = "batch_failures.csv"

_file_lock = threading.Lock()


def load_progress():
    """加载已完成的题目"""
    if os.path.exists(PROGRESS_FILE):
        df = pd.read_csv(PROGRESS_FILE)
        completed_ids = set(df["id"].tolist())
        print(f"📂 发现进度文件，已完成 {len(completed_ids)} 道题")
        return df, completed_ids
    return pd.DataFrame(columns=["id", "ret"]), set()


def log_failure(qid, question, error, tb):
    with _file_lock:
        row = pd.DataFrame([{"id": qid, "question": question[:300], "error": str(error), "traceback": tb.strip()}])
        row.to_csv(FAILURE_LOG, mode='a', header=not os.path.exists(FAILURE_LOG), index=False, encoding="utf-8-sig")


def save_result(qid, ret):
    with _file_lock:
        result_df = pd.DataFrame([{"id": qid, "ret": ret}])
        result_df.to_csv(PROGRESS_FILE, mode='a', header=not os.path.exists(PROGRESS_FILE), index=False, encoding="utf-8-sig")


def main():
    print("=== 批量答题（支持断点续传）===\n")

    # 加载进度
    progress_df, completed_ids = load_progress()

    # 加载题目
    df = pd.read_csv(INPUT_CSV)
    total = len(df)
    remaining = df[~df["id"].isin(completed_ids)]

    print(f"总题数: {total}")
    print(f"已完成: {len(completed_ids)}")
    print(f"待处理: {len(remaining)}\n")

    if len(remaining) == 0:
        print("✅ 所有题目已完成！")
        # 生成最终提交文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"submission_{timestamp}.csv"
        progress_df.sort_values("id").to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"已保存最终文件: {output_path}")
        return

    counters = {"success": len(completed_ids), "fail": 0}
    counter_lock = threading.Lock()

    def process(row):
        import traceback
        qid = row["id"]
        question = str(row["question"])
        try:
            import json as _json
            result = run_agent(question, None)
            final_answer = result.get("final_answer", "")
            used_images = result.get("used_images", [])
            # Handle case where final_answer is a raw JSON string like {"answer": "...", "image_ids": [...]}
            if isinstance(final_answer, str) and final_answer.strip().startswith("{"):
                try:
                    parsed = _json.loads(final_answer)
                    final_answer = parsed.get("answer", final_answer)
                    if not used_images:
                        used_images = parsed.get("image_ids", [])
                except _json.JSONDecodeError:
                    pass
            ret = format_answer(final_answer, used_images)
            save_result(qid, ret)
            with counter_lock:
                counters["success"] += 1
                done = counters["success"] + counters["fail"]
            print(f"✅ [{qid}] 完成 | 进度 {done}/{total} | 成功 {counters['success']} 失败 {counters['fail']}")
        except Exception as e:
            tb = traceback.format_exc()
            log_failure(qid, question, e, tb)
            save_result(qid, f"ERROR: {str(e)}")
            with counter_lock:
                counters["fail"] += 1
                done = counters["success"] + counters["fail"]
            print(f"❌ [{qid}] 失败: {e} | 进度 {done}/{total}")

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(process, row) for _, row in remaining.iterrows()]
        for f in as_completed(futures):
            f.result()

    print(f"\n{'='*80}")
    print(f"=== 处理完成 ===")
    print(f"总计: {total} 道")
    print(f"成功: {counters['success']} 道")
    print(f"失败: {counters['fail']} 道")

    # 生成最终提交文件
    final_df = pd.read_csv(PROGRESS_FILE).sort_values("id")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"submission_{timestamp}.csv"
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 已保存最终文件: {output_path}（{len(final_df)} 条）")
    print(f"进度文件保留在: {PROGRESS_FILE}")


if __name__ == "__main__":
    main()
