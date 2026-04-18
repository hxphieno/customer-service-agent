#!/usr/bin/env python3
"""测试前 N 道题"""
import pandas as pd
from src.agent.graph import run_agent
from src.utils.formatter import format_answer

INPUT_CSV = "question_public.csv"
N = 3  # 测试前 3 道题

def main():
    print(f"=== 测试前 {N} 道题 ===\n")
    df = pd.read_csv(INPUT_CSV)

    results = []
    for idx, row in df.head(N).iterrows():
        qid = row["id"]
        question = str(row["question"])

        print(f"\n{'='*80}")
        print(f"[{qid}] 题目 ({idx+1}/{N})")
        print(f"问题: {question[:200]}{'...' if len(question) > 200 else ''}")
        print(f"{'-'*80}")

        try:
            result = run_agent(question, None)
            final_answer = result.get("final_answer", "")
            used_images = result.get("used_images", [])
            question_type = result.get("question_type", "")
            product = result.get("product", "")
            validation_passed = result.get("validation_passed", False)

            ret = format_answer(final_answer, used_images)

            print(f"✅ 完成")
            print(f"  类型: {question_type} | 产品: {product or '无'}")
            print(f"  验证: {'通过' if validation_passed else '未通过'}")
            print(f"  答案长度: {len(final_answer)} 字符")
            print(f"  图片数量: {len(used_images)}")
            if used_images:
                print(f"  图片 ID: {used_images}")
            print(f"  答案预览: {final_answer[:150]}{'...' if len(final_answer) > 150 else ''}")

            results.append({"id": qid, "ret": ret})
        except Exception as e:
            print(f"❌ 失败: {e}")
            import traceback
            print(f"  错误详情:\n{traceback.format_exc()}")
            results.append({"id": qid, "ret": f"ERROR: {str(e)}"})

    print(f"\n{'='*80}")
    print(f"=== 完成 {len(results)}/{N} 道题 ===")

    # 保存结果
    output_df = pd.DataFrame(results)
    output_df.to_csv("test_results.csv", index=False, encoding="utf-8-sig")
    print(f"结果已保存到 test_results.csv")

if __name__ == "__main__":
    main()
