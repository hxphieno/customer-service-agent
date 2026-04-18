#!/usr/bin/env python3
"""简单测试脚本 - 逐步调试"""
import os
os.environ.pop('ANTHROPIC_BASE_URL', None)  # 移除环境变量

from src.agent.graph import get_graph

# 创建初始状态
initial_state = {
    "question": "冰箱不制冷怎么办？",
    "image_b64": None,
}

print("开始逐步执行...")
graph = get_graph()

# 执行并打印每一步
for i, step in enumerate(graph.stream(initial_state)):
    print(f"\n=== 步骤 {i+1} ===")
    for key, value in step.items():
        print(f"节点: {key}")
        if isinstance(value, dict):
            for k, v in value.items():
                if k == 'retrieved_chunks':
                    print(f"  {k}: {len(v)} chunks")
                elif k == 'final_answer':
                    print(f"  {k}: {v[:200] if v else '(空)'}")
                elif k == 'draft_answer':
                    print(f"  {k}: {v[:200] if v else '(空)'}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"  值: {value}")

print("\n=== 完成 ===")
