#!/bin/bash
# 批量处理脚本 - 确保环境变量正确

# 清除可能冲突的环境变量
unset ANTHROPIC_BASE_URL

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 运行批量处理
python3 run_batch.py
