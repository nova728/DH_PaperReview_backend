#!/bin/bash

# 两个模型的启动脚本
# 使用方案2：两个独立的 vLLM 实例

# ============================================
# 启动 Automatic Review 模型（端口 8000）
# ============================================
echo "启动 Automatic Review 模型..."
python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --model /path/to/scientific-reviewer-7b \
    --served-model-name scientific-reviewer-7b \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.45 \
    --tensor-parallel-size 1 &

# 等待第一个模型启动
sleep 10

# ============================================
# 启动 Deep Review 模型（端口 8001）
# ============================================
echo "启动 Deep Review 模型..."
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8001 \
    --model /path/to/deep-review-7b \
    --served-model-name deep-review-7b \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.45 \
    --tensor-parallel-size 1 &

echo "两个模型启动完成！"
echo "Automatic Review 端口: 8000"
echo "Deep Review 端口: 8001"

# 保持脚本运行
wait
