#!/bin/bash
set -ex

# ReviewTrain Scaling实验脚本
# 专门针对ReviewTrain数据集进行scaling实验（合并训练集和测试集）

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/WX24061/Automatic_Review/generation:$PYTHONPATH
export HCCL_RDMA_TIMEOUT=20
export OMP_NUM_THREADS=1

# 3090优化设置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

# 基础训练参数
MODEL_PATH="/WX24061/models/Qwen3-1.7B/qwen/Qwen3-1.7B"
TRAIN_DATA="../datas/ReviewTrain/hf/train.json"
TEST_DATA="../datas/ReviewTrain/hf/test.json"

# LoRA参数（3090优化）
LORA_RANK=32
LORA_ALPHA=64  # 32*2
LORA_DROPOUT=0.1

# ReviewTrain数据集采样规模
SAMPLING_SIZES=("Full")

# 首先合并训练集和测试集
echo "合并训练集和测试集..."
MERGED_DATA="reviewtrain/merged_train_test.json"
mkdir -p reviewtrain

python3 -c "
import json

# 读取训练集
print('读取训练集...')
with open('$TRAIN_DATA', 'r', encoding='utf-8') as f:
    train_data = json.load(f)
print(f'训练集样本数: {len(train_data)}')

# 读取测试集
print('读取测试集...')
with open('$TEST_DATA', 'r', encoding='utf-8') as f:
    test_data = json.load(f)
print(f'测试集样本数: {len(test_data)}')

# 合并数据集
merged_data = train_data + test_data
print(f'合并后总样本数: {len(merged_data)}')

# 保存合并后的数据集
with open('$MERGED_DATA', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print(f'合并数据集已保存到: $MERGED_DATA')
"

# 获取合并后的总样本数
REVIEWTRAIN_MERGED_SIZE=$(python3 -c "
import json
with open('$MERGED_DATA', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(len(data))
")

echo "开始ReviewTrain Scaling实验（合并数据集）..."
echo "模型路径: $MODEL_PATH"
echo "训练数据: $TRAIN_DATA"
echo "测试数据: $TEST_DATA"
echo "合并数据: $MERGED_DATA"
echo "合并后总样本数: $REVIEWTRAIN_MERGED_SIZE"
echo "LoRA参数: rank=$LORA_RANK, alpha=$LORA_ALPHA"
echo "采样规模: ${SAMPLING_SIZES[@]}"

# 创建结果记录文件
RESULTS_FILE="reviewtrain_scaling_results.json"
echo '{"experiments": []}' > $RESULTS_FILE

# 创建实验目录
mkdir -p checkpoints

# 为每个采样规模进行训练
for sampling_size in "${SAMPLING_SIZES[@]}"; do
    # 确定实际采样数量
    case $sampling_size in
        # "1K") actual_size=1000 ;;
        # "2K") actual_size=2000 ;;
        # "4K") actual_size=4000 ;;
        # "8K") actual_size=8000 ;;
        # "16K") actual_size=16000 ;;
        "Full") actual_size=$REVIEWTRAIN_MERGED_SIZE ;;
    esac
    
    # 创建采样后的数据集
    sampled_data_path="reviewtrain/train_${sampling_size}_merged.json"
    
    echo "创建采样数据集: $sampled_data_path (样本数: $actual_size)"
    
    # 简化的数据采样
    python3 -c "
import json
import random
random.seed(3407)
with open('$MERGED_DATA', 'r', encoding='utf-8') as f:
    data = json.load(f)
if len(data) > $actual_size:
    sampled_data = random.sample(data, $actual_size)
else:
    sampled_data = data
with open('$sampled_data_path', 'w', encoding='utf-8') as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=2)
print(f'采样完成: {len(sampled_data)} 样本 -> $sampled_data_path')
"
    
    # 设置实验名称和输出目录
    RUN_NAME="qwen3_1.7b_reviewtrain_scaling_${sampling_size}_merged"
    OUTPUT_DIR="checkpoints/$RUN_NAME"
    
    echo "开始训练: $RUN_NAME"
    echo "训练数据: $sampled_data_path"
    echo "输出目录: $OUTPUT_DIR"
    
    # 记录开始时间
    start_time=$(date +%s)
    
    # 使用 ms-swift 进行训练（使用合并后的数据集，无验证集）
    swift sft \
        --model $MODEL_PATH \
        --train_type lora \
        --dataset $sampled_data_path \
        --torch_dtype bfloat16 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --learning_rate 2e-4 \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --target_modules q_proj k_proj v_proj o_proj \
        --gradient_accumulation_steps 8 \
        --save_steps 2000 \
        --save_total_limit 3 \
        --logging_steps 20 \
        --max_length 32000 \
        --truncation_strategy right \
        --output_dir $OUTPUT_DIR \
        --gradient_checkpointing \
        --dataloader_num_workers 4 \
        --ddp_find_unused_parameters false \
        --warmup_ratio 0.1 \
        --lr_scheduler_type cosine \
        --weight_decay 0.01 \
        --remove_unused_columns false \
        --seed 3407 \
        --report_to swanlab \
        --swanlab_project automatic-review-scaling \
        --swanlab_exp_name $RUN_NAME \
        --dataloader_pin_memory true \
        --dataloader_persistent_workers true \
        --fp16 false \
        --bf16 true
    
    # 记录结束时间
    end_time=$(date +%s)
    training_time=$((end_time - start_time))
    
    echo "训练完成: $RUN_NAME"
    echo "训练时间: ${training_time} 秒 (${training_time/60} 分钟)"
    
    # 记录实验结果
    python3 -c "
import json
import os

# 读取现有结果
with open('$RESULTS_FILE', 'r', encoding='utf-8') as f:
    results = json.load(f)

# 添加新实验结果
experiment_result = {
    'dataset': 'ReviewTrain_Merged',
    'sampling_size': '$sampling_size',
    'actual_samples': $actual_size,
    'run_name': '$RUN_NAME',
    'output_dir': '$OUTPUT_DIR',
    'training_time_seconds': $training_time,
    'training_time_minutes': $training_time/60,
    'lora_rank': $LORA_RANK,
    'lora_alpha': $LORA_ALPHA,
    'lora_dropout': $LORA_DROPOUT,
    'epochs': 3,
    'batch_size': 2,
    'gradient_accumulation_steps': 4,
    'max_length': 32000,
    'merged_dataset': true
}

results['experiments'].append(experiment_result)

# 保存更新后的结果
with open('$RESULTS_FILE', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f'实验结果已记录到: $RESULTS_FILE')
"
    
    echo "实验 $RUN_NAME 完成！"
    echo "----------------------------------------"
done

echo "ReviewTrain Scaling实验完成！"
echo "结果文件: $RESULTS_FILE"

# 生成ReviewTrain实验总结
python3 -c "
import json
import pandas as pd
from datetime import datetime

# 读取实验结果
with open('$RESULTS_FILE', 'r', encoding='utf-8') as f:
    results = json.load(f)

# 转换为DataFrame进行分析
df = pd.DataFrame(results['experiments'])

# 生成总结报告
summary_file = 'reviewtrain_scaling_summary.txt'
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write('ReviewTrain Scaling实验总结报告（合并数据集）\\n')
    f.write('=' * 50 + '\\n')
    f.write(f'实验时间: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}\\n')
    f.write(f'总实验数: {len(df)}\\n')
    f.write(f'数据集类型: 训练集+测试集合并\\n\\n')
    
    # 时间开销分析
    f.write('时间开销分析:\\n')
    f.write('-' * 30 + '\\n')
    for _, row in df.iterrows():
        f.write(f'采样规模: {row[\"sampling_size\"]} ({row[\"actual_samples\"]} 样本)\\n')
        f.write(f'  训练时间: {row[\"training_time_seconds\"]} 秒 ({row[\"training_time_minutes\"]:.1f} 分钟)\\n')
        f.write(f'  平均每样本时间: {row[\"training_time_seconds\"]/row[\"actual_samples\"]:.3f} 秒/样本\\n\\n')
    
    # 时间趋势分析
    f.write('时间趋势分析:\\n')
    f.write('-' * 30 + '\\n')
    df_sorted = df.sort_values('actual_samples')
    for i, row in df_sorted.iterrows():
        f.write(f'{row[\"sampling_size\"]}: {row[\"training_time_minutes\"]:.1f} 分钟\\n')
    
    # 计算总时间
    total_time = df['training_time_seconds'].sum()
    f.write(f'\\n总实验时间: {total_time} 秒 ({total_time/3600:.1f} 小时)\\n')

print(f'ReviewTrain实验总结已保存到: {summary_file}')
"

echo "ReviewTrain Scaling实验总结已生成！"
echo "实验目录: scaling_experiments/"
echo "检查点目录: scaling_experiments/checkpoints/"
echo "结果文件: $RESULTS_FILE"