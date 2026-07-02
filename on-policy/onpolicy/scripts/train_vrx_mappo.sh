#!/bin/bash

# 遇到错误立即退出
set -e

export CUDA_VISIBLE_DEVICES=0

# ================= 基础配置 =================
ENV_NAME="VRX_USV"
ALGO_NAME="mappo"            # rmappo: 带 RNN 记忆的 MAPPO；mappo: 不带 RNN
EXP_NAME="vrx_2boats_navigation_old"
# EXP_NAME="vrx_2boats_navigation_safe_pinn"
SCENARIO_NAME="sydney_regatta"

# ================= 环境参数 =================
NUM_AGENTS=2                  # 船只数量
N_ROLLOUT_THREADS=10           # 并行环境数 (必须与 train_parallel.py 中的 NUM_ENVS 保持一致！)
EPISODE_LENGTH=1000            # 单个回合最大步数 (对应 MAX_STEPS)
NUM_ENV_STEPS=20000000         # 训练总交互步数 (即跑多少帧之后停止)

# ================= 网络与算法超参数 =================
PPO_EPOCH=15                  # 每次收集完数据后，PPO 网络更新的迭代次数
LR=5e-4                       # Actor (策略网络) 学习率
CRITIC_LR=5e-4                # Critic (价值网络) 学习率
HIDDEN_SIZE=64                # 隐藏层维度大小

echo "================================================================="
echo "正在启动 MAPPO 训练..."
echo "环境: ${ENV_NAME} | 场景: ${SCENARIO_NAME} | 智能体数: ${NUM_AGENTS}"
echo "并行进程数: ${N_ROLLOUT_THREADS} (请确保后台已有等量的 Gazebo 在运行)"
echo "================================================================="

# 倒计时缓冲，提醒用户检查物理环境
sleep 2

# 执行 Python 训练脚本
python train/train_vrx_mappo.py \
    --env_name ${ENV_NAME} \
    --algorithm_name ${ALGO_NAME} \
    --experiment_name ${EXP_NAME} \
    --scenario_name ${SCENARIO_NAME} \
    --num_agents ${NUM_AGENTS} \
    --n_rollout_threads ${N_ROLLOUT_THREADS} \
    --episode_length ${EPISODE_LENGTH} \
    --num_env_steps ${NUM_ENV_STEPS} \
    --ppo_epoch ${PPO_EPOCH} \
    --hidden_size ${HIDDEN_SIZE} \
    --use_centralized_V True \
    --use_ReLU \
    --lr ${LR} \
    --critic_lr ${CRITIC_LR} \
    --use_wandb False \
    # --use_pinn
    

echo "训练运行结束或被手动终止。"
