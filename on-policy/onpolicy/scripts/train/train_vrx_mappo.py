#!/usr/bin/env python
import sys
import os
import torch
import numpy as np
from pathlib import Path

# 导入 on-policy 库的基础配置解析器
from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

# 导入我们刚刚编写的环境和 Runner
# 请确保这里的路径与您实际存放 vrx_env_mappo.py 和 vrx_runner_mappo.py 的位置一致
from onpolicy.envs.vrx.vrx_env_mappo import VRXMAPPOEnv
from onpolicy.runner.shared.vrx_runner_mappo import VRXRunner

def make_train_env(all_args):
    """创建并行环境池"""
    def get_env_fn(rank):
        def init_env():
            # 这里实例化了您的 VRX MAPPO 环境，rank 会自动对应 ROS_DOMAIN_ID
            env = VRXMAPPOEnv(all_args, rank)
            return env
        return init_env

    # 根据线程数决定使用单进程 DummyVecEnv 还是多进程 SubprocVecEnv
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def parse_args(args, parser):
    """在 on-policy 原有参数基础上，增加针对 VRX 环境的自定义参数"""
    parser.add_argument('--scenario_name', type=str, default='sydney_regatta', help="Gazebo world 名字")
    parser.add_argument('--num_agents', type=int, default=2, help="无人船(USV)的数量")
    parser.add_argument('--control_freq', type=float, default=1.0, help="控制频率(Hz)")
    parser.add_argument('--max_steps', type=int, default=200, help="每个回合的最大步数")

    all_args = parser.parse_known_args(args)[0]
    return all_args

def main(args):
    # 1. 解析参数
    parser = get_config()
    all_args = parse_args(args, parser)

    # 2. 设置硬件设备 (CPU/GPU)
    if all_args.cuda and torch.cuda.is_available():
        print(">>> 训练硬件: GPU")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print(">>> 训练硬件: CPU")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 3. 创建日志保存目录
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") \
              / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # 如果有 wandb 则关闭它，强制使用本地 Tensorboard
    all_args.use_wandb = False 

    # 4. 设置全局随机种子以保证可复现性
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # 5. 初始化环境
    print(">>> 正在初始化 VRX MAPPO 环境池...")
    envs = make_train_env(all_args)

    # 6. 配置并初始化 Runner
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": None,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir
    }

    print(">>> 正在初始化 MAPPO Runner...")
    runner = VRXRunner(config)

    # 7. 开始训练
    print(">>> 开始执行强化学习训练循环...")
    
    runner.run()

    # 8. 训练结束清理
    envs.close()
    if all_args.use_wandb:
        import wandb
        wandb.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()

if __name__ == "__main__":
    main(sys.argv[1:])