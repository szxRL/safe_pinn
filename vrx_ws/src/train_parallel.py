import subprocess
import os
import time
import signal
import sys

# ================= 配置区域 =================
NUM_ENVS = 10                  # 并行环境数量
START_DOMAIN_ID = 10          # 起始 ROS_DOMAIN_ID (防止与本机其他 ROS 节点冲突)
LAUNCH_FILE_PATH = 'single_env_launch.py' # 请确保路径正确，建议用绝对路径
USE_SOFTWARE_RENDERING = False  # True: Gazebo 使用软件渲染，降低 GPU 显存占用但会增加 CPU 压力
# ===========================================

def signal_handler(sig, frame):
    print("\n正在接收退出信号，清理所有环境进程...")
    # 这里可以添加更复杂的清理逻辑
    sys.exit(0)

def main():
    # 注册 Ctrl+C 处理
    signal.signal(signal.SIGINT, signal_handler)

    processes = []
    
    print(f"=== 开始启动 {NUM_ENVS} 个并行 VRX 环境 ===")
    print(f"每个环境包含 2 条 WAM-V 船只 (Headless 模式)")

    for i in range(NUM_ENVS):
        # 1. 复制当前环境变量
        env_vars = os.environ.copy()
        
        # 2. 设置 ROS 通信隔离 (最关键)
        # 每个环境拥有独立的 Domain ID，它们的 Topic 互不可见
        # 例如：环境0 的 /clock 不会干扰 环境1 的 /clock
        current_domain_id = str(START_DOMAIN_ID + i)
        env_vars['ROS_DOMAIN_ID'] = current_domain_id
        
        # 3. 设置 Gazebo 物理服务器隔离 (最关键)
        # 必须设置 partition，否则所有 gz 实例会连到同一个 server
        partition_name = f"vrx_env_{i}"
        env_vars['GZ_PARTITION'] = partition_name
        
        # 4. 其他优化设置
        env_vars['GZ_IP'] = '127.0.0.1' # 强制本地通信

        if USE_SOFTWARE_RENDERING:
            # 将 Gazebo 渲染尽量放到 CPU/Mesa，避免和 MAPPO 训练抢 GPU 显存。
            env_vars['CUDA_VISIBLE_DEVICES'] = ''
            env_vars['LIBGL_ALWAYS_SOFTWARE'] = '1'
            env_vars['GALLIUM_DRIVER'] = 'llvmpipe'
        
        print(f" -> 启动环境 [{i}]: Domain ID={current_domain_id}, Partition={partition_name}")

        # 5. 构造启动命令
        # 如果你的 launch 文件在包里，用: ['ros2', 'launch', '你的包名', 'single_env_launch.py']
        # 这里假设是直接运行文件路径
        cmd = ['ros2', 'launch', LAUNCH_FILE_PATH]
        
        # 6. 启动子进程
        proc = subprocess.Popen(cmd, env=env_vars)
        processes.append(proc)
        
        # 稍微等待一下，错峰启动降低 CPU 冲击
        time.sleep(3)

    print(f"\n所有环境已启动。正在运行中... (按 Ctrl+C 退出)")
    
    try:
        # 保持主进程运行
        while True:
            time.sleep(1)
            # 检查是否有进程意外退出
            for i, p in enumerate(processes):
                if p.poll() is not None:
                    print(f"警告: 环境 {i} 已意外退出！")
    except KeyboardInterrupt:
        pass
    finally:
        print("正在终止所有子进程...")
        for p in processes:
            p.terminate()
        print("完成。")

if __name__ == '__main__':
    main()
