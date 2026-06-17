# Safe PINN VRX MAPPO

## 环境版本

本项目当前验证环境如下，建议优先使用相同或兼容版本：

| 组件 | 版本 |
| --- | --- |
| Ubuntu | 24.04.3 LTS (noble) |
| ROS 2 | Jazzy |
| Python | 3.12.3 |
| PyTorch | 2.5.1+cu118 |


## 获取仓库代码

```bash 
git clone https://github.com/szxRL/safe_pinn.git
```

## 编译 VRX 环境

```bash
cd ~/safe_pinn/vrx_ws

source /opt/ros/jazzy/setup.bash

colcon build --merge-install

. install/setup.bash
```

## MAPPO 算法部分

```bash

cd ~/safe_pinn

conda activate mappo_vrx

python -m pip install -r requirements.txt

# install on-policy package
cd on-policy

python -m pip install -e .
```




## 启动 VRX 并行环境

```bash
# 先加载 ROS 本体
source /opt/ros/jazzy/setup.bash

# 再加载你编译好的工作区
source ~/safe_pinn/vrx_ws/install/setup.bash

cd ~/safe_pinn/vrx_ws/src

# 启动并行环境，默认启动10个进程
python3 train_parallel.py
```

## 可视化查看其中一个训练进程

重新开启一个终端：

```bash
# 先加载 ROS 本体
source /opt/ros/jazzy/setup.bash

# 再加载你编译好的工作区
source ~/safe_pinn/vrx_ws/install/setup.bash

export ROS_DOMAIN_ID=10
export GZ_PARTITION=vrx_env_0

gz sim -g
```


## 开启训练算法

重新开启一个终端：

```bash
# 先加载 ROS 本体
source /opt/ros/jazzy/setup.bash

# 再加载你编译好的工作区
source ~/safe_pinn/vrx_ws/install/setup.bash

cd ~/safe_pinn/on-policy/onpolicy/scripts

chmod +x ./train_vrx_mappo.sh

./train_vrx_mappo.sh
```

