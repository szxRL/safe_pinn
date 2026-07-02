# Safe PINN VRX MAPPO

## 环境版本

本项目当前验证环境如下，建议优先使用相同或兼容版本：

| 组件 | 版本 |
| --- | --- |
| Ubuntu | 24.04.3 LTS (noble) |
| ROS 2 | Jazzy |
| Python | 3.12.3 |
| PyTorch | 2.5.1+cu118 |


## 部署docker
docker安装需求。服务器应该是已经安装了docker的，可以使用docker ps 查看docker是否安装成功，应该有一些正在运行的容器。

https://github.com/HonuRobotics/dockwater/wiki/Install-Dependencies

Step 1: Install Docker
Step 2: Set up Nvidia Container Toolkit
这两步服务器应该都已经可以了。


Step 3: Install Rocker
这一步可以尝试运行下面命令看一下是否有rocker。

我的使用rocker的docker启动命令如下：

```bash
rocker --nvidia runtime --env NVIDIA_VISIBLE_DEVICES=1 --env NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute --user --home --shm-size 2g ghcr.io/osrf/vrx-devel:latest /bin/bash
```

其中NVIDIA_VISIBLE_DEVICES=1代表选择某一块显卡作为docker的镜像环境所在位置。


## 获取仓库代码

```bash 
git clone https://github.com/szxRL/safe_pinn.git
```

<!-- ## 编译 VRX 环境

```bash
cd ~/safe_pinn/vrx_ws

source /opt/ros/jazzy/setup.bash

colcon build --merge-install

. install/setup.bash

colcon build --symlink-install --packages-select vrx_gz
source install/setup.bash

```

更新使用colcon build --symlink-install --packages-select vrx_gz后不使用显存，改而使用内存 -->


## 配置MAPPO 算法部分

```bash

cd ~/safe_pinn

conda activate mappo_vrx

python -m pip install -r requirements.txt

# install on-policy package
cd on-policy

python -m pip install -e .
```

## 启动 VRX 并行环境

使用tmux开启终端

然后使用
```bash
rocker --nvidia runtime --env NVIDIA_VISIBLE_DEVICES=1 --env NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute --user --home --shm-size 2g ghcr.io/osrf/vrx-devel:latest /bin/bash
```
开启容器

然后运行
```bash
cd ~/safe_pinn/vrx_ws
# 先加载 ROS 本体
source /opt/ros/jazzy/setup.bash

# 再加载你编译好的工作区
source ~/safe_pinn/vrx_ws/install/setup.bash
colcon build --symlink-install --packages-select vrx_gz
source install/setup.bash
```
更新使用colcon build --symlink-install --packages-select vrx_gz后不使用显存，改而使用内存


```bash
# 先加载 ROS 本体
source /opt/ros/jazzy/setup.bash

# 再加载你编译好的工作区
source ~/safe_pinn/vrx_ws/install/setup.bash

cd ~/safe_pinn/vrx_ws/src

# 启动并行环境，默认启动10个进程
python3 train_parallel.py
```

## 开启训练算法

重新使用tmux开启一个终端：
然后进入之前开启的容器中
```bash
docker exec -it -u szx baabf8a29d4d /bin/bash
```
其中szx是我的用户名 baabf8a29d4d 是docker ps后看到的你启动的docker镜像的id

```bash
# 先加载 ROS 本体
source /opt/ros/jazzy/setup.bash

# 再加载你编译好的工作区
source ~/safe_pinn/vrx_ws/install/setup.bash

conda activate mappo_vrx

cd ~/safe_pinn/on-policy/onpolicy/scripts

chmod +x ./train_vrx_mappo.sh

./train_vrx_mappo.sh
```

## 如果想可视化查看其中一个训练进程

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