git获取仓库代码

```bash 
git clone https://github.com/szxRL/safe_pinn.git
```
编译vrx环境
```bash
cd safe_pinn/vrx_ws

source /opt/ros/jazzy/setup.bash

colcon build --merge-install

. install/setup.bash
```

mappo算法部分
```bash

conda activate xxx

# install on-policy package
cd on-policy

pip install -e .
```




开启终端，启动vrx环境

```bash
# 先加载 ROS 本体
source /opt/ros/jazzy/setup.bash

# 再加载你编译好的工作区
source ~/safe_pinn/vrx_ws/install/setup.bash

cd safe_pinn/vrx_ws/src

# 启动并行环境，默认启动10个进程
python3 train_parallel.py
```

可视化查看其中一个训练进程
重新开启一个终端

```bash
# 先加载 ROS 本体
source /opt/ros/jazzy/setup.bash

# 再加载你编译好的工作区
source ~/safe_pinn/vrx_ws/install/setup.bash

export ROS_DOMAIN_ID=10
export GZ_PARTITION=vrx_env_0

gz sim -g
```


开启训练算法，重新开启一个终端
```bash
# 先加载 ROS 本体
source /opt/ros/jazzy/setup.bash

# 再加载你编译好的工作区
source ~/safe_pinn/vrx_ws/install/setup.bash

cd on-policy/onpolicy/scripts

chmod +x ./train_vrx_mappo.sh

./train_vrx_mappo.sh
```

