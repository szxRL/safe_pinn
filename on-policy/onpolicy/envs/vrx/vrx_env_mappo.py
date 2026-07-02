import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Duration
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import time
import threading
import subprocess
import os

from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker 

# ================= 全局配置 =================
WORLD_NAME = "sydney_regatta"

# 1. 坐标原点
ORIGIN_LAT = -33.72422304
ORIGIN_LON = 150.67972653

MAX_STEPS = 500            
COLLISION_THRESHOLD = 5.0  
GOAL_THRESHOLD = 5.0      
GOAL_REWARD = 20.0
GOAL_STAY_REWARD = 1.0
COLLISION_PENALTY = -50.0
OBSTACLE_PROXIMITY_DISTANCE = 8.0
OBSTACLE_PROXIMITY_PENALTY = 0.2
SENSOR_READY_TIMEOUT = 20.0
RESET_POSITION_TOLERANCE = 15.0
RESET_STATE_TIMEOUT = 20.0
RESET_RETRIES = 3
RESET_SETTLE_SIM_TIME = 1.0

NUM_OBSTACLES = 5         
OBSTACLE_RADIUS = 4      
BOUNDARY_OBSTACLES = [
    (-150.0, -50.0),
    (100.0, -50.0),
]
BOUNDARY_OBSTACLE_SIZE = (5.0, 200.0, 2.0)
OTHER_BOAT_RADIUS = 4.0
MIN_RESET_BOAT_DISTANCE = 2.0 * OTHER_BOAT_RADIUS + COLLISION_THRESHOLD
NUM_RANGE_SECTORS = 12
RANGE_MAX_DISTANCE = 50.0
RANGE_RAYS_PER_SECTOR = 11

CONTROL_FREQ = 1.0        
STEP_DURATION = 1.0 / CONTROL_FREQ  

THRUST_SCALE = 20.0
# ===========================================

def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def latlon_to_xy(lat, lon):
    R = 6378137.0 
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    olat_rad = math.radians(ORIGIN_LAT)
    olon_rad = math.radians(ORIGIN_LON)
    x = R * (lon_rad - olon_rad) * math.cos(olat_rad)
    y = R * (lat_rad - olat_rad)
    return x, y

def ray_circle_intersection(origin, direction, center, radius):
    ox, oy = origin
    dx, dy = direction
    cx, cy = center
    fx, fy = ox - cx, oy - cy
    b = fx * dx + fy * dy
    c = fx * fx + fy * fy - radius * radius
    discriminant = b * b - c
    if discriminant < 0.0:
        return None

    sqrt_disc = math.sqrt(discriminant)
    t_near = -b - sqrt_disc
    t_far = -b + sqrt_disc
    if t_near >= 0.0:
        return t_near
    if t_far >= 0.0:
        return 0.0
    return None

def ray_aabb_intersection(origin, direction, center, size):
    ox, oy = origin
    dx, dy = direction
    cx, cy = center
    sx, sy, _ = size
    min_x, max_x = cx - sx * 0.5, cx + sx * 0.5
    min_y, max_y = cy - sy * 0.5, cy + sy * 0.5

    if min_x <= ox <= max_x and min_y <= oy <= max_y:
        return 0.0

    t_min = -math.inf
    t_max = math.inf

    if abs(dx) < 1e-8:
        if ox < min_x or ox > max_x:
            return None
    else:
        tx1 = (min_x - ox) / dx
        tx2 = (max_x - ox) / dx
        t_min = max(t_min, min(tx1, tx2))
        t_max = min(t_max, max(tx1, tx2))

    if abs(dy) < 1e-8:
        if oy < min_y or oy > max_y:
            return None
    else:
        ty1 = (min_y - oy) / dy
        ty2 = (max_y - oy) / dy
        t_min = max(t_min, min(ty1, ty2))
        t_max = min(t_max, max(ty1, ty2))

    if t_max < 0.0 or t_min > t_max:
        return None
    return max(t_min, 0.0)

class WamvMultiAgent(Node):
    """支持多艘 USV 的 ROS 2 通信节点"""
    def __init__(self, num_agents, rank):
        super().__init__(f'wamv_mappo_agent_{rank}')
        
        self.num_agents = num_agents
        self.rank = rank
        self.rng = np.random.default_rng()
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        
        # 多智能体状态缓存 (新增了 yaw_rate, prev_x, prev_y 用于速度计算)
        self.agents_data = [{
            'lat': None, 'lon': None, 
            'yaw': 0.0, 'yaw_rate': 0.0, 
            'prev_x': None, 'prev_y': None, 
            'range_sectors': np.full(NUM_RANGE_SECTORS, RANGE_MAX_DISTANCE),
            'imu_ready': False,
        } for _ in range(num_agents)]
        
        self.obstacle_positions = np.zeros((NUM_OBSTACLES, 2)) 
        
        # 为船只分配各自专属的目标点 XY 坐标
        self.targets_xy = [
            (45.0, 0.0),      # wamv1 目标点 (原点)
            (-75.0, -100.0)   # wamv2 目标点 (-30, -30)
        ]
        # self.targets_xy = [
        #     (5.0, 0.0),      # wamv1 目标点 (原点)
        #     (-35.0, -100.0)   # wamv2 目标点 (-30, -30)
        # ]
        
        self.pubs_left = []
        self.pubs_right = []
        
        # 动态生成多智能体的话题订阅和发布
        for i in range(self.num_agents):
            robot_name = f"wamv{i+1}"
            
            # 订阅器
            self.create_subscription(NavSatFix, f'/{robot_name}/sensors/gps/gps/fix', 
                                     lambda msg, idx=i: self.gps_callback(msg, idx), 10)
            self.create_subscription(Imu, f'/{robot_name}/sensors/imu/imu/data', 
                                     lambda msg, idx=i: self.imu_callback(msg, idx), 10)
            
            # 发布器
            self.pubs_left.append(self.create_publisher(Float64, f'/{robot_name}/thrusters/left/thrust', 10))
            self.pubs_right.append(self.create_publisher(Float64, f'/{robot_name}/thrusters/right/thrust', 10))

        # RViz
        self.pub_marker = self.create_publisher(Marker, '/visualization_marker', 10)
        self.timer_marker = self.create_timer(1.0, self.publish_rviz_markers)
        
        # 在 Gazebo 注入模型
        self.spawn_gazebo_marker()
        self.spawn_gazebo_obstacles()

    def check_all_ready(self):
        """检查所有智能体的传感器数据是否都已接入"""
        for data in self.agents_data:
            if (
                data['lat'] is None or
                data['lon'] is None or
                not data['imu_ready']
            ):
                return False
        return True

    def check_reset_positions_ready(self, expected_positions, tolerance):
        if not self.check_all_ready():
            return False

        for i, (expected_x, expected_y) in enumerate(expected_positions):
            data = self.agents_data[i]
            curr_x, curr_y = latlon_to_xy(data['lat'], data['lon'])
            if math.hypot(curr_x - expected_x, curr_y - expected_y) > tolerance:
                return False
        return True

    def reset_state_status(self, expected_positions):
        parts = []
        for i, (expected_x, expected_y) in enumerate(expected_positions):
            data = self.agents_data[i]
            gps_ready = data['lat'] is not None and data['lon'] is not None
            if gps_ready:
                curr_x, curr_y = latlon_to_xy(data['lat'], data['lon'])
                pos_err = math.hypot(curr_x - expected_x, curr_y - expected_y)
                pos_text = f"xy=({curr_x:.1f},{curr_y:.1f}) err={pos_err:.1f}m"
            else:
                pos_text = "xy=None err=None"

            parts.append(
                f"agent{i}: gps={gps_ready} imu={data['imu_ready']} "
                f"expected=({expected_x:.1f},{expected_y:.1f}) {pos_text}"
            )
        return "; ".join(parts)

    def wait_for_fresh_reset_state(self, expected_positions, timeout_sec=RESET_STATE_TIMEOUT):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok():
            if self.check_reset_positions_ready(expected_positions, RESET_POSITION_TOLERANCE):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return False

    def clear_sensor_cache(self):
        for data in self.agents_data:
            data['lat'] = None
            data['lon'] = None
            data['yaw'] = 0.0
            data['yaw_rate'] = 0.0
            data['prev_x'] = None
            data['prev_y'] = None
            data['range_sectors'] = np.full(NUM_RANGE_SECTORS, RANGE_MAX_DISTANCE)
            data['imu_ready'] = False

    def wait_sim_time(self, duration_sec):
        target_duration = Duration(seconds=duration_sec)
        start_time = self.get_clock().now()
        while rclpy.ok():
            current_time = self.get_clock().now()
            if current_time.nanoseconds == 0:
                time.sleep(0.001)
                start_time = self.get_clock().now()
                continue
            if (current_time - start_time) >= target_duration:
                break
            time.sleep(0.001)

    def compute_geometric_ranges(self, boat_x, boat_y, other_boats=None):
        ranges = np.full(NUM_RANGE_SECTORS, RANGE_MAX_DISTANCE, dtype=np.float32)
        sector_width = 2.0 * math.pi / NUM_RANGE_SECTORS
        sample_count = max(1, RANGE_RAYS_PER_SECTOR)
        other_boats = other_boats or []

        for sector in range(NUM_RANGE_SECTORS):
            sector_start = -math.pi + sector * sector_width
            if sample_count == 1:
                ray_angles = [sector_start + 0.5 * sector_width]
            else:
                ray_angles = [
                    sector_start + sector_width * j / (sample_count - 1)
                    for j in range(sample_count)
                ]

            min_distance = RANGE_MAX_DISTANCE
            for angle in ray_angles:
                direction = (math.cos(angle), math.sin(angle))

                for obs_x, obs_y in self.obstacle_positions:
                    hit = ray_circle_intersection(
                        (boat_x, boat_y),
                        direction,
                        (float(obs_x), float(obs_y)),
                        OBSTACLE_RADIUS,
                    )
                    if hit is not None and hit < min_distance:
                        min_distance = hit

                for other_x, other_y in other_boats:
                    hit = ray_circle_intersection(
                        (boat_x, boat_y),
                        direction,
                        (float(other_x), float(other_y)),
                        OTHER_BOAT_RADIUS,
                    )
                    if hit is not None and hit < min_distance:
                        min_distance = hit

                for boundary_center in BOUNDARY_OBSTACLES:
                    hit = ray_aabb_intersection(
                        (boat_x, boat_y),
                        direction,
                        boundary_center,
                        BOUNDARY_OBSTACLE_SIZE,
                    )
                    if hit is not None and hit < min_distance:
                        min_distance = hit

            ranges[sector] = float(np.clip(min_distance, 0.0, RANGE_MAX_DISTANCE))

        return ranges

    # ---------------- 传感器回调函数 ----------------
    def gps_callback(self, msg, idx):
        self.agents_data[idx]['lat'] = msg.latitude
        self.agents_data[idx]['lon'] = msg.longitude

    def imu_callback(self, msg, idx):
        q = msg.orientation
        # 提取姿态 (Yaw)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.agents_data[idx]['yaw'] = math.atan2(siny_cosp, cosy_cosp)
        # 提取角速度 (Yaw Rate)
        self.agents_data[idx]['yaw_rate'] = msg.angular_velocity.z
        self.agents_data[idx]['imu_ready'] = True

    # ---------------- 动作与重置机制 ----------------
    def send_action(self, idx, left, right):
        msg_l, msg_r = Float64(), Float64()
        msg_l.data, msg_r.data = float(left), float(right)
        self.pubs_left[idx].publish(msg_l)
        self.pubs_right[idx].publish(msg_r)

    def teleport_model(self, model_name, x, y, z=0.5, wait=False):
        req_content = f'name: "{model_name}", position: {{x: {x}, y: {y}, z: {z}}}, orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}'
        cmd = ["gz", "service", "-s", f"/world/{WORLD_NAME}/set_pose", 
               "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean", 
               "--timeout", "2000", "--req", req_content]
        if wait:
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3.0,
                )
                if result.returncode != 0:
                    self.get_logger().warning(
                        f"Teleport service failed for {model_name} at ({x:.1f}, {y:.1f})."
                    )
                    return False
                return True
            except subprocess.TimeoutExpired:
                self.get_logger().warning(
                    f"Teleport service timed out for {model_name} at ({x:.1f}, {y:.1f})."
                )
                return False

        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True

    def reset_pose_and_obstacles(self):
        """重置多艘船和障碍物，确保安全距离"""
        boat_positions = []
        
        # 1. 随机传送所有船只到安全初始位置
        for i in range(self.num_agents):
            valid = False
            while not valid:
                bx = np.random.uniform(-75.0, 45.0)
                by = np.random.uniform(-100.0, 0.0)
                # 检查船只之间的距离
                if all(math.hypot(bx - p[0], by - p[1]) > MIN_RESET_BOAT_DISTANCE for p in boat_positions):
                    valid = True

            boat_positions.append((bx, by))
            self.teleport_model(f"wamv{i+1}", bx, by, z=0.5, wait=True)
            self.send_action(i, 0.0, 0.0)
            
        # 2. 传送障碍物
        for i in range(NUM_OBSTACLES):
            valid = False
            while not valid:
                obs_x = self.rng.uniform(-75.0, 45.0)
                obs_y = self.rng.uniform(-100.0, 0.0)
                
                # 检查: 是否避开了所有的目标点
                too_close_to_target = False
                for tx, ty in self.targets_xy:
                    if math.hypot(obs_x - tx, obs_y - ty) < 2 * GOAL_THRESHOLD:
                        too_close_to_target = True
                        break
                
                if too_close_to_target:
                    continue 
                
                # 检查: 是否避开了所有的船
                too_close_to_boat = False
                for bx, by in boat_positions:
                    if math.hypot(obs_x - bx, obs_y - by) < 15.0:
                        too_close_to_boat = True
                        break
                
                if too_close_to_boat:
                    continue
                    
                valid = True
                
            self.obstacle_positions[i] = [obs_x, obs_y]
            self.teleport_model(f"rl_obstacle_{self.rank}_{i}", obs_x, obs_y, z=0.5)
        # print(self.obstacle_positions)

        # 3. 固定矩形边界障碍物
        for i, (boundary_x, boundary_y) in enumerate(BOUNDARY_OBSTACLES):
            self.teleport_model(f"rl_boundary_obstacle_{self.rank}_{i}", boundary_x, boundary_y, z=1.0)

        return boat_positions

    def spawn_gazebo_marker(self):
        """在 Gazebo 注入目标区域可视化模型 (修复重合问题)"""
        # 颜色: wamv1(绿), wamv2(蓝), wamv3(黄)
        colors = ['0 1 0 0.4', '0 0 1 0.4', '1 1 0 0.4'] 
        
        for i in range(self.num_agents):
            tx, ty = self.targets_xy[i]
            marker_name = f'rl_target_marker_{self.rank}_{i}'
            color = colors[i % len(colors)]
            
            sdf_xml = f"<?xml version='1.0' ?><sdf version='1.6'><model name='{marker_name}'><static>true</static><pose>0 0 0 0 0 0</pose><link name='link'><visual name='target_zone'><pose>0 0 0.1 0 0 0</pose><geometry><cylinder><radius>{GOAL_THRESHOLD}</radius><length>0.2</length></cylinder></geometry><material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual></link></model></sdf>"
            path = f'/tmp/{marker_name}.sdf'
            
            with open(path, 'w') as f: 
                f.write(sdf_xml)
                
            subprocess.run(['ros2', 'run', 'ros_gz_sim', 'create', '-world', WORLD_NAME, '-name', marker_name, '-remove'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['ros2', 'run', 'ros_gz_sim', 'create', '-world', WORLD_NAME, '-name', marker_name, '-file', path, '-x', str(tx), '-y', str(ty), '-z', '0.0'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            threading.Timer(1.0, lambda name=marker_name, x=tx, y=ty: self.teleport_model(name, x, y, z=0.0)).start()

    def spawn_gazebo_obstacles(self):
        def spawn_single_obstacle(i):
            obs_name = f"rl_obstacle_{self.rank}_{i}"
            sdf_xml = f"<?xml version='1.0' ?><sdf version='1.6'><model name='{obs_name}'><static>true</static><pose>0 0 -10 0 0 0</pose><link name='link'><collision name='collision'><geometry><sphere><radius>{OBSTACLE_RADIUS}</radius></sphere></geometry></collision><visual name='visual'><geometry><sphere><radius>{OBSTACLE_RADIUS}</radius></sphere></geometry><material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse></material></visual></link></model></sdf>"
            path = f'/tmp/{obs_name}.sdf'
            with open(path, 'w') as f: f.write(sdf_xml)
            subprocess.run(['ros2', 'run', 'ros_gz_sim', 'create', '-world', WORLD_NAME, '-name', obs_name, '-remove'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['ros2', 'run', 'ros_gz_sim', 'create', '-world', WORLD_NAME, '-name', obs_name, '-file', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        threads = [threading.Thread(target=spawn_single_obstacle, args=(i,)) for i in range(NUM_OBSTACLES)]
        for t in threads: t.start()
        for t in threads: t.join()

        def spawn_boundary_obstacle(i, x, y):
            obs_name = f"rl_boundary_obstacle_{self.rank}_{i}"
            size_x, size_y, size_z = BOUNDARY_OBSTACLE_SIZE
            sdf_xml = f"<?xml version='1.0' ?><sdf version='1.6'><model name='{obs_name}'><static>true</static><pose>0 0 0 0 0 0</pose><link name='link'><collision name='collision'><geometry><box><size>{size_x} {size_y} {size_z}</size></box></geometry></collision><visual name='visual'><geometry><box><size>{size_x} {size_y} {size_z}</size></box></geometry><material><ambient>0.8 0.2 0.2 1</ambient><diffuse>0.8 0.2 0.2 1</diffuse></material></visual></link></model></sdf>"
            path = f'/tmp/{obs_name}.sdf'
            with open(path, 'w') as f:
                f.write(sdf_xml)
            subprocess.run(['ros2', 'run', 'ros_gz_sim', 'create', '-world', WORLD_NAME, '-name', obs_name, '-remove'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['ros2', 'run', 'ros_gz_sim', 'create', '-world', WORLD_NAME, '-name', obs_name, '-file', path, '-x', str(x), '-y', str(y), '-z', '1.0'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        boundary_threads = [
            threading.Thread(target=spawn_boundary_obstacle, args=(i, x, y))
            for i, (x, y) in enumerate(BOUNDARY_OBSTACLES)
        ]
        for t in boundary_threads: t.start()
        for t in boundary_threads: t.join()

    def publish_rviz_markers(self):
        pass


class VRXMAPPOEnv(gym.Env):
    """符合 on-policy (MAPPO) 规范的多智能体环境封装"""
    def __init__(self, args, rank):
        super().__init__()
        self.num_agents = 2
        self.rank = rank
        self.max_steps = MAX_STEPS
        
        domain_id = 10 + rank
        os.environ['ROS_DOMAIN_ID'] = str(domain_id)
        os.environ['GZ_PARTITION'] = f"vrx_env_{rank}"
        
        if not rclpy.ok():
            rclpy.init()
            
        self.node = WamvMultiAgent(self.num_agents, self.rank)
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()

        # 观测维度: 位置/速度/目标偏差(6) + 几何距离12扇区 + yaw/yaw_rate/距离/角度差(4)
        self.obs_dim = 22
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        
        for _ in range(self.num_agents):
            self.action_space.append(spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32))
            self.observation_space.append(spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32))
            self.share_observation_space.append(spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim * self.num_agents,), dtype=np.float32))

        self.step_count = 0
        self.last_distances = [0.0] * self.num_agents
        self.goal_reached = [False] * self.num_agents
        self.last_surge_cmds = np.zeros(self.num_agents, dtype=np.float32)
        self.last_yaw_cmds = np.zeros(self.num_agents, dtype=np.float32)
        self.last_heading_errors = np.zeros(self.num_agents, dtype=np.float32)
        self.last_left_thrusts = np.zeros(self.num_agents, dtype=np.float32)
        self.last_right_thrusts = np.zeros(self.num_agents, dtype=np.float32)
        print("env_test")

    def _get_obs(self):
        wait_start = time.monotonic()
        while not self.node.check_all_ready():
            if time.monotonic() - wait_start > SENSOR_READY_TIMEOUT:
                raise RuntimeError("Timed out waiting for fresh VRX sensor data.")
            time.sleep(0.001)

        obs_list = []
        distances = []
        current_positions = [
            latlon_to_xy(data['lat'], data['lon'])
            for data in self.node.agents_data
        ]
        
        for i in range(self.num_agents):
            data = self.node.agents_data[i]
            curr_x, curr_y = current_positions[i]
            other_boats = [
                pos for j, pos in enumerate(current_positions)
                if j != i
            ]
            data['range_sectors'] = self.node.compute_geometric_ranges(
                curr_x,
                curr_y,
                other_boats,
            )
            
            # --- 计算线速度 (差分法) ---
            if data['prev_x'] is None or data['prev_y'] is None:
                vx, vy = 0.0, 0.0
            else:
                vx = (curr_x - data['prev_x']) / STEP_DURATION
                vy = (curr_y - data['prev_y']) / STEP_DURATION
                
            # 更新为下一次差分做准备
            data['prev_x'] = curr_x
            data['prev_y'] = curr_y

            target_x, target_y = self.node.targets_xy[i]
            
            # 用于计算奖励和角度差的相对变量 (目标 - 当前)
            dx, dy = target_x - curr_x, target_y - curr_y
            
            # 用于 PINN 物理能量场计算的目标偏差 (当前 - 目标)
            goal_x_offset = curr_x - target_x
            goal_y_offset = curr_y - target_y
            
            distance = math.hypot(dx, dy)
            target_angle = math.atan2(dy, dx)
            angle_diff = wrap_to_pi(target_angle - data['yaw'])
            
            # 获取姿态和角速度
            yaw = data['yaw']
            yaw_rate = data['yaw_rate']
            
            # ================== 核心修改区 ==================
            # 严格按照 SafePinnPPOActorCore 要求的切片顺序拼接
            agent_obs = [
                curr_x, curr_y,               # [0, 1] 坐标 (位置)
                vx, vy,                       # [2, 3] 速度 (动能)
                goal_x_offset, goal_y_offset  # [4, 5] 目标偏差 (引力势能)
            ] 
            
            # [6:18] 12 个几何距离扇区 (障碍物/其他船排斥势能)
            agent_obs += data['range_sectors'].tolist()
            
            # [18:22] 附加状态 (供 std_net 和 Critic 网络学习非线性特征)
            agent_obs += [yaw, yaw_rate, distance, angle_diff]
            # ================================================
            
            obs_list.append(agent_obs)
            distances.append(distance)

        obs_array = np.array(obs_list, dtype=np.float32)
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=100.0, neginf=-100.0)
        return obs_array, distances

    def _get_share_obs(self, obs):
        flat_obs = obs.reshape(-1)
        return np.tile(flat_obs, (self.num_agents, 1))

    def _reset_command_filters(self):
        self.last_surge_cmds.fill(0.0)
        self.last_yaw_cmds.fill(0.0)
        self.last_heading_errors.fill(0.0)
        self.last_left_thrusts.fill(0.0)
        self.last_right_thrusts.fill(0.0)

    def _global_force_to_motor_cmd(self, idx, fx, fy):
        data = self.node.agents_data[idx]
        boat_yaw = data['yaw']
        fx_world = float(fx)
        fy_world = float(fy)

        # 将策略输出的全局期望力投影到船体坐标系。
        # surge 是沿船头方向的分量；sway 是横向分量，欠驱动船用它生成转向命令。
        cos_yaw = math.cos(boat_yaw)
        sin_yaw = math.sin(boat_yaw)
        surge = fx_world * cos_yaw + fy_world * sin_yaw
        sway = -fx_world * sin_yaw + fy_world * cos_yaw

        surge_cmd = surge
        # 左推力更大时船会向右转；期望横向左移时需要右推力更大，所以取 -sway。
        yaw_cmd = -sway
        heading_error = wrap_to_pi(math.atan2(sway, surge))

        left_thrust = (surge_cmd + yaw_cmd) * THRUST_SCALE
        right_thrust = (surge_cmd - yaw_cmd) * THRUST_SCALE

        self.last_surge_cmds[idx] = surge_cmd
        self.last_yaw_cmds[idx] = yaw_cmd
        self.last_heading_errors[idx] = heading_error
        self.last_left_thrusts[idx] = left_thrust
        self.last_right_thrusts[idx] = right_thrust

        return left_thrust, right_thrust

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.goal_reached = [False] * self.num_agents
        self._reset_command_filters()
        
        # Reset 时清空上一个 episode 遗留的位置，防止产生极其离谱的瞬移速度
        for i in range(self.num_agents):
            self.node.agents_data[i]['prev_x'] = None
            self.node.agents_data[i]['prev_y'] = None

        reset_positions = None
        last_status = "reset was not attempted"
        for attempt in range(1, RESET_RETRIES + 1):
            self.node.clear_sensor_cache()
            reset_positions = self.node.reset_pose_and_obstacles()
            self.node.wait_sim_time(RESET_SETTLE_SIM_TIME)
            self.node.clear_sensor_cache()

            if self.node.wait_for_fresh_reset_state(reset_positions):
                break

            last_status = self.node.reset_state_status(reset_positions)
            self.node.get_logger().warning(
                f"Reset attempt {attempt}/{RESET_RETRIES} timed out in env rank {self.rank}: {last_status}"
            )
            for i in range(self.num_agents):
                self.node.send_action(i, 0.0, 0.0)
            self.node.wait_sim_time(RESET_SETTLE_SIM_TIME)
        else:
            raise RuntimeError(
                "Timed out waiting for reset GPS/IMU data near teleported boat positions. "
                f"env_rank={self.rank}; {last_status}"
            )

        obs, self.last_distances = self._get_obs()
        share_obs = self._get_share_obs(obs)

        available_actions = None 
        return obs, share_obs, available_actions

    def step(self, actions):
        actions = np.nan_to_num(actions, nan=0.0, posinf=10.0, neginf=-10.0)
        actions = np.clip(actions, -10.0, 10.0)
        for i in range(self.num_agents):
            fx = actions[i][0]
            fy = actions[i][1]
            left_thrust, right_thrust = self._global_force_to_motor_cmd(i, fx, fy)
            self.node.send_action(i, left_thrust, right_thrust)

        self.node.wait_sim_time(STEP_DURATION)
        self.step_count += 1
        
        obs, current_distances = self._get_obs()
        share_obs = self._get_share_obs(obs)
        
        team_reward = 0.0
        collision_occurred = False
        newly_reached = [False] * self.num_agents
        range_mins = []
        proximity_penalties = []
        goal_stay_rewards = []
        
        for i in range(self.num_agents):
            agent_data = self.node.agents_data[i]
            range_min = float(np.min(agent_data['range_sectors']))
            range_mins.append(range_min)

            reward = 0.0
            currently_in_goal = current_distances[i] < GOAL_THRESHOLD
            reward += (self.last_distances[i] - current_distances[i])

            if currently_in_goal:
                reward += GOAL_STAY_REWARD
                goal_stay_rewards.append(GOAL_STAY_REWARD)
                if not self.goal_reached[i]:
                    reward += GOAL_REWARD
                    self.goal_reached[i] = True
                    newly_reached[i] = True
            else:
                goal_stay_rewards.append(0.0)

            proximity_penalty = 0.0
            if range_min < OBSTACLE_PROXIMITY_DISTANCE:
                danger = (OBSTACLE_PROXIMITY_DISTANCE - range_min) / OBSTACLE_PROXIMITY_DISTANCE
                proximity_penalty = OBSTACLE_PROXIMITY_PENALTY * danger * danger
                reward -= proximity_penalty
            proximity_penalties.append(proximity_penalty)
            
            if range_min < COLLISION_THRESHOLD:
                reward += COLLISION_PENALTY
                collision_occurred = True
                
            team_reward += reward
            self.last_distances[i] = current_distances[i]

        all_currently_in_goal = all(distance < GOAL_THRESHOLD for distance in current_distances)

        shared_reward = team_reward / self.num_agents
        rewards = [[shared_reward] for _ in range(self.num_agents)]
        
        timeout = (self.step_count >= self.max_steps)
        is_done = collision_occurred or timeout
        dones = [is_done] * self.num_agents

        if collision_occurred:
            terminal_reason = "collision"
        elif timeout:
            terminal_reason = "max_steps"
        else:
            terminal_reason = None

        infos = [
            {
                "distance": float(current_distances[i]),
                "range_min": range_mins[i],
                "goal_reached": bool(self.goal_reached[i]),
                "currently_in_goal": bool(current_distances[i] < GOAL_THRESHOLD),
                "all_currently_in_goal": bool(all_currently_in_goal),
                "newly_reached": bool(newly_reached[i]),
                "goal_stay_reward": float(goal_stay_rewards[i]),
                "proximity_penalty": float(proximity_penalties[i]),
                "station_keeping_active": False,
                "policy_active": True,
                "collision": bool(range_mins[i] < COLLISION_THRESHOLD),
                "terminal_reason": terminal_reason,
                "surge_cmd": float(self.last_surge_cmds[i]),
                "yaw_cmd": float(self.last_yaw_cmds[i]),
                "heading_error": float(self.last_heading_errors[i]),
                "left_thrust": float(self.last_left_thrusts[i]),
                "right_thrust": float(self.last_right_thrusts[i]),
            }
            for i in range(self.num_agents)
        ]
            
        available_actions = None
        
        return obs, share_obs, rewards, dones, infos, available_actions

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()
        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1.0)
