import os
import yaml
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable, OpaqueFunction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# ================= 全局配置 =================
WORLD_NAME = 'sydney_regatta'
ROBOTS = [
    {'name': 'wamv1', 'x': '-532', 'y': '166', 'z': '0.5', 'R': '0', 'P': '0', 'Y': '1.0'},
    {'name': 'wamv2', 'x': '-532', 'y': '172', 'z': '0.5', 'R': '0', 'P': '0', 'Y': '1.0'},
    # {'name': 'wamv3', 'x': '-542', 'y': '162', 'z': '0.5', 'R': '0', 'P': '0', 'Y': '1.0'},
]

def generate_launch_description():
    vrx_gz_share = get_package_share_directory('vrx_gz')
    world_path = os.path.join(vrx_gz_share, 'worlds', f'{WORLD_NAME}.sdf')

    vrx_prefix = os.path.dirname(os.path.dirname(vrx_gz_share))
    model_path = os.path.join(vrx_prefix, 'share', 'vrx_gz', 'models')
    new_res_path = f"{model_path}:{os.environ.get('GZ_SIM_RESOURCE_PATH', '')}"

    ld = LaunchDescription([
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', new_res_path)
    ])

    ld.add_action(ExecuteProcess(
        cmd=['gz', 'sim', '-s', '-r','-z', '100000', '--headless-render', world_path],
        output='screen'
    ))

    ld.add_action(OpaqueFunction(function=launch_bridge_node))

    spawn_launch = os.path.join(vrx_gz_share, 'launch', 'spawn.launch.py')
    wamv_config = os.path.join(vrx_gz_share, 'config', 'wamv_config.yaml')

    for robot in ROBOTS:
        ld.add_action(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(spawn_launch),
            launch_arguments={
                'world': WORLD_NAME,
                'sim_mode': 'sim',
                'bridge_competition_topics': 'false',
                'name': robot['name'],
                'x': robot['x'], 'y': robot['y'], 'z': robot['z'],
                'R': robot['R'], 'P': robot['P'], 'Y': robot['Y'],
                'spawn_config_file': wamv_config,
            }.items()
        ))

    return ld

def launch_bridge_node(context, *args, **kwargs):
    # 【核心修复】QoS 必须为 Best Effort 以匹配 Gazebo 传感器
    QOS_CONFIG = {
        'reliability': 'best_effort',
        'durability': 'volatile',
        'history': 'keep_last',
        'depth': 1
    }
    
    bridge_config = []
    
    bridge_config.append({'topic_name': '/clock', 'ros_type_name': 'rosgraph_msgs/msg/Clock', 'gz_type_name': 'gz.msgs.Clock', 'direction': 'GZ_TO_ROS'})

    for robot in ROBOTS:
        name = robot['name']
        
        # 1. GPS (NavSatFix)
        bridge_config.append({
            'gz_topic_name': f'/world/{WORLD_NAME}/model/{name}/link/{name}/gps_wamv_link/sensor/navsat/navsat',
            'ros_topic_name': f'/{name}/sensors/gps/gps/fix',
            'ros_type_name': 'sensor_msgs/msg/NavSatFix',
            'gz_type_name': 'gz.msgs.NavSat',
            'direction': 'GZ_TO_ROS',
            'qos': QOS_CONFIG # <--- 必须加上 QoS
        })

        # 2. IMU
        bridge_config.append({
            'gz_topic_name': f'/world/{WORLD_NAME}/model/{name}/link/{name}/imu_wamv_link/sensor/imu_wamv_sensor/imu',
            'ros_topic_name': f'/{name}/sensors/imu/imu/data',
            'ros_type_name': 'sensor_msgs/msg/Imu',
            'gz_type_name': 'gz.msgs.IMU',
            'direction': 'GZ_TO_ROS',
            'qos': QOS_CONFIG
        })
        
        # 3. Thrusters (Control)
        for side in ['left', 'right']:
            bridge_config.append({
                'gz_topic_name': f'/{name}/thrusters/{side}/thrust',
                'ros_topic_name': f'/{name}/thrusters/{side}/thrust',
                'ros_type_name': 'std_msgs/msg/Float64',
                'gz_type_name': 'gz.msgs.Double',
                'direction': 'ROS_TO_GZ'
            })
            
        # 4. Teleport (Set Pose)
        bridge_config.append({
            'gz_topic_name': f'/model/{name}/set_pose',
            'ros_topic_name': f'/{name}/set_pose',
            'ros_type_name': 'geometry_msgs/msg/Pose',
            'gz_type_name': 'gz.msgs.Pose',
            'direction': 'ROS_TO_GZ'
        })

    env_id = os.environ.get('GZ_PARTITION', '')
    yaml_suffix = env_id if env_id else 'default'
    yaml_path = f'/tmp/vrx_bridge_final_{yaml_suffix}.yaml'
    
    with open(yaml_path, 'w') as f:
        yaml.dump(bridge_config, f)

    node_env = {}
    if env_id:
        node_env['GZ_PARTITION'] = env_id

    return [Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': yaml_path}],
        output='screen',
        additional_env=node_env
    )]
