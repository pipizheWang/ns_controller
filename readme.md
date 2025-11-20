# 终端1：启动飞机1 UAV0
cd ~/PX4-Autopilot
conda activate torch
source /opt/ros/jazzy/setup.bash
export PX4_GZ_MODEL=x500
export PX4_SYS_AUTOSTART=4001

export PX4_GZ_MODEL_POSE="0,0,1.0,0,0,0"
./build/px4_sitl_default/bin/px4 -i 0

export PX4_GZ_MODEL_POSE="0,0,2.0,0,0,0"
./build/px4_sitl_default/bin/px4 -i 1

# 启动地面站

# 终端2：mavros通信
conda activate torch
source /opt/ros/jazzy/setup.bash

## 启动 UAV0 的 MAVROS
ros2 run mavros mavros_node --ros-args -r __ns:=/uav0 \
    -p fcu_url:=udp://:14540@127.0.0.1:14557 \
    -p target_system_id:=1 \
    --params-file ~/mavros_config.yaml

## 启动 UAV1 的 MAVROS
ros2 run mavros mavros_node --ros-args -r __ns:=/uav1 \
    -p fcu_url:=udp://:14541@127.0.0.1:14559 \
    -p target_system_id:=2 \
    --params-file ~/mavros_config.yaml


# 终端3：OFF-board模式和解锁电机
conda activate torch
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 service call /uav0/mavros/set_mode mavros_msgs/srv/SetMode "{custom_mode: 'OFFBOARD'}"
ros2 service call /uav1/mavros/set_mode mavros_msgs/srv/SetMode "{custom_mode: 'OFFBOARD'}"

ros2 run nl_controller traj_controller_NL0
ros2 run nl_controller traj_controller_NL1

ros2 param set /traj_controller_uav0 traj_mode true
ros2 param set /traj_controller_uav1 traj_mode true

ros2 service call /uav0/mavros/cmd/arming mavros_msgs/srv/CommandBool "{value: true}"
ros2 service call /uav1/mavros/cmd/arming mavros_msgs/srv/CommandBool "{value: true}"



colcon build --packages-select ns_controller


# 其他
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

从加速度——油门的映射：thrust = -0.0015 * a² + 0.0764 * a + 0.1237
悬停：PWM=770

卡号：sertmm2@stu.xidian.edu.cn
密码：4G399y5DKPAg