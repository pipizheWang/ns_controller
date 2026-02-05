# NS Controller 操作指南


## 终端 0：启动 PX4 仿真

### 启动自研下洗力模块
conda deactivate
cd ~/gz_downwash_plugin
export GZ_SIM_RESOURCE_PATH=~/PX4-Autopilot/Tools/simulation/gz/models:$GZ_SIM_RESOURCE_PATH
export GZ_SIM_SYSTEM_PLUGIN_PATH=$PWD/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
gz sim -v 4 -r "$PWD/worlds/two_drones_downwash.sdf"

conda deactivate
cd ~/PX4-Autopilot
source build/px4_sitl_default/rootfs/gz_env.sh
PX4_SYS_AUTOSTART=4001 PX4_GZ_WORLD=x500_downwash_world PX4_SIM_MODEL=gz_x500_nolegcol PX4_GZ_MODEL_NAME=x500_0 ./build/px4_sitl_default/bin/px4 -i 0

conda deactivate
cd ~/PX4-Autopilot
source build/px4_sitl_default/rootfs/gz_env.sh
PX4_SYS_AUTOSTART=4001 PX4_GZ_WORLD=x500_downwash_world PX4_SIM_MODEL=gz_x500_nolegcol PX4_GZ_MODEL_NAME=x500_1 ./build/px4_sitl_default/bin/px4 -i 1

gz sim -r "$PWD/worlds/two_drones_downwash.sdf"
## 终端 1：启动 PX4 仿真

### 环境准备
```bash
cd ~/PX4-Autopilot
conda activate torch
source /opt/ros/jazzy/setup.bash
export PX4_GZ_MODEL=x500
export PX4_SYS_AUTOSTART=4001
```

### 启动 UAV0
```bash
export PX4_GZ_MODEL_POSE="-2,0,5.0,0,0,0"
./build/px4_sitl_default/bin/px4 -i 0
```

### 启动 UAV1
```bash
export PX4_GZ_MODEL_POSE="2,0,5.5,0,0,0"
./build/px4_sitl_default/bin/px4 -i 1
```

---

## 终端 2：MAVROS 通信

### 环境准备
```bash
conda activate torch
source /opt/ros/jazzy/setup.bash
```

### 启动 UAV0 的 MAVROS
```bash
ros2 run mavros mavros_node --ros-args -r __ns:=/uav0 \
    -p fcu_url:=udp://:14540@127.0.0.1:14557 \
    -p target_system_id:=1 \
    --params-file ~/mavros_config0.yaml
```

### 启动 UAV1 的 MAVROS
```bash
ros2 run mavros mavros_node --ros-args -r __ns:=/uav1 \
    -p fcu_url:=udp://:14541@127.0.0.1:14559 \
    -p target_system_id:=2 \
    --params-file ~/mavros_config1.yaml
```

---

## 终端 3：控制器启动与配置

### 环境准备
```bash
conda activate torch
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

### 编译包
```bash
colcon build --packages-select ns_controller
```

### 设置 OFFBOARD 模式
```bash
ros2 service call /uav0/set_mode mavros_msgs/srv/SetMode "{custom_mode: 'OFFBOARD'}"
ros2 service call /uav1/set_mode mavros_msgs/srv/SetMode "{custom_mode: 'OFFBOARD'}"
```

### 启动控制器（选择其一）

#### 非线性控制器 (NL)
```bash
ros2 run ns_controller traj_controller_NL1
ros2 run ns_controller traj_controller_NL0
```

#### 神经符号控制器 (NS)
```bash
ros2 run ns_controller f_est
ros2 run ns_controller traj_controller_NS1
ros2 run ns_controller traj_controller_NS0
```

### 解锁电机
```bash
ros2 service call /uav1/cmd/arming mavros_msgs/srv/CommandBool "{value: true}"
ros2 service call /uav0/cmd/arming mavros_msgs/srv/CommandBool "{value: true}"
```

### 启动轨迹同步

#### NS 模式
```bash
ros2 param set /traj_controller_NL0 traj_mode true
ros2 param set /traj_controller_NL1 traj_mode true

ros2 run ns_controller traj_sync --ros-args -p controller_mode:=NS
```

#### NL 模式
```bash
ros2 param set /traj_controller_NS0 traj_mode true
ros2 param set /traj_controller_NS1 traj_mode true

ros2 run ns_controller traj_sync --ros-args -p controller_mode:=NL
```

### 控制轨迹同步开关

#### 开启同步
```bash
ros2 service call /sync_traj_mode std_srvs/srv/SetBool "{data: true}"
```

#### 关闭同步
```bash
ros2 service call /sync_traj_mode std_srvs/srv/SetBool "{data: false}"
```

---

## 其他配置

### 代理设置
```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
```

### 系统参数

**加速度到油门的映射公式：**
```
thrust = -0.0015 * a² + 0.0764 * a + 0.1237
```

**悬停参数：**
- PWM = 770

---

## 账号信息

**邮箱：** sertmm2@stu.xidian.edu.cn  
**密码：** 4G399y5DKPAg

> ⚠️ **注意：** 建议将敏感信息（如账号密码）从公开仓库中移除，使用环境变量或配置文件管理。