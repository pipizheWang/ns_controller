#!/usr/bin/env python3

# 控制姿态+扰动力补偿节点，SR中NS控制器（加入气动扰动力估计补偿）
import rclpy
import numpy as np
from numpy.linalg import norm
from rclpy.node import Node
from mavros_msgs.msg import AttitudeTarget
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
from scipy.spatial.transform import Rotation as R
from rclpy.qos import QoSProfile, ReliabilityPolicy
from .traj import TargetTraj
from rclpy.clock import Clock
import csv
from pathlib import Path
from datetime import datetime


class TrajController(Node):
    def __init__(self, name):
        # 初始化
        super().__init__(name)
        self.get_logger().info("Node running: %s" % name)

        # 控制频率
        self.control_rate = 50.0
        # UAV0 使用 FLAG=0（z=1.0），与 UAV1 水平轨迹一致
        self.traj = TargetTraj(FLAG=0)

        # 初始化时钟
        self.clock = Clock()

        # 初始化状态变量
        self.current_pa = None
        self.current_velo = None

        # 气动力估计（仅z轴方向）默认0，单位 N，世界坐标系向上为正
        self.aero_force_z = 0.0

        # 轨迹计时器
        self.traj_t = -1.0
        self.t_0 = self.clock.now()

        # 订阅和发布
        qos_best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # 订阅 UAV0 状态
        self.pa_sub_ = self.create_subscription(
            PoseStamped, '/uav0/local_position/pose', self.pa_cb, qos_best_effort)
        self.velo_sub_ = self.create_subscription(
            TwistStamped, '/uav0/local_position/velocity_local', self.velo_cb, qos_best_effort)

        # 订阅 UAV0 气动扰动力估计
        self.aero_force_sub_ = self.create_subscription(
            Vector3Stamped, '/uav0/estimated_aero_force', self.aero_force_cb, qos_best_effort)

        # 发布控制指令
        self.controller_pub_ = self.create_publisher(
            AttitudeTarget, '/uav0/setpoint_raw/attitude', qos_reliable)

        self.controller_timer_ = self.create_timer(1 / self.control_rate, self.controller_cb)

        # 初始化参数
        self.declare_parameter('sliding_gain', [0.3, 0.3, 0.5])  # 滑模跟踪增益
        self.declare_parameter('tracking_gain', [3.0, 3.0, 5.0])  # 跟踪增益
        self.declare_parameter('traj_mode', False)  # 轨迹模式开关
        self.declare_parameter('mass', 2.0)  # 质量 kg

        # 系统常量
        self.gravity = 9.8
        self.thrust_efficiency = 0.74

        # 初始化日志记录
        self.setup_flight_log()

    def setup_flight_log(self):

        def resolve_source_log_dir():
            current = Path(__file__).resolve()
            for parent in current.parents:
                if parent.name == 'px4_ws':
                    candidate = parent / 'src' / 'ns_controller' / 'log'
                    return candidate
            return Path.cwd() / 'log'

        log_dir = resolve_source_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
        log_filename = f'{timestamp}_0.csv'
        self.log_file_path = log_dir / log_filename

        self.log_file = open(self.log_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)

        header = [
            'timestamp',
            'x', 'y', 'z',
            'vx', 'vy', 'vz',
            'roll', 'pitch', 'yaw',
            'x_des', 'y_des', 'z_des',
            'Fa_z', 'aero_comp_z'
        ]
        self.csv_writer.writerow(header)
        self.log_file.flush()

        self.get_logger().info(f"Flight log initialized: {self.log_file_path}")

    def log_flight_data(self, pose, velo, rotation_matrix, traj_p, aero_comp_z):
        r = R.from_matrix(rotation_matrix)
        euler_angles = r.as_euler('xyz', degrees=True)
        roll, pitch, yaw = euler_angles
        current_time = self.get_clock().now().nanoseconds * 1e-9
        data_row = [
            current_time,
            pose[0, 0], pose[1, 0], pose[2, 0],
            velo[0, 0], velo[1, 0], velo[2, 0],
            roll, pitch, yaw,
            traj_p[0, 0], traj_p[1, 0], traj_p[2, 0],
            self.aero_force_z,
            aero_comp_z
        ]
        self.csv_writer.writerow(data_row)
        self.log_file.flush()

    def destroy_node(self):
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
            self.get_logger().info(f"Flight log saved to: {self.log_file_path}")
        super().destroy_node()

    def pa_cb(self, msg):
        self.current_pa = msg

    def velo_cb(self, msg):
        self.current_velo = msg

    def aero_force_cb(self, msg: Vector3Stamped):
        # 假设消息z分量为世界坐标系向上为正的气动扰动力(N)
        self.aero_force_z = float(msg.vector.z)

    def update_trajectory_time(self, traj_mode):
        if traj_mode and self.traj_t == -1.0:
            self.t_0 = self.clock.now()
            self.traj_t = 0.0
            self.get_logger().info("Starting trajectory tracking")
        elif traj_mode:
            self.traj_t = (self.clock.now() - self.t_0).nanoseconds * 1e-9
        else:
            self.traj_t = -1.0
            self.t_0 = self.clock.now()

    def get_current_state(self):
        pose = np.array([
            [self.current_pa.pose.position.x],
            [self.current_pa.pose.position.y],
            [self.current_pa.pose.position.z]
        ])
        velo = np.array([
            [self.current_velo.twist.linear.x],
            [self.current_velo.twist.linear.y],
            [self.current_velo.twist.linear.z]
        ])
        quaternion = [
            self.current_pa.pose.orientation.x,
            self.current_pa.pose.orientation.y,
            self.current_pa.pose.orientation.z,
            self.current_pa.pose.orientation.w
        ]
        r = R.from_quat(quaternion)
        rotation_matrix = r.as_matrix()
        body_z = np.dot(rotation_matrix, np.array([[0], [0], [1]]))
        return pose, velo, rotation_matrix, body_z

    def calculate_desired_force(self, pose, velo, traj_p, traj_v, traj_a, sliding_gain, tracking_gain, mass):
        # 计算加速度控制（原始逻辑）
        s = (velo - traj_v + sliding_gain * (pose - traj_p))
        a_r = traj_a - sliding_gain * (velo - traj_v) + np.array([[0], [0], [self.gravity]])
        F_sp = a_r - tracking_gain * s  # 这里仍是“期望加速度”形式

        # 扰动力补偿：将估计的气动力转换为加速度并叠加
        # 假设 aero_force_z 为正表示向上额外升力，补偿应增加期望加速度以抵消扰动不利影响
        aero_comp_z = self.aero_force_z / mass  # 加速度 (m/s^2)
        F_sp[2, 0] += aero_comp_z

        # 限制输出范围
        F_sp = np.clip(F_sp, np.array([[-5.0], [-5.0], [0.0]]), np.array([[5.0], [5.0], [19.6]]))
        return F_sp, aero_comp_z

    def calculate_attitude_from_force(self, F_sp, body_z, yaw_sp):
        thrust = float(np.dot(F_sp.T, body_z))
        attitude_target = AttitudeTarget()
        normalized_thrust = -0.0015 * thrust * thrust + 0.0764 * thrust + 0.1237
        attitude_target.thrust = np.clip(normalized_thrust, 0.0, 1.0)

        body_z_sp = F_sp / norm(F_sp)
        x_C = np.array([[np.cos(yaw_sp)], [np.sin(yaw_sp)], [0]])
        body_y_sp = np.cross(body_z_sp.flatten(), x_C.flatten()).reshape(3, 1)
        body_y_sp = body_y_sp / norm(body_y_sp)
        body_x_sp = np.cross(body_y_sp.flatten(), body_z_sp.flatten()).reshape(3, 1)
        RM_sp = np.hstack([body_x_sp, body_y_sp, body_z_sp])
        r_sp = R.from_matrix(RM_sp)
        quaternion_sp = r_sp.as_quat()
        attitude_target.orientation.x = quaternion_sp[0]
        attitude_target.orientation.y = quaternion_sp[1]
        attitude_target.orientation.z = quaternion_sp[2]
        attitude_target.orientation.w = quaternion_sp[3]
        attitude_target.type_mask = int(7)
        return attitude_target

    def controller_cb(self):
        if self.current_pa is None or self.current_velo is None:
            self.get_logger().warn("Waiting for pose and velocity data...")
            return

        sliding_gain = np.array(self.get_parameter('sliding_gain').value).reshape(3, 1)
        tracking_gain = np.array(self.get_parameter('tracking_gain').value).reshape(3, 1)
        traj_mode = self.get_parameter('traj_mode').value
        mass = float(self.get_parameter('mass').value)

        self.update_trajectory_time(traj_mode)
        traj_p = self.traj.pose(self.traj_t)
        traj_v = self.traj.velo(self.traj_t)
        traj_a = self.traj.acce(self.traj_t)
        traj_yaw = self.traj.yaw(self.traj_t)

        pose, velo, rotation_matrix, body_z = self.get_current_state()
        F_sp, aero_comp_z = self.calculate_desired_force(
            pose, velo, traj_p, traj_v, traj_a, sliding_gain, tracking_gain, mass)

        attitude_target = self.calculate_attitude_from_force(F_sp, body_z, traj_yaw)
        self.controller_pub_.publish(attitude_target)

        # 记录日志
        self.log_flight_data(pose, velo, rotation_matrix, traj_p, aero_comp_z)

        if self.get_logger().get_effective_level() <= rclpy.logging.LoggingSeverity.DEBUG:
            position_error = norm(pose - traj_p)
            self.get_logger().debug(
                f"Traj time: {self.traj_t:.2f}, Thrust: {attitude_target.thrust:.3f}, PosErr: {position_error:.3f}m, Fa_z={self.aero_force_z:.3f}N, a_comp={aero_comp_z:.3f}m/s^2")


def main(args=None):
    node = None
    try:
        rclpy.init(args=args)
        node = TrajController("traj_controller_NS0")
        rclpy.spin(node)
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
