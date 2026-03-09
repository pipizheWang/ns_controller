#!/usr/bin/env python3

"""
气动扰动力估计节点
用于双机编队的气动相互作用力估计
基于训练好的神经网络模型预测垂直方向的气动扰动力
"""

import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
from rclpy.qos import QoSProfile, ReliabilityPolicy
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class PhiNet(nn.Module):
    """特征编码网络（输入6维：相对位置+相对速度）"""
    def __init__(self, inputdim=6, hiddendim=20):
        super(PhiNet, self).__init__()
        self.fc1 = nn.Linear(inputdim, 25)
        self.fc2 = nn.Linear(25, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, hiddendim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class RhoNet(nn.Module):
    """力解码网络（输出1维：Faz，单位克力）"""
    def __init__(self, hiddendim=20):
        super(RhoNet, self).__init__()
        self.fc1 = nn.Linear(hiddendim, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class AeroForceEstimator(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("Node running: %s" % name)
        
        # 控制频率（与控制器保持一致）
        self.estimation_rate = 50.0
        
        # 物理常量
        self.gravity = 9.81  # m/s^2
        
        # UAV0 和 UAV1 的状态
        self.uav0_pose = None
        self.uav0_velo = None
        self.uav1_pose = None
        self.uav1_velo = None
        
        # 加载神经网络模型
        self.load_neural_networks()
        
        # 设置订阅和发布
        self.setup_ros_interface()
        
        # 创建定时器
        self.estimation_timer = self.create_timer(
            1.0 / self.estimation_rate, 
            self.estimation_callback
        )

        # 调试用计数器：每 50 次回调（约 1 秒）打印一次诊断信息
        self._dbg_counter = 0

        self.get_logger().info("Aerodynamic force estimator initialized")
    
    def load_neural_networks(self):
        """加载训练好的神经网络模型"""
        # 沿父目录链向上查找 px4_ws，再定位到 src/ns_controller/train/model
        # （兼容 colcon build 安装后 __file__ 指向 site-packages 的情况）
        def resolve_model_dir():
            current = Path(__file__).resolve()
            for parent in current.parents:
                if parent.name == 'px4_ws':
                    candidate = parent / 'src' / 'ns_controller' / 'train' / 'model'
                    return candidate
            # 回退：相对 __file__ 推算（仅在直接从源目录运行时有效）
            return Path(__file__).parent.parent / 'train' / 'model'

        model_dir = resolve_model_dir()
        self.get_logger().info(f"Model directory resolved to: {model_dir}")

        if not model_dir.exists():
            self.get_logger().warn(f"Model directory not found: {model_dir}")
            self.get_logger().warn("Using untrained networks (for testing only)")

        # 初始化网络（hidden_dim=20，inputdim=6）
        self.phi_net = PhiNet(inputdim=6, hiddendim=20)
        self.rho_net = RhoNet(hiddendim=20)

        # 尝试加载预训练模型
        try:
            phi_path = model_dir / 'phi_net.pth'
            rho_path = model_dir / 'rho_net.pth'

            if phi_path.exists():
                self.phi_net.load_state_dict(
                    torch.load(phi_path, map_location='cpu', weights_only=True))
                self.get_logger().info(f"Loaded phi_net model from {phi_path}")
            else:
                self.get_logger().warn(f"phi_net.pth not found at {phi_path}")

            if rho_path.exists():
                self.rho_net.load_state_dict(
                    torch.load(rho_path, map_location='cpu', weights_only=True))
                self.get_logger().info(f"Loaded rho_net model from {rho_path}")
            else:
                self.get_logger().warn(f"rho_net.pth not found at {rho_path}")

        except Exception as e:
            self.get_logger().error(f"Failed to load models: {e}")
            self.get_logger().warn("Proceeding with untrained networks")

        # 设置为评估模式，使用单精度（与训练时一致）
        self.phi_net.eval()
        self.rho_net.eval()
        self.phi_net.float()
        self.rho_net.float()
    
    def setup_ros_interface(self):
        """设置ROS接口：订阅和发布"""
        qos_best_effort = QoSProfile(
            depth=10, 
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        qos_reliable = QoSProfile(
            depth=10, 
            reliability=ReliabilityPolicy.RELIABLE
        )
        
        # 订阅 UAV0 的状态
        self.uav0_pose_sub = self.create_subscription(
            PoseStamped,
            '/uav0/local_position/pose',
            self.uav0_pose_callback,
            qos_best_effort
        )
        
        self.uav0_velo_sub = self.create_subscription(
            TwistStamped,
            '/uav0/local_position/velocity_local',
            self.uav0_velo_callback,
            qos_best_effort
        )
        
        # 订阅 UAV1 的状态
        self.uav1_pose_sub = self.create_subscription(
            PoseStamped,
            '/uav1/local_position/pose',
            self.uav1_pose_callback,
            qos_best_effort
        )
        
        self.uav1_velo_sub = self.create_subscription(
            TwistStamped,
            '/uav1/local_position/velocity_local',
            self.uav1_velo_callback,
            qos_best_effort
        )
        
        # 发布估计的气动力（仅 UAV0，UAV1 为上层飞机不受扰动估计）
        self.uav0_aero_force_pub = self.create_publisher(
            Vector3Stamped,
            '/uav0/estimated_aero_force',
            qos_reliable
        )
    
    def uav0_pose_callback(self, msg):
        """UAV0 位置回调"""
        self.uav0_pose = msg
    
    def uav0_velo_callback(self, msg):
        """UAV0 速度回调"""
        self.uav0_velo = msg
    
    def uav1_pose_callback(self, msg):
        """UAV1 位置回调"""
        self.uav1_pose = msg
    
    def uav1_velo_callback(self, msg):
        """UAV1 速度回调"""
        self.uav1_velo = msg
    
    def construct_network_input(self, lower_pose, lower_velo, upper_pose, upper_velo):
        """
        构造神经网络输入（6维）

        输入维度：6
        [0:3] - 上层飞机相对下层飞机的位置 (upper_pos - lower_pos)
        [3:6] - 上层飞机相对下层飞机的速度 (upper_vel - lower_vel)

        参数:
            lower_pose/lower_velo: 下层飞机（受扰动，即 UAV0）的位姿和速度
            upper_pose/upper_velo: 上层飞机（产生扰动，即 UAV1）的位姿和速度
        """
        lower_pos = np.array([
            lower_pose.pose.position.x,
            lower_pose.pose.position.y,
            lower_pose.pose.position.z
        ], dtype=np.float32)

        lower_vel = np.array([
            lower_velo.twist.linear.x,
            lower_velo.twist.linear.y,
            lower_velo.twist.linear.z
        ], dtype=np.float32)

        upper_pos = np.array([
            upper_pose.pose.position.x,
            upper_pose.pose.position.y,
            upper_pose.pose.position.z
        ], dtype=np.float32)

        upper_vel = np.array([
            upper_velo.twist.linear.x,
            upper_velo.twist.linear.y,
            upper_velo.twist.linear.z
        ], dtype=np.float32)

        # 修正初始位置偏移，与日志记录保持一致：
        # UAV0（下层/lower）的坐标原点偏移 (+2, +2, 0)，需减去
        # UAV1（上层/upper）的坐标原点偏移 (-2, -2, 0)，需加上
        lower_pos_corrected = lower_pos - np.array([2.0, 2.0, 0.0], dtype=np.float32)
        upper_pos_corrected = upper_pos + np.array([2.0, 2.0, 0.0], dtype=np.float32)

        # 6维输入向量：[相对位置(3), 相对速度(3)]
        network_input = np.empty(6, dtype=np.float32)
        network_input[0:3] = upper_pos_corrected - lower_pos_corrected
        network_input[3:6] = upper_vel - lower_vel

        # ── 调试信息（随 _dbg_counter 控制频率） ──
        self._dbg_print_data = {
            'lower_pos_raw': lower_pos.copy(),
            'upper_pos_raw': upper_pos.copy(),
            'lower_pos_corr': lower_pos_corrected.copy(),
            'upper_pos_corr': upper_pos_corrected.copy(),
            'lower_vel': lower_vel.copy(),
            'upper_vel': upper_vel.copy(),
            'network_input': network_input.copy(),
        }

        return network_input
    
    def estimate_aero_force(self, network_input):
        """
        使用神经网络估计下层飞机受到的气动力

        参数:
            network_input: 6维 numpy 数组 [相对位置(3), 相对速度(3)]，float32
        返回:
            Fa_z: 垂直方向的气动力（牛顿）
        """
        # 转换为 float32 torch tensor，增加 batch 维度
        inputs = torch.from_numpy(network_input).float().unsqueeze(0)

        with torch.no_grad():
            # 单链路推理：phi_net → rho_net
            features = self.phi_net(inputs)       # (1, hiddendim)
            Fa_gram = self.rho_net(features)      # (1, 1)，单位：克力

            # 克力转换为牛顿
            Fa_z_newton = Fa_gram[0, 0].item() * self.gravity / 1000.0

        return Fa_z_newton
    
    def estimation_callback(self):
        """估计回调函数（仅估计 UAV0 受到的气动力）"""
        # 检查是否收到所有必要的数据
        if (self.uav0_pose is None or self.uav0_velo is None or
                self.uav1_pose is None or self.uav1_velo is None):
            # self.get_logger().warn("Waiting for all UAV state data...")
            return

        try:
            # 估计 UAV0（下层飞机）受到的气动力（来自上层 UAV1 的影响）
            # 输入：upper(UAV1) - lower(UAV0) 的相对位置和速度
            network_input = self.construct_network_input(
                self.uav0_pose, self.uav0_velo,
                self.uav1_pose, self.uav1_velo
            )
            Fa_z_uav0 = self.estimate_aero_force(network_input)

            # ── 每 50 次回调打印一次诊断信息（约 1 秒一次） ──
            self._dbg_counter += 1
            if self._dbg_counter % 50 == 1:
                d = self._dbg_print_data
                self.get_logger().info(
                    f"[DBG #{self._dbg_counter}]\n"
                    f"  UAV0 raw pos  : x={d['lower_pos_raw'][0]:.4f}  y={d['lower_pos_raw'][1]:.4f}  z={d['lower_pos_raw'][2]:.4f}\n"
                    f"  UAV1 raw pos  : x={d['upper_pos_raw'][0]:.4f}  y={d['upper_pos_raw'][1]:.4f}  z={d['upper_pos_raw'][2]:.4f}\n"
                    f"  UAV0 corr pos : x={d['lower_pos_corr'][0]:.4f}  y={d['lower_pos_corr'][1]:.4f}  z={d['lower_pos_corr'][2]:.4f}\n"
                    f"  UAV1 corr pos : x={d['upper_pos_corr'][0]:.4f}  y={d['upper_pos_corr'][1]:.4f}  z={d['upper_pos_corr'][2]:.4f}\n"
                    f"  UAV0 vel      : vx={d['lower_vel'][0]:.4f}  vy={d['lower_vel'][1]:.4f}  vz={d['lower_vel'][2]:.4f}\n"
                    f"  UAV1 vel      : vx={d['upper_vel'][0]:.4f}  vy={d['upper_vel'][1]:.4f}  vz={d['upper_vel'][2]:.4f}\n"
                    f"  NN input [rel_pos|rel_vel]: "
                    f"[{d['network_input'][0]:.4f}, {d['network_input'][1]:.4f}, {d['network_input'][2]:.4f} | "
                    f"{d['network_input'][3]:.4f}, {d['network_input'][4]:.4f}, {d['network_input'][5]:.4f}]\n"
                    f"  --> Fa_z = {Fa_z_uav0:.6f} N  ({Fa_z_uav0*1000/self.gravity:.4f} gf)"
                )

            # 发布 UAV0 的气动力估计
            force_msg = Vector3Stamped()
            force_msg.header.stamp = self.get_clock().now().to_msg()
            force_msg.header.frame_id = "world"
            force_msg.vector.x = 0.0  # x 方向暂时为 0
            force_msg.vector.y = 0.0  # y 方向暂时为 0
            force_msg.vector.z = Fa_z_uav0  # z 方向气动力（牛顿）
            self.uav0_aero_force_pub.publish(force_msg)

            # DEBUG 级别打印调试信息
            if self.get_logger().get_effective_level() <= rclpy.logging.LoggingSeverity.DEBUG:
                rel_dist = np.linalg.norm(network_input[0:3])
                self.get_logger().debug(
                    f"UAV0: z={self.uav0_pose.pose.position.z:.2f}m, Fa_z={Fa_z_uav0:.3f}N | "
                    f"Rel_dist={rel_dist:.2f}m"
                )

        except Exception as e:
            self.get_logger().error(f"Error in estimation: {e}")


def main(args=None):
    node = None
    try:
        rclpy.init(args=args)
        node = AeroForceEstimator("f_est")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
