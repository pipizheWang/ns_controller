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
    """特征编码网络"""
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
    """力解码网络"""
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
        
        # 场景编码：LL2L = 11（两架大飞机）
        self.scene_encoder = {
            'Ge2L': 0, 'Ge2S': 1, 'L2L': 2, 'S2S': 3, 
            'L2S': 4, 'S2L': 5, 'SS2L': 6, 'SL2L': 7,
            'LL2S': 8, 'SL2S': 9, 'SS2S': 10, 'LL2L': 11
        }
        self.scene_type = self.scene_encoder['LL2L']  # 两架大飞机场景
        
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
        
        self.get_logger().info("Aerodynamic force estimator initialized")
    
    def load_neural_networks(self):
        """加载训练好的神经网络模型"""
        # 根据用户提供的相对路径：ns_controller/phi_G.pth 等
        # 这里直接使用当前文件所在目录作为模型目录
        model_dir = Path(__file__).parent
        # 说明：假设权重文件与本脚本同目录（包根）且文件名为 phi_G.pth, phi_L.pth, rho_L.pth
        # 如果文件不存在则保持未训练网络，仅用于测试
        if not model_dir.exists():
            self.get_logger().warn(f"Model directory not found: {model_dir}")
            self.get_logger().warn("Using untrained networks (for testing only)")
        
        # 初始化网络（hidden_dim=20，根据训练代码）
        self.phi_G_net = PhiNet(inputdim=4, hiddendim=20)  # 地面效应
        self.phi_L_net = PhiNet(inputdim=6, hiddendim=20)  # 大飞机特征
        self.rho_L_net = RhoNet(hiddendim=20)              # 大飞机解码器
        
        # 尝试加载预训练模型
        try:
            phi_G_path = model_dir / 'phi_G.pth'
            phi_L_path = model_dir / 'phi_L.pth'
            rho_L_path = model_dir / 'rho_L.pth'
            
            if phi_G_path.exists():
                self.phi_G_net.load_state_dict(torch.load(phi_G_path, map_location='cpu'))
                self.get_logger().info(f"Loaded phi_G model from {phi_G_path}")
            
            if phi_L_path.exists():
                self.phi_L_net.load_state_dict(torch.load(phi_L_path, map_location='cpu'))
                self.get_logger().info(f"Loaded phi_L model from {phi_L_path}")
            
            if rho_L_path.exists():
                self.rho_L_net.load_state_dict(torch.load(rho_L_path, map_location='cpu'))
                self.get_logger().info(f"Loaded rho_L model from {rho_L_path}")
                
        except Exception as e:
            self.get_logger().error(f"Failed to load models: {e}")
            self.get_logger().warn("Proceeding with untrained networks")
        
        # 设置为评估模式
        self.phi_G_net.eval()
        self.phi_L_net.eval()
        self.rho_L_net.eval()
        
        # 使用双精度（与训练时一致）
        self.phi_G_net.double()
        self.phi_L_net.double()
        self.rho_L_net.double()
    
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
        
        # 发布估计的气动力（UAV0）
        self.uav0_aero_force_pub = self.create_publisher(
            Vector3Stamped,
            '/uav0/estimated_aero_force',
            qos_reliable
        )
        
        # 发布估计的气动力（UAV1）
        self.uav1_aero_force_pub = self.create_publisher(
            Vector3Stamped,
            '/uav1/estimated_aero_force',
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
    
    def construct_network_input(self, self_pose, self_velo, other_pose, other_velo):
        """
        构造神经网络输入
        
        输入维度：19
        [0:3]   - 自身相对地面的位置 (0-x, 0-y, 0-z)
        [3:6]   - 自身相对地面的速度 (0-vx, 0-vy, 0-vz)
        [6:9]   - 邻居相对自身的位置 (x21, y21, z21)
        [9:12]  - 邻居相对自身的速度 (vx21, vy21, vz21)
        [12:15] - 第三架飞机位置（本场景无，填0）
        [15:18] - 第三架飞机速度（本场景无，填0）
        [18]    - 场景类型编码（LL2L=11）
        """
        # 提取位置和速度
        self_pos = np.array([
            self_pose.pose.position.x,
            self_pose.pose.position.y,
            self_pose.pose.position.z
        ])
        
        self_vel = np.array([
            self_velo.twist.linear.x,
            self_velo.twist.linear.y,
            self_velo.twist.linear.z
        ])
        
        other_pos = np.array([
            other_pose.pose.position.x,
            other_pose.pose.position.y,
            other_pose.pose.position.z
        ])
        
        other_vel = np.array([
            other_velo.twist.linear.x,
            other_velo.twist.linear.y,
            other_velo.twist.linear.z
        ])
        
        # 构造输入向量
        network_input = np.zeros(19, dtype=np.float64)
        
        # [0:3] 自身相对地面的位置（负值）
        network_input[0:3] = 0 - self_pos
        
        # [3:6] 自身相对地面的速度（负值）
        network_input[3:6] = 0 - self_vel
        
        # [6:9] 邻居相对自身的位置
        network_input[6:9] = other_pos - self_pos
        
        # [9:12] 邻居相对自身的速度
        network_input[9:12] = other_vel - self_vel
        
        # [12:18] 没有第三架飞机，保持为0
        
        # [18] 场景类型编码
        network_input[18] = self.scene_type
        
        return network_input
    
    def estimate_aero_force(self, network_input):
        """
        使用神经网络估计气动力
        
        返回：
            Fa_z: 垂直方向的气动力（牛顿）
        """
        # 转换为torch tensor
        inputs = torch.from_numpy(network_input).unsqueeze(0)  # 增加batch维度
        
        # 神经网络前向传播
        with torch.no_grad():
            # LL2L场景：rho_L(phi_G(地面效应) + phi_L(邻居飞机))
            # phi_G使用 [2:6]: [z, vx, vy, vz]（地面到自身）
            # phi_L使用 [6:12]: [相对位置xyz, 相对速度xyz]
            ground_feature = self.phi_G_net(inputs[:, 2:6])
            neighbor_feature = self.phi_L_net(inputs[:, 6:12])
            
            # 组合特征并通过解码器
            combined_feature = ground_feature + neighbor_feature
            Fa_gram = self.rho_L_net(combined_feature)
            
            # 输出是克力（gram force），转换为牛顿
            Fa_z_newton = Fa_gram[0, 0].item() * self.gravity / 1000.0
        
        return Fa_z_newton
    
    def estimation_callback(self):
        """估计回调函数"""
        # 检查是否收到所有必要的数据
        if (self.uav0_pose is None or self.uav0_velo is None or
            self.uav1_pose is None or self.uav1_velo is None):
            # self.get_logger().warn("Waiting for all UAV state data...")
            return
        
        try:
            # 估计UAV0受到的气动力（来自UAV1的影响）
            input_uav0 = self.construct_network_input(
                self.uav0_pose, self.uav0_velo,
                self.uav1_pose, self.uav1_velo
            )
            Fa_z_uav0 = self.estimate_aero_force(input_uav0)
            
            # 估计UAV1受到的气动力（来自UAV0的影响）
            input_uav1 = self.construct_network_input(
                self.uav1_pose, self.uav1_velo,
                self.uav0_pose, self.uav0_velo
            )
            Fa_z_uav1 = self.estimate_aero_force(input_uav1)
            
            # 发布UAV0的气动力估计
            force_msg_uav0 = Vector3Stamped()
            force_msg_uav0.header.stamp = self.get_clock().now().to_msg()
            force_msg_uav0.header.frame_id = "world"
            force_msg_uav0.vector.x = 0.0  # x方向暂时为0
            force_msg_uav0.vector.y = 0.0  # y方向暂时为0
            force_msg_uav0.vector.z = Fa_z_uav0  # z方向气动力（牛顿）
            self.uav0_aero_force_pub.publish(force_msg_uav0)
            
            # 发布UAV1的气动力估计
            force_msg_uav1 = Vector3Stamped()
            force_msg_uav1.header.stamp = self.get_clock().now().to_msg()
            force_msg_uav1.header.frame_id = "world"
            force_msg_uav1.vector.x = 0.0
            force_msg_uav1.vector.y = 0.0
            force_msg_uav1.vector.z = Fa_z_uav1
            self.uav1_aero_force_pub.publish(force_msg_uav1)
            
            # 打印调试信息（可选）
            if self.get_logger().get_effective_level() <= rclpy.logging.LoggingSeverity.DEBUG:
                # 计算相对位置
                rel_pos = np.array([
                    self.uav1_pose.pose.position.x - self.uav0_pose.pose.position.x,
                    self.uav1_pose.pose.position.y - self.uav0_pose.pose.position.y,
                    self.uav1_pose.pose.position.z - self.uav0_pose.pose.position.z
                ])
                rel_dist = np.linalg.norm(rel_pos)
                
                self.get_logger().debug(
                    f"UAV0: z={self.uav0_pose.pose.position.z:.2f}m, Fa_z={Fa_z_uav0:.3f}N | "
                    f"UAV1: z={self.uav1_pose.pose.position.z:.2f}m, Fa_z={Fa_z_uav1:.3f}N | "
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
