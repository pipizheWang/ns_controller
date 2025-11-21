#!/usr/bin/env python3

"""
轨迹同步触发节点
提供服务来同步启动/停止两架无人机的轨迹跟踪模式
支持 NL (无补偿) 和 NS (有补偿) 两种控制器模式
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
import time


class TrajSyncNode(Node):
    def __init__(self):
        super().__init__('traj_sync_node')
        self.get_logger().info("Trajectory Sync Node started")
        
        # 声明参数：控制器模式 ('NL' 或 'NS')
        self.declare_parameter('controller_mode', 'NS')
        
        # 创建服务
        self.srv = self.create_service(
            SetBool, 
            'sync_traj_mode', 
            self.sync_traj_callback
        )
        
        # 获取控制器模式
        self.controller_mode = self.get_parameter('controller_mode').value
        self.get_logger().info(f"Controller mode: {self.controller_mode}")
        
        # 根据模式创建参数设置客户端
        if self.controller_mode == 'NL':
            uav0_service = '/traj_controller_NL0/set_parameters'
            uav1_service = '/traj_controller_NL1/set_parameters'
        elif self.controller_mode == 'NS':
            uav0_service = '/traj_controller_NS0/set_parameters'
            uav1_service = '/traj_controller_NS1/set_parameters'
        else:
            self.get_logger().error(f"Unknown controller mode: {self.controller_mode}")
            self.get_logger().error("Valid modes are: 'NL' or 'NS'")
            raise ValueError(f"Invalid controller_mode: {self.controller_mode}")
        
        self.param_client_uav0 = self.create_client(SetParameters, uav0_service)
        self.param_client_uav1 = self.create_client(SetParameters, uav1_service)
        
        # 等待服务可用
        self.get_logger().info(f"Waiting for parameter services ({self.controller_mode} mode)...")
        self.get_logger().info(f"  - {uav0_service}")
        self.get_logger().info(f"  - {uav1_service}")
        
        try:
            self.param_client_uav0.wait_for_service(timeout_sec=5.0)
            self.param_client_uav1.wait_for_service(timeout_sec=5.0)
            self.get_logger().info("✓ Parameter services available")
        except Exception as e:
            self.get_logger().error(f"✗ Timeout waiting for services. Make sure controllers are running!")
            self.get_logger().error(f"  Error: {e}")
    
    def sync_traj_callback(self, request, response):
        """
        同步设置两架无人机的 traj_mode 参数
        request.data: True=启动轨迹, False=停止轨迹
        """
        traj_mode = request.data
        
        self.get_logger().info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.get_logger().info(f"Syncing trajectory mode to: {traj_mode}")
        self.get_logger().info(f"Controller mode: {self.controller_mode}")
        
        try:
            # 准备参数
            param = Parameter()
            param.name = 'traj_mode'
            param.value = ParameterValue()
            param.value.type = ParameterType.PARAMETER_BOOL
            param.value.bool_value = traj_mode
            
            # 创建请求
            req_uav0 = SetParameters.Request()
            req_uav0.parameters = [param]
            
            req_uav1 = SetParameters.Request()
            req_uav1.parameters = [param]
            
            # 同时发送请求（几乎同时）
            self.get_logger().info("Sending parameter requests to both UAVs...")
            future_uav0 = self.param_client_uav0.call_async(req_uav0)
            future_uav1 = self.param_client_uav1.call_async(req_uav1)
            
            # 等待响应
            rclpy.spin_until_future_complete(self, future_uav0, timeout_sec=1.0)
            rclpy.spin_until_future_complete(self, future_uav1, timeout_sec=1.0)
            
            # 检查结果
            uav0_success = future_uav0.done() and future_uav0.result() is not None
            uav1_success = future_uav1.done() and future_uav1.result() is not None
            
            if uav0_success and uav1_success:
                self.get_logger().info("✓ UAV0 parameter set successfully")
                self.get_logger().info("✓ UAV1 parameter set successfully")
                self.get_logger().info(f"✓ Both UAVs ({self.controller_mode} mode) trajectory synchronized!")
                response.success = True
                response.message = f"[{self.controller_mode}] Trajectory mode set to {traj_mode} for both UAVs"
            else:
                error_msg = []
                if not uav0_success:
                    self.get_logger().error("✗ Failed to set parameter for UAV0")
                    error_msg.append("UAV0")
                if not uav1_success:
                    self.get_logger().error("✗ Failed to set parameter for UAV1")
                    error_msg.append("UAV1")
                
                response.success = False
                response.message = f"Failed to synchronize: {', '.join(error_msg)}"
                
        except Exception as e:
            self.get_logger().error(f"✗ Error during sync: {e}")
            response.success = False
            response.message = str(e)
        
        self.get_logger().info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = TrajSyncNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
