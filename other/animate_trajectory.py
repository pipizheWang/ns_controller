#!/usr/bin/env python3
"""
无人机3D轨迹动画工具（双日志版本支持）
读取CSV格式的飞行日志并生成3D动画，展示无人机的飞行轨迹和姿态变化。

本版本支持将“最新的两份日志（*_0.csv 与 *_1.csv）”同时绘制到同一个动画中。
也可通过 --log0/--log1 指定文件路径。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import csv
import sys
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d


class QuadrotorModel:
    """四旋翼无人机模型"""
    
    def __init__(self, arm_length=0.15):
        """
        初始化四旋翼模型
        
        Args:
            arm_length: 机臂长度（米）
        """
        self.arm_length = arm_length
        self.rotor_radius = arm_length * 0.25  # 旋翼半径
        
        # 定义机身框架（X型配置）
        # 四个机臂的端点位置（机体坐标系）
        self.arm_angles = np.array([45, 135, 225, 315]) * np.pi / 180  # 转换为弧度
        
    def get_body_frame(self):
        """
        获取机体框架的顶点（机体坐标系）
        
        Returns:
            机臂端点坐标列表
        """
        # 计算四个机臂端点
        arms = []
        for angle in self.arm_angles:
            x = self.arm_length * np.cos(angle)
            y = self.arm_length * np.sin(angle)
            arms.append([x, y, 0])
        
        return np.array(arms)
    
    def rotation_matrix(self, roll, pitch, yaw):
        """
        计算旋转矩阵（ZYX欧拉角，航空顺序）
        
        Args:
            roll: 滚转角（弧度）
            pitch: 俯仰角（弧度）
            yaw: 偏航角（弧度）
            
        Returns:
            3x3旋转矩阵
        """
        # 滚转矩阵（绕X轴）
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # 俯仰矩阵（绕Y轴）
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # 偏航矩阵（绕Z轴）
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # 组合旋转矩阵：R = Rz * Ry * Rx
        return Rz @ Ry @ Rx
    
    def transform(self, position, roll, pitch, yaw):
        """
        将机体坐标系的点转换到世界坐标系
        
        Args:
            position: 无人机位置 [x, y, z]
            roll: 滚转角（弧度）
            pitch: 俯仰角（弧度）
            yaw: 偏航角（弧度）
            
        Returns:
            转换后的机臂端点坐标和旋翼圆心坐标
        """
        # 获取机体框架
        body_frame = self.get_body_frame()
        
        # 计算旋转矩阵
        R = self.rotation_matrix(roll, pitch, yaw)
        
        # 旋转并平移
        world_frame = (R @ body_frame.T).T + position
        
        return world_frame


class TrajectoryAnimator:
    """轨迹动画生成器"""
    
    def __init__(self, log_file_path, arm_length=0.15):
        """
        初始化轨迹动画器
        
        Args:
            log_file_path: 日志文件路径
            arm_length: 无人机机臂长度
        """
        self.log_file_path = Path(log_file_path)
        self.data = None
        self.quadrotor = QuadrotorModel(arm_length=arm_length)
        self.fig = None
        self.ax = None
        self.frame_skip = 1  # 跳帧设置，用于加速动画
        
    def load_data(self):
        """从CSV文件加载飞行日志数据"""
        if not self.log_file_path.exists():
            raise FileNotFoundError(f"日志文件不存在: {self.log_file_path}")
        
        # 读取CSV文件
        with open(self.log_file_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if len(rows) == 0:
            raise ValueError("日志文件为空")
        
        # 提取数据
        self.data = {
            'timestamp': np.array([float(row['timestamp']) for row in rows]),
            'x': np.array([float(row['x']) for row in rows]),
            'y': np.array([float(row['y']) for row in rows]),
            'z': np.array([float(row['z']) for row in rows]),
            'roll': np.array([float(row['roll']) for row in rows]),
            'pitch': np.array([float(row['pitch']) for row in rows]),
            'yaw': np.array([float(row['yaw']) for row in rows]),
            'x_des': np.array([float(row['x_des']) for row in rows]),
            'y_des': np.array([float(row['y_des']) for row in rows]),
            'z_des': np.array([float(row['z_des']) for row in rows]),
        }
        
        # 将时间戳转换为相对时间
        self.data['time'] = self.data['timestamp'] - self.data['timestamp'][0]
        
        print(f"成功加载 {len(rows)} 条数据记录")
        print(f"飞行时长: {self.data['time'][-1]:.2f} 秒")
        
        # 自动调整跳帧率，使动画大约有300-500帧
        total_frames = len(rows)
        target_frames = 400
        self.frame_skip = max(1, total_frames // target_frames)
        print(f"跳帧率: {self.frame_skip} (动画将有约 {total_frames // self.frame_skip} 帧)")
        
    def setup_plot(self):
        """设置绘图环境"""
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 设置标题
        self.fig.suptitle('Quadrotor Trajectory Animation', fontsize=16, fontweight='bold')
        
        # 设置坐标轴标签
        self.ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
        self.ax.set_ylabel('Y (m)', fontsize=12, labelpad=10)
        self.ax.set_zlabel('Z (m)', fontsize=12, labelpad=10)
        
        # 计算坐标轴范围
        all_x = np.concatenate([self.data['x'], self.data['x_des']])
        all_y = np.concatenate([self.data['y'], self.data['y_des']])
        all_z = np.concatenate([self.data['z'], self.data['z_des']])
        
        # 找到最大范围，使用相等的坐标轴比例
        max_range = np.array([
            all_x.max() - all_x.min(),
            all_y.max() - all_y.min(),
            all_z.max() - all_z.min()
        ]).max() / 2.0
        
        # 添加一些边距
        max_range *= 1.2
        
        mid_x = (all_x.max() + all_x.min()) * 0.5
        mid_y = (all_y.max() + all_y.min()) * 0.5
        mid_z = (all_z.max() + all_z.min()) * 0.5
        
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(4.5, 5.5)  # 固定z轴范围为4.5-5.5米
        
        # 设置视角
        self.ax.view_init(elev=25, azim=45)
        
        # 绘制期望轨迹（半透明红色虚线）
        self.ax.plot(self.data['x_des'], self.data['y_des'], self.data['z_des'],
                    'r--', linewidth=2, alpha=0.3, label='Desired Path')
        
        # 绘制起点标记
        self.ax.scatter(self.data['x'][0], self.data['y'][0], self.data['z'][0],
                       c='green', s=200, marker='o', label='Start', zorder=10)
        
        # 绘制终点标记
        self.ax.scatter(self.data['x_des'][-1], self.data['y_des'][-1], self.data['z_des'][-1],
                       c='red', s=200, marker='s', label='Goal', zorder=10)
        
        self.ax.legend(loc='upper right', fontsize=10)
        self.ax.grid(True, alpha=0.3)
        
    def init_animation(self):
        """初始化动画元素"""
        # 初始化轨迹线（蓝色）
        self.trajectory_line, = self.ax.plot([], [], [], 'b-', linewidth=2.5, 
                                            alpha=0.8, label='Actual Path')
        
        # 初始化无人机机臂（4条线段，从中心到各个机臂端点）
        self.arm_lines = []
        for i in range(4):
            line, = self.ax.plot([], [], [], 'k-', linewidth=3)
            self.arm_lines.append(line)
        
        # 初始化旋翼圆圈（4个）
        self.rotor_circles = []
        for i in range(4):
            # 在3D中绘制圆圈比较复杂，我们用多边形近似
            line, = self.ax.plot([], [], [], 'gray', linewidth=2, alpha=0.7)
            self.rotor_circles.append(line)
        
        # 添加信息文本
        self.info_text = self.ax.text2D(0.02, 0.95, '', transform=self.ax.transAxes,
                                       fontsize=10, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        return [self.trajectory_line] + self.arm_lines + self.rotor_circles + [self.info_text]
    
    def update_frame(self, frame):
        """更新动画帧"""
        # 实际数据索引（考虑跳帧）
        idx = frame * self.frame_skip
        if idx >= len(self.data['x']):
            idx = len(self.data['x']) - 1
        
        # 当前位置和姿态
        x = self.data['x'][idx]
        y = self.data['y'][idx]
        z = self.data['z'][idx]
        roll = self.data['roll'][idx]
        pitch = self.data['pitch'][idx]
        yaw = self.data['yaw'][idx]
        
        # 更新已飞过的轨迹
        self.trajectory_line.set_data(self.data['x'][:idx+1], self.data['y'][:idx+1])
        self.trajectory_line.set_3d_properties(self.data['z'][:idx+1])
        
        # 获取转换后的机臂端点
        position = np.array([x, y, z])
        # CSV中的姿态角是度数，需要转换为弧度传递给transform函数
        roll_rad = np.deg2rad(roll)
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)
        arm_endpoints = self.quadrotor.transform(position, roll_rad, pitch_rad, yaw_rad)
        
        # 更新机臂
        for i, line in enumerate(self.arm_lines):
            # 从中心到机臂端点
            line.set_data([position[0], arm_endpoints[i, 0]], 
                         [position[1], arm_endpoints[i, 1]])
            line.set_3d_properties([position[2], arm_endpoints[i, 2]])
        
        # 更新旋翼（绘制为圆圈）
        for i, line in enumerate(self.rotor_circles):
            # 在每个机臂端点绘制旋翼圆圈
            center = arm_endpoints[i]
            
            # 生成圆圈点（在垂直于机臂的平面上）
            n_points = 20
            theta = np.linspace(0, 2*np.pi, n_points)
            
            # 旋翼在机体XY平面上的圆（简化处理）
            radius = self.quadrotor.rotor_radius
            circle_x = radius * np.cos(theta)
            circle_y = radius * np.sin(theta)
            circle_z = np.zeros_like(theta)
            
            # 将圆圈转换到世界坐标系
            R = self.quadrotor.rotation_matrix(roll_rad, pitch_rad, yaw_rad)
            circle_points = np.vstack([circle_x, circle_y, circle_z])
            world_circle = (R @ circle_points).T + center
            
            line.set_data(world_circle[:, 0], world_circle[:, 1])
            line.set_3d_properties(world_circle[:, 2])
        
        # 更新信息文本
        time_str = f"Time: {self.data['time'][idx]:.2f}s / {self.data['time'][-1]:.2f}s"
        pos_str = f"Position: ({x:.2f}, {y:.2f}, {z:.2f}) m"
        # CSV中的姿态角已经是度数，无需转换
        att_str = f"Attitude: R={roll:.1f}° P={pitch:.1f}° Y={yaw:.1f}°"
        
        # 计算误差
        error_x = x - self.data['x_des'][idx]
        error_y = y - self.data['y_des'][idx]
        error_z = z - self.data['z_des'][idx]
        error_total = np.sqrt(error_x**2 + error_y**2 + error_z**2)
        error_str = f"Tracking Error: {error_total:.3f} m"
        
        self.info_text.set_text(f"{time_str}\n{pos_str}\n{att_str}\n{error_str}")
        
        return [self.trajectory_line] + self.arm_lines + self.rotor_circles + [self.info_text]
    
    def create_animation(self, interval=50, save_path=None):
        """
        创建动画
        
        Args:
            interval: 帧间隔（毫秒）
            save_path: 保存路径（如果为None则只显示）
        """
        if self.data is None:
            self.load_data()
        
        self.setup_plot()
        
        # 计算帧数
        n_frames = len(self.data['x']) // self.frame_skip
        
        print(f"\n开始生成动画...")
        print(f"总帧数: {n_frames}")
        print(f"帧间隔: {interval} ms")
        
        # 创建动画
        anim = FuncAnimation(
            self.fig,
            self.update_frame,
            init_func=self.init_animation,
            frames=n_frames,
            interval=interval,
            blit=False,  # 3D动画不能使用blit
            repeat=True
        )
        
        # 保存或显示
        if save_path is not None:
            save_path = Path(save_path)
            print(f"\n正在保存动画到: {save_path}")
            
            if save_path.suffix == '.gif':
                writer = PillowWriter(fps=1000//interval)
                anim.save(save_path, writer=writer, dpi=100)
                print(f"动画已保存为GIF: {save_path}")
            elif save_path.suffix == '.mp4':
                anim.save(save_path, writer='ffmpeg', fps=1000//interval, dpi=100)
                print(f"动画已保存为MP4: {save_path}")
            else:
                print(f"警告: 不支持的文件格式 {save_path.suffix}，将保存为GIF")
                save_path = save_path.with_suffix('.gif')
                writer = PillowWriter(fps=1000//interval)
                anim.save(save_path, writer=writer, dpi=100)
                print(f"动画已保存: {save_path}")
        
        return anim


class DualTrajectoryAnimator:
    """双日志轨迹动画：将两架无人机的轨迹绘制在同一个 3D 动画中。"""

    def __init__(self, log0_path, log1_path, arm_length=0.15):
        self.log0_path = Path(log0_path)
        self.log1_path = Path(log1_path)
        self.data0 = None
        self.data1 = None
        self.quad0 = QuadrotorModel(arm_length=arm_length)
        self.quad1 = QuadrotorModel(arm_length=arm_length)
        self.fig = None
        self.ax = None
        self.frame_skip0 = 1
        self.frame_skip1 = 1

        # 时间戳对齐参数
        self.time_offset = None  # log1相对于log0的时间偏差
        self.aligned_time_range = None  # 对齐后的时间范围 (start, end)
        self.aligned_duration = None  # 动画总时长

        # 绘图元素
        self.trajectory_line0 = None
        self.trajectory_line1 = None
        self.arm_lines0 = []
        self.arm_lines1 = []
        self.rotor_circles0 = []
        self.rotor_circles1 = []
        self.info_text = None

    def _load_one(self, path):
        if not path.exists():
            raise FileNotFoundError(f"日志文件不存在: {path}")
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            raise ValueError(f"日志文件为空: {path}")
        data = {
            'timestamp': np.array([float(row['timestamp']) for row in rows]),
            'x': np.array([float(row['x']) for row in rows]),
            'y': np.array([float(row['y']) for row in rows]),
            'z': np.array([float(row['z']) for row in rows]),
            'roll': np.array([float(row['roll']) for row in rows]),
            'pitch': np.array([float(row['pitch']) for row in rows]),
            'yaw': np.array([float(row['yaw']) for row in rows]),
            'x_des': np.array([float(row['x_des']) for row in rows]),
            'y_des': np.array([float(row['y_des']) for row in rows]),
            'z_des': np.array([float(row['z_des']) for row in rows]),
        }
        data['time'] = data['timestamp'] - data['timestamp'][0]
        return data, len(rows)

    def _align_timestamps(self):
        """
        根据时间戳对齐两个日志文件。
        处理整数秒偏差和采样率不一致的问题。
        """
        ts0 = self.data0['timestamp']
        ts1 = self.data1['timestamp']
        
        # 粗对齐：找到两个日志时间戳的大致对齐点
        # 取两个日志开始时间中较晚的作为对齐起点
        start_ts0 = ts0[0]
        start_ts1 = ts1[0]
        aligned_start = max(start_ts0, start_ts1)
        
        # 取两个日志结束时间中较早的作为对齐终点
        end_ts0 = ts0[-1]
        end_ts1 = ts1[-1]
        aligned_end = min(end_ts0, end_ts1)
        
        # 检查是否有重叠
        if aligned_start >= aligned_end:
            raise ValueError(
                f"两个日志没有足够的时间重叠。"
                f"log0: [{start_ts0:.3f}, {end_ts0:.3f}], "
                f"log1: [{start_ts1:.3f}, {end_ts1:.3f}]"
            )
        
        # 计算时间偏差（log1相对于log0）
        # 这里我们用两个日志的起始时间差作为初始偏差估计
        self.time_offset = start_ts1 - start_ts0
        
        # 保存对齐的时间范围（使用相对时间）
        self.aligned_time_range = (aligned_start - start_ts0, aligned_end - start_ts0)
        self.aligned_duration = self.aligned_time_range[1] - self.aligned_time_range[0]
        
        print(f"\n时间戳对齐信息:")
        print(f"  log0 起始时间戳: {start_ts0:.6f}")
        print(f"  log1 起始时间戳: {start_ts1:.6f}")
        print(f"  时间偏差 (log1 - log0): {self.time_offset:.6f} 秒")
        print(f"  对齐后的时间范围: [{self.aligned_time_range[0]:.2f}, {self.aligned_time_range[1]:.2f}] 秒")
        print(f"  对齐后的总时长: {self.aligned_duration:.2f} 秒")

    def _find_nearest_frame(self, target_time, timestamps):
        """
        根据目标时间找到最接近的帧索引。
        使用二分查找加速搜索。
        
        Args:
            target_time: 目标时间（相对时间）
            timestamps: 时间戳数组（绝对时间戳）
        
        Returns:
            最接近的帧索引
        """
        # 将相对时间转换为绝对时间戳
        target_ts = target_time + timestamps[0]
        
        # 使用二分查找找到最接近的时间戳
        idx = np.searchsorted(timestamps, target_ts)
        
        # 检查边界
        if idx <= 0:
            return 0
        if idx >= len(timestamps):
            return len(timestamps) - 1
        
        # 比较两个相邻索引，返回更接近的一个
        if abs(timestamps[idx] - target_ts) < abs(timestamps[idx-1] - target_ts):
            return idx
        else:
            return idx - 1

    def load_data(self):
        self.data0, n0 = self._load_one(self.log0_path)
        self.data1, n1 = self._load_one(self.log1_path)
        print(f"log0: {self.log0_path.name} -> {n0} 条记录, 时长 {self.data0['time'][-1]:.2f}s")
        print(f"log1: {self.log1_path.name} -> {n1} 条记录, 时长 {self.data1['time'][-1]:.2f}s")

        # 根据时间戳对齐两个日志
        self._align_timestamps()

        # 自动跳帧，使动画在合理帧数范围，基于对齐后的时长
        target_frames = 400
        # 使用对齐后的时间长度来计算跳帧率
        estimated_total_frames = min(n0, n1)
        frame_skip = max(1, estimated_total_frames // target_frames)
        self.frame_skip0 = frame_skip
        self.frame_skip1 = frame_skip
        
        # 重新计算帧数（基于对齐后的时间范围）
        n_frames_aligned = int(self.aligned_duration * 50 / frame_skip)  # 假设50Hz采样
        print(f"自动跳帧率: {frame_skip}, 对齐后动画帧数约: {n_frames_aligned}")

    def setup_plot(self):
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.fig.suptitle('Dual Quadrotor Trajectory Animation (log0 + log1)', fontsize=16, fontweight='bold')
        self.ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
        self.ax.set_ylabel('Y (m)', fontsize=12, labelpad=10)
        self.ax.set_zlabel('Z (m)', fontsize=12, labelpad=10)

        # 结合两份日志的数据计算范围
        all_x = np.concatenate([self.data0['x'], self.data0['x_des'], self.data1['x'], self.data1['x_des']])
        all_y = np.concatenate([self.data0['y'], self.data0['y_des'], self.data1['y'], self.data1['y_des']])
        all_z = np.concatenate([self.data0['z'], self.data0['z_des'], self.data1['z'], self.data1['z_des']])
        max_range = np.array([
            all_x.max() - all_x.min(),
            all_y.max() - all_y.min(),
            all_z.max() - all_z.min(),
        ]).max() / 2.0
        max_range *= 1.2
        mid_x = (all_x.max() + all_x.min()) * 0.5
        mid_y = (all_y.max() + all_y.min()) * 0.5
        mid_z = (all_z.max() + all_z.min()) * 0.5
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(4.5, 5.5)  # 固定z轴范围为4.5-6米

        self.ax.view_init(elev=25, azim=45)

        # 期望轨迹（半透明虚线）：蓝色对应 log0，橙色对应 log1
        self.ax.plot(self.data0['x_des'], self.data0['y_des'], self.data0['z_des'],
                     color='#1f77b4', linestyle='--', linewidth=2, alpha=0.35, label='Desired 0')
        self.ax.plot(self.data1['x_des'], self.data1['y_des'], self.data1['z_des'],
                     color='#ff7f0e', linestyle='--', linewidth=2, alpha=0.35, label='Desired 1')

        # 起点/终点标记
        self.ax.scatter(self.data0['x'][0], self.data0['y'][0], self.data0['z'][0],
                        c='green', s=160, marker='o', label='Start 0', zorder=10)
        self.ax.scatter(self.data0['x'][-1], self.data0['y'][-1], self.data0['z'][-1],
                        c='red', s=160, marker='s', label='End 0', zorder=10)
        self.ax.scatter(self.data1['x'][0], self.data1['y'][0], self.data1['z'][0],
                        c='lime', s=140, marker='o', label='Start 1', zorder=10)
        self.ax.scatter(self.data1['x'][-1], self.data1['y'][-1], self.data1['z'][-1],
                        c='darkred', s=140, marker='s', label='End 1', zorder=10)

        self.ax.legend(loc='upper right', fontsize=10)
        self.ax.grid(True, alpha=0.3)

    def init_animation(self):
        # 轨迹线
        self.trajectory_line0, = self.ax.plot([], [], [], color='#1f77b4', linewidth=2.5, alpha=0.9, label='Actual 0')
        self.trajectory_line1, = self.ax.plot([], [], [], color='#ff7f0e', linewidth=2.5, alpha=0.9, label='Actual 1')

        # 两架无人机机臂
        self.arm_lines0 = []
        self.arm_lines1 = []
        for _ in range(4):
            l0, = self.ax.plot([], [], [], color='navy', linewidth=3)
            l1, = self.ax.plot([], [], [], color='saddlebrown', linewidth=3)
            self.arm_lines0.append(l0)
            self.arm_lines1.append(l1)

        # 旋翼圆圈（用折线近似）
        self.rotor_circles0 = []
        self.rotor_circles1 = []
        for _ in range(4):
            c0, = self.ax.plot([], [], [], color='gray', linewidth=2, alpha=0.7)
            c1, = self.ax.plot([], [], [], color='gray', linewidth=2, alpha=0.7)
            self.rotor_circles0.append(c0)
            self.rotor_circles1.append(c1)

        self.info_text = self.ax.text2D(
            0.02, 0.95, '', transform=self.ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        return [self.trajectory_line0, self.trajectory_line1] + \
               self.arm_lines0 + self.arm_lines1 + \
               self.rotor_circles0 + self.rotor_circles1 + [self.info_text]

    def _update_one(self, idx, data, quad, arm_lines, rotor_circles, traj_line):
        """
        更新单个无人机的动画元素。
        
        Args:
            idx: 数据索引
            data: 该无人机的数据字典
            quad: 该无人机的四旋翼模型
            arm_lines: 机臂线段列表
            rotor_circles: 旋翼圆圈列表
            traj_line: 轨迹线对象
        
        Returns:
            (位置向量, 姿态角度数组, 跟踪误差)
        """
        # 位置与姿态
        x, y, z = data['x'][idx], data['y'][idx], data['z'][idx]
        roll = np.deg2rad(data['roll'][idx])
        pitch = np.deg2rad(data['pitch'][idx])
        yaw = np.deg2rad(data['yaw'][idx])

        # 轨迹（仅绘制到当前索引为止）
        traj_line.set_data(data['x'][:idx+1], data['y'][:idx+1])
        traj_line.set_3d_properties(data['z'][:idx+1])

        # 机臂
        position = np.array([x, y, z])
        arm_endpoints = quad.transform(position, roll, pitch, yaw)
        for i, line in enumerate(arm_lines):
            line.set_data([position[0], arm_endpoints[i, 0]], [position[1], arm_endpoints[i, 1]])
            line.set_3d_properties([position[2], arm_endpoints[i, 2]])

        # 旋翼
        n_points = 20
        theta = np.linspace(0, 2*np.pi, n_points)
        radius = quad.rotor_radius
        circle_local = np.vstack([radius*np.cos(theta), radius*np.sin(theta), np.zeros_like(theta)])
        R = quad.rotation_matrix(roll, pitch, yaw)
        for i, line in enumerate(rotor_circles):
            world_circle = (R @ circle_local).T + arm_endpoints[i]
            line.set_data(world_circle[:, 0], world_circle[:, 1])
            line.set_3d_properties(world_circle[:, 2])

        # 误差
        ex = x - data['x_des'][idx]
        ey = y - data['y_des'][idx]
        ez = z - data['z_des'][idx]
        err = np.sqrt(ex**2 + ey**2 + ez**2)
        return position, np.rad2deg([roll, pitch, yaw]), err

    def update_frame(self, frame):
        """
        根据时间戳同步更新两个日志的动画帧。
        使用共同的"参考时间"找到每个日志中最接近的数据点。
        """
        # 根据帧数计算参考时间（相对于对齐时间范围的开始）
        ref_time = self.aligned_time_range[0] + frame * 0.02 * self.frame_skip0  # 50Hz采样
        
        # 限制在对齐时间范围内
        ref_time = np.clip(ref_time, self.aligned_time_range[0], self.aligned_time_range[1])
        
        # 在两个日志中找到最接近的帧
        # 对于log0，直接使用相对时间
        idx0 = self._find_nearest_frame(ref_time, self.data0['timestamp'])
        
        # 对于log1，需要考虑时间偏差
        # log1的参考时间需要转换为log1的参考系统
        ref_time_log1 = ref_time + self.time_offset
        idx1 = self._find_nearest_frame(ref_time_log1, self.data1['timestamp'])
        
        # 更新两个无人机
        pos0, att0_deg, err0 = self._update_one(idx0, self.data0, self.quad0, self.arm_lines0, self.rotor_circles0, self.trajectory_line0)
        pos1, att1_deg, err1 = self._update_one(idx1, self.data1, self.quad1, self.arm_lines1, self.rotor_circles1, self.trajectory_line1)

        # 获取实际的相对时间（用于显示）
        t0 = self.data0['time'][idx0]
        t1 = self.data1['time'][idx1]
        
        # 构建信息文本
        t_str = f"Aligned Time: {ref_time - self.aligned_time_range[0]:.2f}s / {self.aligned_duration:.2f}s"
        info = (
            f"{t_str}\n"
            f"log0 (t={t0:.2f}s) Pos=({pos0[0]:.2f},{pos0[1]:.2f},{pos0[2]:.2f}) "
            f"Att=({att0_deg[0]:.1f}°, {att0_deg[1]:.1f}°, {att0_deg[2]:.1f}°) Err={err0:.3f}m\n"
            f"log1 (t={t1:.2f}s) Pos=({pos1[0]:.2f},{pos1[1]:.2f},{pos1[2]:.2f}) "
            f"Att=({att1_deg[0]:.1f}°, {att1_deg[1]:.1f}°, {att1_deg[2]:.1f}°) Err={err1:.3f}m"
        )
        self.info_text.set_text(info)

        return (
            [self.trajectory_line0, self.trajectory_line1]
            + self.arm_lines0 + self.arm_lines1
            + self.rotor_circles0 + self.rotor_circles1
            + [self.info_text]
        )

    def create_animation(self, interval=50, save_path=None):
        if self.data0 is None or self.data1 is None:
            self.load_data()

        self.setup_plot()
        
        # 基于对齐后的时长计算帧数（50Hz采样）
        n_frames = int(self.aligned_duration * 50 / self.frame_skip0)

        print(f"\n开始生成双日志动画（时间戳同步）...")
        print(f"对齐时间范围: {self.aligned_time_range[0]:.2f}s - {self.aligned_time_range[1]:.2f}s")
        print(f"对齐总时长: {self.aligned_duration:.2f}s")
        print(f"帧间隔: {interval} ms")
        print(f"总帧数: {n_frames}")

        anim = FuncAnimation(
            self.fig,
            self.update_frame,
            init_func=self.init_animation,
            frames=n_frames,
            interval=interval,
            blit=False,
            repeat=True,
        )

        if save_path is not None:
            save_path = Path(save_path)
            print(f"\n正在保存动画到: {save_path}")
            if save_path.suffix == '.gif':
                writer = PillowWriter(fps=1000//interval)
                anim.save(save_path, writer=writer, dpi=100)
                print(f"动画已保存为GIF: {save_path}")
            elif save_path.suffix == '.mp4':
                anim.save(save_path, writer='ffmpeg', fps=1000//interval, dpi=100)
                print(f"动画已保存为MP4: {save_path}")
            else:
                print(f"警告: 不支持的文件格式 {save_path.suffix}，将保存为GIF")
                save_path = save_path.with_suffix('.gif')
                writer = PillowWriter(fps=1000//interval)
                anim.save(save_path, writer=writer, dpi=100)
                print(f"动画已保存: {save_path}")

        return anim


class DualTrajectoryAnimatorXZ(DualTrajectoryAnimator):
    """双日志轨迹动画：XZ平面视图（2D）"""

    def setup_plot(self):
        """设置XZ平面视角的绘图环境"""
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.fig.suptitle('Dual Quadrotor Trajectory Animation - XZ View (log0 + log1)', fontsize=16, fontweight='bold')
        self.ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
        self.ax.set_ylabel('Y (m)', fontsize=12, labelpad=10)
        self.ax.set_zlabel('Z (m)', fontsize=12, labelpad=10)

        # 结合两份日志的数据计算范围
        all_x = np.concatenate([self.data0['x'], self.data0['x_des'], self.data1['x'], self.data1['x_des']])
        all_y = np.concatenate([self.data0['y'], self.data0['y_des'], self.data1['y'], self.data1['y_des']])
        all_z = np.concatenate([self.data0['z'], self.data0['z_des'], self.data1['z'], self.data1['z_des']])
        max_range = np.array([
            all_x.max() - all_x.min(),
            all_y.max() - all_y.min(),
            all_z.max() - all_z.min(),
        ]).max() / 2.0
        max_range *= 1.2
        mid_x = (all_x.max() + all_x.min()) * 0.5
        mid_y = (all_y.max() + all_y.min()) * 0.5
        mid_z = (all_z.max() + all_z.min()) * 0.5
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(4.5, 5.5)  # 固定z轴范围为4.5-5.5米

        # XZ平面视角：仰角0度（平视），方位角-90度（正对XZ平面）
        self.ax.view_init(elev=0, azim=-90)

        # 期望轨迹（半透明虚线）：蓝色对应 log0，橙色对应 log1
        self.ax.plot(self.data0['x_des'], self.data0['y_des'], self.data0['z_des'],
                     color='#1f77b4', linestyle='--', linewidth=2, alpha=0.35, label='Desired 0')
        self.ax.plot(self.data1['x_des'], self.data1['y_des'], self.data1['z_des'],
                     color='#ff7f0e', linestyle='--', linewidth=2, alpha=0.35, label='Desired 1')

        # 起点/终点标记
        self.ax.scatter(self.data0['x'][0], self.data0['y'][0], self.data0['z'][0],
                        c='green', s=160, marker='o', label='Start 0', zorder=10)
        self.ax.scatter(self.data0['x'][-1], self.data0['y'][-1], self.data0['z'][-1],
                        c='red', s=160, marker='s', label='End 0', zorder=10)
        self.ax.scatter(self.data1['x'][0], self.data1['y'][0], self.data1['z'][0],
                        c='lime', s=140, marker='o', label='Start 1', zorder=10)
        self.ax.scatter(self.data1['x'][-1], self.data1['y'][-1], self.data1['z'][-1],
                        c='darkred', s=140, marker='s', label='End 1', zorder=10)

        self.ax.legend(loc='upper right', fontsize=10)
        self.ax.grid(True, alpha=0.3)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quadrotor Trajectory Animation Tool (Dual logs supported)')
    # 可选指定两份日志；若未指定则自动在 log 目录里选择最新一对 (_0 与 _1)
    parser.add_argument('--log0', type=str, default=None, help='Path to log file ending with _0.csv')
    parser.add_argument('--log1', type=str, default=None, help='Path to log file ending with _1.csv')
    parser.add_argument('--save', type=str, default=None,
                       help='Save animation to file (support .gif and .mp4)')
    parser.add_argument('--interval', type=int, default=50,
                       help='Frame interval in milliseconds (default: 50)')
    parser.add_argument('--arm-length', type=float, default=0.5,
                       help='Quadrotor arm length in meters (default: 0.5)')
    parser.add_argument('--list', action='store_true',
                       help='List all log files in log directory')
    parser.add_argument('--view', type=str, default='3d', choices=['3d', 'xz'],
                       help='View mode: 3d (default) or xz (XZ plane 2D view)')
    
    args = parser.parse_args()
    
    # 获取log目录路径（源码同级：.../ns_controller/log）
    current_file = Path(__file__).resolve()
    package_dir = current_file.parent.parent
    log_dir = package_dir / 'log'
    
    # 列出所有日志文件
    if args.list:
        if log_dir.exists():
            log_files = sorted(log_dir.glob('*.csv'))
            if log_files:
                print(f"\n在 {log_dir} 中找到以下日志文件：")
                for i, log_file in enumerate(log_files, 1):
                    size_kb = log_file.stat().st_size / 1024
                    print(f"  {i}. {log_file.name} ({size_kb:.1f} KB)")
            else:
                print(f"在 {log_dir} 中没有找到日志文件")
        else:
            print(f"日志目录不存在: {log_dir}")
        return
    
    # 选择两份日志：优先使用 --log0/--log1，否则在 log_dir 自动匹配最新一对
    def find_latest_pair(directory: Path):
        csvs = list(directory.glob('*.csv'))
        if not csvs:
            return None
        pair_map = {}
        for f in csvs:
            stem = f.stem
            if stem.endswith('_0') or stem.endswith('_1'):
                base = stem[:-2]
                suffix = stem[-1]
                rec = pair_map.get(base, {'0': None, '1': None, 'mtime': 0.0})
                rec[suffix] = f
                rec['mtime'] = max(rec['mtime'], f.stat().st_mtime)
                pair_map[base] = rec
        candidates = [(base, rec) for base, rec in pair_map.items() if rec['0'] and rec['1']]
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[1]['mtime'])
        base, rec = candidates[-1]
        return rec['0'], rec['1'], base

    if args.log0 and args.log1:
        log0 = Path(args.log0)
        log1 = Path(args.log1)
        if not log0.exists() or not log1.exists():
            print("错误: --log0 或 --log1 路径不存在")
            sys.exit(1)
        base_name = None
    else:
        if not log_dir.exists():
            print(f"错误: 日志目录不存在: {log_dir}")
            sys.exit(1)
        pair = find_latest_pair(log_dir)
        if not pair:
            print(f"错误: 在 {log_dir} 中未找到匹配的一对日志(*_0.csv 与 *_1.csv)")
            print("使用 --list 查看可用的日志文件")
            sys.exit(1)
        log0, log1, base_name = pair
        print(f"使用最新一对日志: {log0.name}  |  {log1.name}")
    
    # 确定保存路径
    save_path = None
    if args.save:
        save_path = Path(args.save)
    else:
        # 默认保存到 log 目录，文件名用共同前缀 + _dual_animation.gif
        if base_name is None:
            # 尝试根据两个文件名构造
            stem0 = Path(log0).stem
            stem1 = Path(log1).stem
            base_name = stem0.rsplit('_', 1)[0] if stem0.rsplit('_', 1)[0] == stem1.rsplit('_', 1)[0] else f"{stem0}__{stem1}"
        save_path = log_dir / f'{base_name}_dual_animation.gif'
        print(f"未指定保存路径，将保存到: {save_path}")
    
    try:
        # 根据视图模式选择动画器类
        if args.view == 'xz':
            animator = DualTrajectoryAnimatorXZ(log0, log1, arm_length=args.arm_length)
            print("使用XZ平面视图模式")
        else:
            animator = DualTrajectoryAnimator(log0, log1, arm_length=args.arm_length)
            print("使用3D视图模式")
        
        animator.load_data()

        # 生成动画
        anim = animator.create_animation(interval=args.interval, save_path=save_path)
        
        # 显示动画窗口
        print("\n显示动画窗口（关闭窗口以退出）...")
        plt.show()
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
