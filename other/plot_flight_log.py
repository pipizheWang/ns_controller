#!/usr/bin/env python3
"""
飞行日志绘图工具（双日志版本）

读取 CSV 格式的飞行日志并绘制：
- 位置响应、跟踪误差、姿态角、轨迹曲线

本版本针对一次生成两份日志（文件名分别以 _0.csv 和 _1.csv 结尾）的情形：
- 每张图都使用两行布局：
    - 上排：_0.csv 的绘制
    - 下排：_1.csv 的绘制
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import sys


class FlightLogPlotter:
    def __init__(self, log_file_path):
        """
        初始化飞行日志绘图器
        
        Args:
            log_file_path: 日志文件路径
        """
        self.log_file_path = Path(log_file_path)
        self.data = None
        self.has_aero_data = False
        # 时间戳对齐相关属性
        self.aligned_indices = None  # 对齐后的有效索引范围
        
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
        
        # 提取数据（适配新的字段名）
        # tick是微秒单位
        self.data = {
            'timestamp': np.array([float(row['tick']) for row in rows]),
            'x': np.array([float(row['pos_x']) for row in rows]),
            'y': np.array([float(row['pos_y']) for row in rows]),
            'z': np.array([float(row['pos_z']) for row in rows]),
            'vx': np.array([float(row['vel_x']) for row in rows]),
            'vy': np.array([float(row['vel_y']) for row in rows]),
            'vz': np.array([float(row['vel_z']) for row in rows]),
        }
        
        # 姿态角（如果存在）
        if 'roll' in rows[0]:
            self.data['roll'] = np.array([float(row['roll']) for row in rows])
            self.data['pitch'] = np.array([float(row['pitch']) for row in rows])
            self.data['yaw'] = np.array([float(row['yaw']) for row in rows])
        else:
            # 如果没有姿态角数据，使用全零
            self.data['roll'] = np.zeros(len(rows))
            self.data['pitch'] = np.zeros(len(rows))
            self.data['yaw'] = np.zeros(len(rows))
        
        # 期望位置（如果存在）
        if 'x_des' in rows[0]:
            self.data['x_des'] = np.array([float(row['x_des']) for row in rows])
            self.data['y_des'] = np.array([float(row['y_des']) for row in rows])
            self.data['z_des'] = np.array([float(row['z_des']) for row in rows])
        else:
            # 如果没有期望位置，使用实际位置（误差为0）
            self.data['x_des'] = self.data['x'].copy()
            self.data['y_des'] = self.data['y'].copy()
            self.data['z_des'] = self.data['z'].copy()
        
        # 检查是否有扰动力数据列（NS控制器生成的日志）
        self.has_aero_data = 'Fa_z' in rows[0]
        if self.has_aero_data:
            self.data['Fa_z'] = np.array([float(row['Fa_z']) for row in rows])
        
        # 将时间戳转换为相对时间（从0开始）
        # tick是微秒单位，转换为秒
        self.data['time'] = (self.data['timestamp'] - self.data['timestamp'][0]) / 1e6
        
        # 计算跟踪误差
        self.data['error_x'] = self.data['x'] - self.data['x_des']
        self.data['error_y'] = self.data['y'] - self.data['y_des']
        self.data['error_z'] = self.data['z'] - self.data['z_des']
        
        # 初始化对齐索引（暂未设置）
        self.aligned_indices = (0, len(rows))
        
        print(f"成功加载 {len(rows)} 条数据记录")
        if self.has_aero_data:
            print(f"  检测到气动扰动力数据 (Fa_z)")
        print(f"飞行时长: {self.data['time'][-1]:.2f} 秒")
    
    def get_aligned_data(self):
        """获取对齐后的数据索引范围"""
        if self.aligned_indices is None:
            return 0, len(self.data['x'])
        return self.aligned_indices
def _align_timestamps_dual(plotter0: FlightLogPlotter, plotter1: FlightLogPlotter):
    """
    根据时间戳对齐两个日志文件。
    处理整数秒偏差和采样率不一致的问题。
    
    Args:
        plotter0: 第一个日志绘图器
        plotter1: 第二个日志绘图器
    """
    # 使用相对时间（秒）进行对齐
    time0 = plotter0.data['time']
    time1 = plotter1.data['time']
    
    # 粗对齐：找到两个日志时间戳的大致对齐点
    # 取两个日志开始时间中较晚的作为对齐起点
    start_t0 = time0[0]
    start_t1 = time1[0]
    aligned_start = max(start_t0, start_t1)
    
    # 取两个日志结束时间中较早的作为对齐终点
    end_t0 = time0[-1]
    end_t1 = time1[-1]
    aligned_end = min(end_t0, end_t1)
    
    # 检查是否有重叠
    if aligned_start >= aligned_end:
        raise ValueError(
            f"两个日志没有足够的时间重叠。"
            f"log0: [{start_t0:.3f}s, {end_t0:.3f}s], "
            f"log1: [{start_t1:.3f}s, {end_t1:.3f}s]"
        )
    
    # 计算时间偏差（log1相对于log0）
    time_offset = start_t1 - start_t0
    
    # 找到对齐起点和终点对应的索引
    idx0_start = np.searchsorted(time0, aligned_start)
    idx0_end = np.searchsorted(time0, aligned_end)
    idx1_start = np.searchsorted(time1, aligned_start)
    idx1_end = np.searchsorted(time1, aligned_end)
    
    # 设置对齐后的索引范围
    plotter0.aligned_indices = (idx0_start, idx0_end)
    plotter1.aligned_indices = (idx1_start, idx1_end)
    
    aligned_duration = aligned_end - aligned_start
    
    print(f"\n时间戳对齐信息:")
    print(f"  log0 时间范围: [{start_t0:.3f}s, {end_t0:.3f}s]")
    print(f"  log1 时间范围: [{start_t1:.3f}s, {end_t1:.3f}s]")
    print(f"  时间偏差 (log1 - log0): {time_offset:.6f} 秒")
    print(f"  对齐后的时间范围: [{aligned_start:.3f}s, {aligned_end:.3f}s]")
    print(f"  对齐后的总时长: {aligned_duration:.2f} 秒")
    print(f"  log0 对齐数据范围: [{idx0_start}, {idx0_end}] ({idx0_end - idx0_start} 个点)")
    print(f"  log1 对齐数据范围: [{idx1_start}, {idx1_end}] ({idx1_end - idx1_start} 个点)")

        
def _plot_position_response_dual(pl0: FlightLogPlotter, pl1: FlightLogPlotter):
    """双行位置响应图：上排 log0，下排 log1（各 1x3）。使用对齐后的数据范围。"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Position Response (Top: _0, Bottom: _1)', fontsize=14, fontweight='bold')

    axes_labels = ['X Axis', 'Y Axis', 'Z Axis']
    pos_keys = ['x', 'y', 'z']
    des_keys = ['x_des', 'y_des', 'z_des']
    
    # 获取对齐数据范围
    idx0_start, idx0_end = pl0.aligned_indices
    idx1_start, idx1_end = pl1.aligned_indices

    # 上排：_0（仅绘制对齐范围内的数据）
    for i, (ax, label, pos_key, des_key) in enumerate(zip(axes[0], axes_labels, pos_keys, des_keys)):
        time_slice = pl0.data['time'][idx0_start:idx0_end]
        ax.plot(time_slice, pl0.data[pos_key][idx0_start:idx0_end], 'b-', linewidth=2, label='Actual')
        ax.plot(time_slice, pl0.data[des_key][idx0_start:idx0_end], 'r--', linewidth=2, label='Desired')
        ax.set_ylabel('Position (m)', fontsize=11)
        ax.set_title(label + ' (log0)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    # 下排：_1（仅绘制对齐范围内的数据）
    for i, (ax, label, pos_key, des_key) in enumerate(zip(axes[1], axes_labels, pos_keys, des_keys)):
        time_slice = pl1.data['time'][idx1_start:idx1_end]
        ax.plot(time_slice, pl1.data[pos_key][idx1_start:idx1_end], 'b-', linewidth=2, label='Actual')
        ax.plot(time_slice, pl1.data[des_key][idx1_start:idx1_end], 'r--', linewidth=2, label='Desired')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Position (m)', fontsize=11)
        ax.set_title(label + ' (log1)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    return fig


def _plot_tracking_error_dual(pl0: FlightLogPlotter, pl1: FlightLogPlotter):
    """双行跟踪误差图。使用对齐后的数据范围。"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Tracking Error (Top: _0, Bottom: _1)', fontsize=14, fontweight='bold')

    axes_labels = ['X Error', 'Y Error', 'Z Error']
    error_keys = ['error_x', 'error_y', 'error_z']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 获取对齐数据范围
    idx0_start, idx0_end = pl0.aligned_indices
    idx1_start, idx1_end = pl1.aligned_indices

    # 上排：_0
    for i, (ax, label, error_key, color) in enumerate(zip(axes[0], axes_labels, error_keys, colors)):
        time_slice = pl0.data['time'][idx0_start:idx0_end]
        error_slice = pl0.data[error_key][idx0_start:idx0_end]
        ax.plot(time_slice, error_slice, color=color, linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('Error (m)', fontsize=11)
        ax.set_title(label + ' (log0)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        mean_error = np.mean(np.abs(error_slice))
        max_error = np.max(np.abs(error_slice))
        ax.text(0.02, 0.98, f'Mean: {mean_error:.3f}m\nMax: {max_error:.3f}m',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

    # 下排：_1
    for i, (ax, label, error_key, color) in enumerate(zip(axes[1], axes_labels, error_keys, colors)):
        time_slice = pl1.data['time'][idx1_start:idx1_end]
        error_slice = pl1.data[error_key][idx1_start:idx1_end]
        ax.plot(time_slice, error_slice, color=color, linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Error (m)', fontsize=11)
        ax.set_title(label + ' (log1)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        mean_error = np.mean(np.abs(error_slice))
        max_error = np.max(np.abs(error_slice))
        ax.text(0.02, 0.98, f'Mean: {mean_error:.3f}m\nMax: {max_error:.3f}m',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

    plt.tight_layout()
    return fig


def _plot_attitude_dual(pl0: FlightLogPlotter, pl1: FlightLogPlotter):
    """双行姿态角图。使用对齐后的数据范围。"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Attitude Angles (Top: _0, Bottom: _1)', fontsize=14, fontweight='bold')

    axes_labels = ['Roll', 'Pitch', 'Yaw']
    attitude_keys = ['roll', 'pitch', 'yaw']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 获取对齐数据范围
    idx0_start, idx0_end = pl0.aligned_indices
    idx1_start, idx1_end = pl1.aligned_indices

    # 上排：_0
    for i, (ax, label, att_key, color) in enumerate(zip(axes[0], axes_labels, attitude_keys, colors)):
        time_slice = pl0.data['time'][idx0_start:idx0_end]
        angle_slice = pl0.data[att_key][idx0_start:idx0_end]
        ax.plot(time_slice, angle_slice, color=color, linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('Angle (deg)', fontsize=11)
        ax.set_title(label + ' (log0)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        mean_angle = np.mean(np.abs(angle_slice))
        max_angle = np.max(np.abs(angle_slice))
        ax.text(0.02, 0.98, f'Mean: {mean_angle:.2f}°\nMax: {max_angle:.2f}°',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

    # 下排：_1
    for i, (ax, label, att_key, color) in enumerate(zip(axes[1], axes_labels, attitude_keys, colors)):
        time_slice = pl1.data['time'][idx1_start:idx1_end]
        angle_slice = pl1.data[att_key][idx1_start:idx1_end]
        ax.plot(time_slice, angle_slice, color=color, linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Angle (deg)', fontsize=11)
        ax.set_title(label + ' (log1)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        mean_angle = np.mean(np.abs(angle_slice))
        max_angle = np.max(np.abs(angle_slice))
        ax.text(0.02, 0.98, f'Mean: {mean_angle:.2f}°\nMax: {max_angle:.2f}°',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

    plt.tight_layout()
    return fig


def _set_equal_3d(ax, x, y, z):
    max_range = np.array([
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def _plot_aero_force_dual(pl0: FlightLogPlotter, pl1: FlightLogPlotter):
    """
    绘制气动扰动力响应曲线（仅当日志包含 Fa_z 时）
    布局：1行2列，左图为 log0，右图为 log1
    使用对齐后的数据范围。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Aerodynamic Force Response', fontsize=14, fontweight='bold')

    # 将 Fa_z 从牛顿（N）转换为克力（gf）: 1 N = 1000/9.8 gf ≈ 102.04 gf
    conversion_factor = 1000.0 / 9.8
    
    # 获取对齐数据范围
    idx0_start, idx0_end = pl0.aligned_indices
    idx1_start, idx1_end = pl1.aligned_indices

    # 左图：log0
    ax = axes[0]
    time_slice0 = pl0.data['time'][idx0_start:idx0_end]
    Fa_z_gf_0 = pl0.data['Fa_z'][idx0_start:idx0_end] * conversion_factor
    ax.plot(time_slice0, Fa_z_gf_0, 'b-', linewidth=2, label='Estimated Fa_z')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Fa_z (gf)', fontsize=11)
    ax.set_title('Aerodynamic Force (log0)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    mean_fa = np.mean(np.abs(Fa_z_gf_0))
    max_fa = np.max(np.abs(Fa_z_gf_0))
    ax.text(0.02, 0.98, f'Fa_z Mean: {mean_fa:.2f}gf\nFa_z Max: {max_fa:.2f}gf',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

    # 右图：log1
    ax = axes[1]
    time_slice1 = pl1.data['time'][idx1_start:idx1_end]
    Fa_z_gf_1 = pl1.data['Fa_z'][idx1_start:idx1_end] * conversion_factor
    ax.plot(time_slice1, Fa_z_gf_1, 'b-', linewidth=2, label='Estimated Fa_z')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Fa_z (gf)', fontsize=11)
    ax.set_title('Aerodynamic Force (log1)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    mean_fa = np.mean(np.abs(Fa_z_gf_1))
    max_fa = np.max(np.abs(Fa_z_gf_1))
    ax.text(0.02, 0.98, f'Fa_z Mean: {mean_fa:.2f}gf\nFa_z Max: {max_fa:.2f}gf',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

    plt.tight_layout()
    return fig


def _plot_trajectory_dual(pl0: FlightLogPlotter, pl1: FlightLogPlotter):
    """双行轨迹图：每行包含 3D 轨迹和 XY 平面轨迹。使用对齐后的数据范围。"""
    fig = plt.figure(figsize=(14, 10))
    
    # 获取对齐数据范围
    idx0_start, idx0_end = pl0.aligned_indices
    idx1_start, idx1_end = pl1.aligned_indices
    
    # 对齐后的数据切片
    x0_aligned = pl0.data['x'][idx0_start:idx0_end]
    y0_aligned = pl0.data['y'][idx0_start:idx0_end]
    z0_aligned = pl0.data['z'][idx0_start:idx0_end]
    x0_des_aligned = pl0.data['x_des'][idx0_start:idx0_end]
    y0_des_aligned = pl0.data['y_des'][idx0_start:idx0_end]
    z0_des_aligned = pl0.data['z_des'][idx0_start:idx0_end]
    
    x1_aligned = pl1.data['x'][idx1_start:idx1_end]
    y1_aligned = pl1.data['y'][idx1_start:idx1_end]
    z1_aligned = pl1.data['z'][idx1_start:idx1_end]
    x1_des_aligned = pl1.data['x_des'][idx1_start:idx1_end]
    y1_des_aligned = pl1.data['y_des'][idx1_start:idx1_end]
    z1_des_aligned = pl1.data['z_des'][idx1_start:idx1_end]

    # 顶行（log0）
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    # 底行（log1）
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224)

    # log0 3D
    ax1.plot(x0_aligned, y0_aligned, z0_aligned, 'b-', linewidth=2, label='Actual', alpha=0.8)
    ax1.plot(x0_des_aligned, y0_des_aligned, z0_des_aligned, 'r--', linewidth=2, label='Desired', alpha=0.8)
    ax1.scatter(x0_aligned[0], y0_aligned[0], z0_aligned[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(x0_aligned[-1], y0_aligned[-1], z0_aligned[-1], c='red', s=100, marker='s', label='End', zorder=5)
    ax1.set_xlabel('X (m)', fontsize=11); ax1.set_ylabel('Y (m)', fontsize=11); ax1.set_zlabel('Z (m)', fontsize=11)
    ax1.set_title('3D Trajectory (log0)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9); ax1.grid(True, alpha=0.3)
    _set_equal_3d(ax1, x0_aligned, y0_aligned, z0_aligned)

    # log0 XY
    ax2.plot(x0_aligned, y0_aligned, 'b-', linewidth=2, label='Actual', alpha=0.8)
    ax2.plot(x0_des_aligned, y0_des_aligned, 'r--', linewidth=2, label='Desired', alpha=0.8)
    ax2.scatter(x0_aligned[0], y0_aligned[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax2.scatter(x0_aligned[-1], y0_aligned[-1], c='red', s=100, marker='s', label='End', zorder=5)
    ax2.set_xlabel('X (m)', fontsize=11); ax2.set_ylabel('Y (m)', fontsize=11)
    ax2.set_title('XY Plane Trajectory (log0)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9); ax2.grid(True, alpha=0.3); ax2.axis('equal')

    # log1 3D
    ax3.plot(x1_aligned, y1_aligned, z1_aligned, 'b-', linewidth=2, label='Actual', alpha=0.8)
    ax3.plot(x1_des_aligned, y1_des_aligned, z1_des_aligned, 'r--', linewidth=2, label='Desired', alpha=0.8)
    ax3.scatter(x1_aligned[0], y1_aligned[0], z1_aligned[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax3.scatter(x1_aligned[-1], y1_aligned[-1], z1_aligned[-1], c='red', s=100, marker='s', label='End', zorder=5)
    ax3.set_xlabel('X (m)', fontsize=11); ax3.set_ylabel('Y (m)', fontsize=11); ax3.set_zlabel('Z (m)', fontsize=11)
    ax3.set_title('3D Trajectory (log1)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9); ax3.grid(True, alpha=0.3)
    _set_equal_3d(ax3, x1_aligned, y1_aligned, z1_aligned)

    # log1 XY
    ax4.plot(x1_aligned, y1_aligned, 'b-', linewidth=2, label='Actual', alpha=0.8)
    ax4.plot(x1_des_aligned, y1_des_aligned, 'r--', linewidth=2, label='Desired', alpha=0.8)
    ax4.scatter(x1_aligned[0], y1_aligned[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax4.scatter(x1_aligned[-1], y1_aligned[-1], c='red', s=100, marker='s', label='End', zorder=5)
    ax4.set_xlabel('X (m)', fontsize=11); ax4.set_ylabel('Y (m)', fontsize=11)
    ax4.set_title('XY Plane Trajectory (log1)', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9); ax4.grid(True, alpha=0.3); ax4.axis('equal')

    plt.tight_layout()
    return fig


def plot_all_dual(plotter0: FlightLogPlotter, plotter1: FlightLogPlotter, save_dir=None):
    """针对两份日志，绘制并可选保存全部图表。根据时间戳进行对齐。"""
    for pl in (plotter0, plotter1):
        if pl.data is None:
            pl.load_data()

    # 根据时间戳对齐两个日志
    _align_timestamps_dual(plotter0, plotter1)

    # 使用英文标签，避免中文字体问题
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    print("\nGenerating dual plots (time-aligned)...")

    # 1. 位置响应曲线（双行）
    print("1. Plotting dual position response...")
    fig1 = _plot_position_response_dual(plotter0, plotter1)

    # 2. 跟踪误差曲线（双行）
    print("2. Plotting dual tracking error...")
    fig2 = _plot_tracking_error_dual(plotter0, plotter1)

    # 3. 姿态角曲线（双行）
    print("3. Plotting dual attitude angles...")
    fig3 = _plot_attitude_dual(plotter0, plotter1)

    # 4. 轨迹曲线（双行：每行 3D+XY）
    print("4. Plotting dual trajectories...")
    fig4 = _plot_trajectory_dual(plotter0, plotter1)

    # 5. 气动扰动力曲线（仅当日志包含相关数据时）
    fig5 = None
    if plotter0.has_aero_data and plotter1.has_aero_data:
        print("5. Plotting aerodynamic force response...")
        fig5 = _plot_aero_force_dual(plotter0, plotter1)
    else:
        print("5. Skipping aerodynamic force plot (no Fa_z data in logs)")

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)

        stem0 = plotter0.log_file_path.stem
        stem1 = plotter1.log_file_path.stem

        # 尝试提取共同前缀（去掉最后的 _0/_1）
        def common_base(s0, s1):
            base0 = s0.rsplit('_', 1)[0]
            base1 = s1.rsplit('_', 1)[0]
            return base0 if base0 == base1 else f"{s0}__{s1}"

        base = common_base(stem0, stem1)

        fig1_path = save_path / f'{base}_position_response_dual.png'
        fig2_path = save_path / f'{base}_tracking_error_dual.png'
        fig3_path = save_path / f'{base}_attitude_dual.png'
        fig4_path = save_path / f'{base}_trajectory_dual.png'

        print(f"\nSaving plots to: {save_path}")
        fig1.savefig(fig1_path, dpi=300, bbox_inches='tight'); print(f"  - {fig1_path.name}")
        fig2.savefig(fig2_path, dpi=300, bbox_inches='tight'); print(f"  - {fig2_path.name}")
        fig3.savefig(fig3_path, dpi=300, bbox_inches='tight'); print(f"  - {fig3_path.name}")
        fig4.savefig(fig4_path, dpi=300, bbox_inches='tight'); print(f"  - {fig4_path.name}")
        
        # 保存气动力图（如果有）
        if fig5 is not None:
            fig5_path = save_path / f'{base}_aero_force_dual.png'
            fig5.savefig(fig5_path, dpi=300, bbox_inches='tight'); print(f"  - {fig5_path.name}")

    print("\nAll dual plots generated successfully!")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Flight Log Plotting Tool')
    # 可选指定两份日志；若未指定则自动在 log 目录里选择最新一对 (_0 与 _1)
    parser.add_argument('--log0', type=str, default=None, help='Path to log file ending with _0.csv')
    parser.add_argument('--log1', type=str, default=None, help='Path to log file ending with _1.csv')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save plots (default: log/plots)')
    parser.add_argument('--list', action='store_true',
                       help='List all log files in log directory')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not show plot windows')
    
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
        # 收集以 _0 / _1 结尾的文件及其基名
        pair_map = {}
        for f in csvs:
            stem = f.stem  # e.g., 2025-11-20_1532_0
            if stem.endswith('_0') or stem.endswith('_1'):
                base = stem[:-2]  # 去掉后缀 _0/_1
                suffix = stem[-1]
                rec = pair_map.get(base, {'0': None, '1': None, 'mtime': 0.0})
                rec[suffix] = f
                # 记录该基名的最近修改时间（最大者）
                rec['mtime'] = max(rec['mtime'], f.stat().st_mtime)
                pair_map[base] = rec
        # 选择同时拥有 0 和 1 的最近的一对
        candidates = [ (base, rec) for base, rec in pair_map.items() if rec['0'] and rec['1'] ]
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[1]['mtime'])
        base, rec = candidates[-1]
        return rec['0'], rec['1']

    if args.log0 and args.log1:
        log0 = Path(args.log0)
        log1 = Path(args.log1)
        if not log0.exists() or not log1.exists():
            print("Error: --log0 or --log1 path does not exist.")
            sys.exit(1)
    else:
        if not log_dir.exists():
            print(f"Error: Log directory does not exist: {log_dir}")
            sys.exit(1)
        pair = find_latest_pair(log_dir)
        if not pair:
            print(f"Error: No matched latest pair (*_0.csv & *_1.csv) found in {log_dir}")
            print("Use --list to see available log files")
            sys.exit(1)
        log0, log1 = pair
        print(f"Using latest pair: {log0.name}  |  {log1.name}")
    
    try:
        # 创建绘图器并绘制
        plotter0 = FlightLogPlotter(log0)
        plotter1 = FlightLogPlotter(log1)

        # 如果没有指定保存目录，默认保存到 log 目录下的 plots 子目录
        save_dir = (log_dir / 'plots') if args.save_dir is None else Path(args.save_dir)

        # 设置不显示图形窗口
        if args.no_show:
            import matplotlib
            matplotlib.use('Agg')

        plot_all_dual(plotter0, plotter1, save_dir=save_dir)

        if not args.no_show:
            plt.show()
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
