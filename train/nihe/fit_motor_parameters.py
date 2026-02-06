#!/usr/bin/env python3
"""
拟合电机推力模型参数
模型: force = p_0 + p_1*pwm + p_2*pwm^2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from pathlib import Path
import platform

# 配置Mac中文字体
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置参数
MASS = 2.0  # kg
GRAVITY = 9.81  # m/s^2
NUM_MOTORS = 4
ACC_RATE = 0.1  # 加速度变化率: a_z = 0.1 * t (m/s^3)

def load_data(csv_file):
    """加载CSV数据"""
    df = pd.read_csv(csv_file)
    return df

def prepare_data(df):
    """
    准备拟合数据
    将所有4个电机的数据合并到一起
    根据轨迹时间计算每个时刻的推力: F = m*(g + 0.1*t)
    """
    # 只保留前405帧数据
    df = df.iloc[:405].copy()
    print(f"使用前405帧数据进行拟合（已过滤后续错误数据）")
    
    pwm_data = []
    vol_data = []
    force_data = []
    
    # 计算相对时间（从实验开始计时）
    traj_time = df['Traj_time'].values
    t_relative = traj_time - traj_time[0]  # 相对于第一个数据点的时间
    
    # 提取每个电机的PWM和电压数据
    for i in range(NUM_MOTORS):
        pwm_col = f'PWM_{i}'
        if pwm_col in df.columns:
            pwm = df[pwm_col].values
            vol = df['Battery_Voltage'].values
            
            # 计算每个时刻的加速度和总推力
            # a_z(t) = 0.1 * t (t从0开始)
            # F_total(t) = m * (g + a_z) = m * (g + 0.1*t)
            # F_per_motor(t) = F_total(t) / 4
            a_z = ACC_RATE * t_relative
            F_total = MASS * (GRAVITY + a_z)
            force = F_total / NUM_MOTORS
            
            pwm_data.append(pwm)
            vol_data.append(vol)
            force_data.append(force)
    
    # 合并所有电机的数据
    pwm_all = np.concatenate(pwm_data)
    vol_all = np.concatenate(vol_data)
    force_all = np.concatenate(force_data)
    
    return pwm_all, vol_all, force_all

def create_design_matrix(pwm):
    """
    创建设计矩阵 X
    force = p_0 + p_1*pwm + p_2*pwm^2
    X = [1, pwm, pwm^2]
    """
    n = len(pwm)
    X = np.zeros((n, 3))
    X[:, 0] = 1.0          # p_0 (常数项)
    X[:, 1] = pwm          # p_1 (pwm)
    X[:, 2] = pwm ** 2     # p_2 (pwm^2)
    
    return X

def fit_parameters(pwm, force):
    """
    使用最小二乘法拟合参数
    """
    X = create_design_matrix(pwm)
    
    # 最小二乘法: X @ params = force
    params, residuals, rank, s = np.linalg.lstsq(X, force, rcond=None)
    
    # 计算拟合误差
    force_pred = X @ params
    mse = np.mean((force - force_pred) ** 2)
    rmse = np.sqrt(mse)
    
    return params, rmse, force_pred

def plot_results(pwm, force_true, force_pred):
    """绘制拟合结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 真实力 vs 预测力
    ax = axes[0, 0]
    ax.scatter(force_true, force_pred, alpha=0.5, s=1)
    ax.plot([force_true.min(), force_true.max()], 
            [force_true.min(), force_true.max()], 
            'r--', lw=2, label='理想拟合')
    ax.set_xlabel('真实推力 (N)')
    ax.set_ylabel('预测推力 (N)')
    ax.set_title('真实 vs 预测推力')
    ax.legend()
    ax.grid(True)
    
    # 2. 残差分布
    ax = axes[0, 1]
    residuals = force_true - force_pred
    ax.hist(residuals, bins=50, edgecolor='black')
    ax.set_xlabel('残差 (N)')
    ax.set_ylabel('频数')
    ax.set_title(f'残差分布 (均值: {np.mean(residuals):.4f} N)')
    ax.grid(True)
    
    # 3. PWM vs 推力
    ax = axes[1, 0]
    ax.scatter(pwm, force_true, alpha=0.3, s=1, label='真实值', color='blue')
    ax.scatter(pwm, force_pred, alpha=0.3, s=1, label='拟合值', color='red')
    ax.set_xlabel('PWM')
    ax.set_ylabel('推力 (N)')
    ax.set_title('PWM vs 推力')
    ax.legend()
    ax.grid(True)
    
    # 4. 残差 vs PWM
    ax = axes[1, 1]
    ax.scatter(pwm, residuals, alpha=0.5, s=1)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('PWM')
    ax.set_ylabel('残差 (N)')
    ax.set_title('残差 vs PWM')
    ax.grid(True)
    
    plt.tight_layout()
    return fig

def main():
    # 获取CSV文件路径
    csv_file = Path(__file__).parent / 'trajectory_log_2026-02-04_22-40-49.csv'
    
    print(f"读取数据文件: {csv_file}")
    df = load_data(csv_file)
    print(f"数据行数: {len(df)}")
    
    # 准备数据
    print("\n准备拟合数据...")
    pwm, vol, force = prepare_data(df)
    print(f"总样本数: {len(pwm)} (每个电机 {len(pwm)//NUM_MOTORS} 个样本)")
    print(f"PWM 范围: [{pwm.min():.0f}, {pwm.max():.0f}]")
    print(f"电压 范围: [{vol.min():.2f}, {vol.max():.2f}] V")
    print(f"推力 范围: [{force.min():.4f}, {force.max():.4f}] N")
    print(f"原始时间范围: [{df['Traj_time'].min():.2f}, {df['Traj_time'].max():.2f}] s")
    print(f"相对时间范围: [0.00, {df['Traj_time'].max()-df['Traj_time'].min():.2f}] s")
    
    # 拟合参数
    print("\n拟合参数...")
    params, rmse, force_pred = fit_parameters(pwm, force)
    
    # 输出结果
    print("\n" + "="*60)
    print("拟合结果:")
    print("="*60)
    print(f"p_0 (常数项)         : {params[0]:12.6f}")
    print(f"p_1 (pwm系数)        : {params[1]:12.6f}")
    print(f"p_2 (pwm^2系数)      : {params[2]:12.6e}")
    print("="*60)
    print(f"RMSE: {rmse:.6f} N")
    avg_force = force.mean()
    print(f"平均推力: {avg_force:.4f} N")
    print(f"相对误差: {rmse/avg_force*100:.2f}%")
    print("="*60)
    
    # 保存参数到文件
    output_file = Path(__file__).parent / 'motor_parameters.txt'
    with open(output_file, 'w') as f:
        f.write("电机推力模型参数\n")
        f.write("="*60 + "\n")
        f.write("模型: force = p_0 + p_1*pwm + p_2*pwm^2\n")
        f.write("="*60 + "\n")
        f.write(f"p_0 = {params[0]:.6f}\n")
        f.write(f"p_1 = {params[1]:.6f}\n")
        f.write(f"p_2 = {params[2]:.6e}\n")
        f.write("="*60 + "\n")
        f.write(f"RMSE: {rmse:.6f} N\n")
        f.write(f"平均推力: {avg_force:.4f} N\n")
        f.write(f"相对误差: {rmse/avg_force*100:.2f}%\n")
    print(f"\n参数已保存到: {output_file}")
    
    # 绘制结果
    print("\n绘制拟合结果...")
    fig = plot_results(pwm, force, force_pred)
    plot_file = Path(__file__).parent / 'motor_fit_results.png'
    fig.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {plot_file}")
    
    # 显示图表
    plt.show()
    
    # 测试几个样本点
    print("\n" + "="*60)
    print("样本预测测试 (前5个数据点的第0号电机):")
    print("="*60)
    for i in range(min(5, len(df))):
        pwm_test = df['PWM_0'].iloc[i]
        X_test = create_design_matrix(np.array([pwm_test]))
        force_test = (X_test @ params)[0]
        print(f"PWM={pwm_test:.0f} -> Force={force_test:.4f}N")

if __name__ == '__main__':
    main()
