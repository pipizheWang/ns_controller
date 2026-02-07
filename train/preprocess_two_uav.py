import numpy as np
import pandas as pd
from utils import interpolation_cubic, Fa, get_data

# ===== 数据处理函数 =====

# 1. 读取你的CSV文件
def load_your_csv(filename):
    """读取你采集的CSV文件"""
    df = pd.read_csv(filename)
    
    # 转换为代码期望的格式
    Data = {
        'time': (df['tick'].values - df['tick'].values[0]) / 1e6,  # 转换为秒
        'pos': df[['pos_x', 'pos_y', 'pos_z']].values,
        'vel': df[['vel_x', 'vel_y', 'vel_z']].values,
        'acc': df[['acc_x', 'acc_y', 'acc_z']].values,
        'qua': df[['qua_x', 'qua_y', 'qua_z', 'qua_w']].values,
        'pwm': df[['pwm_1', 'pwm_2', 'pwm_3', 'pwm_4']].values,
        'vol': df['voltage'].values,
        'tau_u': np.zeros((len(df), 3)),  # 力矩（未测量，初始化为0）
        'omega': np.zeros((len(df), 3)),  # 角速度（未测量，初始化为0）
    }
    return Data

# 2. 预处理两架无人机的数据
def preprocess_two_uav_data(uav1_csv, uav2_csv, save_path='./'):
    """
    预处理双机数据
    
    参数:
        uav1_csv: 下方飞机（受干扰）的CSV文件路径
        uav2_csv: 上方飞机（产生干扰）的CSV文件路径
        save_path: 保存处理后数据的路径
    """
    # 读取原始数据
    Data1_raw = load_your_csv(uav1_csv)
    Data2_raw = load_your_csv(uav2_csv)
    
    # 插值到统一时间网格（100Hz）
    t0 = max(Data1_raw['time'][0], Data2_raw['time'][0])
    t1 = min(Data1_raw['time'][-1], Data2_raw['time'][-1])
    
    Data1_int = interpolation_cubic(t0, t1, Data1_raw, 0, len(Data1_raw['time']))
    Data2_int = interpolation_cubic(t0, t1, Data2_raw, 0, len(Data2_raw['time']))
    
    # 计算气动力（需要你的无人机质量和推力模型参数）
    g = 9.81
    m = 2.0  # 千克
    
    # 推力拟合参数（单位：N）
    # 模型: force = p_0 + p_1*pwm + p_2*pwm^2
    p_0 = -184.900182
    p_1 = 0.472594
    p_2 = -2.935641e-04
    
    Data1 = Fa(Data1_int, m, g, p_0, p_1, p_2)
    Data2 = Fa(Data2_int, m, g, p_0, p_1, p_2)
    
    # 生成xy训练数据对（下方飞机受上方飞机干扰）
    data_input, data_output = get_data(D1=Data1, D2=Data2, typ='fa_imu')
    
    # 保存为numpy格式
    np.save(f'{save_path}/data_input_xy.npy', data_input)
    np.save(f'{save_path}/data_output_xy.npy', data_output)
    
    print(f'✓ 预处理完成!')
    print(f'  输入形状: {data_input.shape}')
    print(f'  输出形状: {data_output.shape}')
    print(f'  保存位置: {save_path}')
    
    return data_input, data_output

# 3. 使用示例
if __name__ == '__main__':
    # 注意：uav1_csv是下方飞机（受扰动的飞机），uav2_csv是上方飞机（产生扰动的飞机）
    uav_lower_file = 'data/uav0_xy.csv'  # UAV0在下面
    uav_upper_file = 'data/uav1_xy.csv'  # UAV1在上面
    
    data_input, data_output = preprocess_two_uav_data(
        uav1_csv=uav_lower_file,   # D1: 下方飞机，输出其受到的气动力
        uav2_csv=uav_upper_file,   # D2: 上方飞机，计算相对位置
        save_path='data'
    )