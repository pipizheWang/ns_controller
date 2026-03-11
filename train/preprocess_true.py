import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from utils import interpolation_cubic, Fa, get_data

# ===== 数据处理函数 =====

# 1a. 读取下层飞机的完整CSV文件（含姿态、PWM等）
# 注意：下层飞机（nl.py）控制周期为 1/50Hz = 0.02s，日志采样率为 50Hz
def load_your_csv(filename):
    """读取下层飞机采集的CSV文件"""
    df = pd.read_csv(filename)
    
    # 转换为代码期望的格式
    # 保留绝对时间戳，用于与上层飞机日志做时间对齐
    Data = {
        'time': df['Time'].values,  # 绝对时间戳（秒）
        'pos': df[['Pos_x', 'Pos_y', 'Pos_z']].values,
        'vel': df[['Vel_x', 'Vel_y', 'Vel_z']].values,
        'acc': df[['Acc_x', 'Acc_y', 'Acc_z']].values,
        'qua': df[['Qua_x', 'Qua_y', 'Qua_z', 'Qua_w']].values,
        'pwm': df[['RPM_1', 'RPM_2', 'RPM_3', 'RPM_4']].values,  # 使用RPM（krpm）替代PWM
        'vol': np.zeros(len(df)),  # 无电压数据，置为0
        'tau_u': np.zeros((len(df), 3)),  # 力矩（未测量，初始化为0）
        'omega': np.zeros((len(df), 3)),  # 角速度（未测量，初始化为0）
    }
    return Data

# 1b. 读取上层飞机的简化CSV文件（仅含时间、位置、速度）
# 注意：上层飞机（EKFLogger）是订阅回调驱动，频率由 /cf_wz/odom 发布率决定（CrazyFlie 通常为 100Hz）
def load_upper_csv(filename):
    """读取上层飞机采集的简化CSV文件（列: timestamp, x, y, z, vx, vy, vz）"""
    df = pd.read_csv(filename)
    Data = {
        'time': df['timestamp'].values,  # 绝对时间戳（秒），用于时间对齐
        'pos': df[['x', 'y', 'z']].values.astype(np.float64),
        'vel': df[['vx', 'vy', 'vz']].values.astype(np.float64),
    }
    return Data

# 1c. 对上层飞机数据进行简化插值（仅插值 pos 和 vel）
def interpolation_simple(t0, t1, Data):
    """将上层飞机的 pos/vel 插值到 100Hz 均匀时间网格"""
    time = np.linspace(t0, t1, int(100 * (t1 - t0) + 1))
    pos = np.zeros((time.shape[0], 3))
    vel = np.zeros((time.shape[0], 3))

    x = Data['time']
    for i in range(3):
        pos[:, i] = interp1d(x, Data['pos'][:, i], kind='linear')(time)
        vel[:, i] = interp1d(x, Data['vel'][:, i], kind='linear')(time)

    return {'time': time, 'pos': pos, 'vel': vel}

# 2. 预处理两架无人机的数据
def preprocess_two_uav_data(uav1_csv, uav2_csv, save_path='./'):
    """
    预处理双机数据
    
    参数:
        uav1_csv: 下方飞机（受干扰）的CSV文件路径，需含完整飞控数据
        uav2_csv: 上方飞机（产生干扰）的CSV文件路径，仅含时间/位置/速度
        save_path: 保存处理后数据的路径
    """
    # 读取原始数据（上下层使用不同的加载函数）
    Data1_raw = load_your_csv(uav1_csv)
    Data2_raw = load_upper_csv(uav2_csv)
    
    # 插值到统一时间网格（100Hz）
    t0 = max(Data1_raw['time'][0], Data2_raw['time'][0])
    t1 = min(Data1_raw['time'][-1], Data2_raw['time'][-1])
    
    Data1_int = interpolation_cubic(t0, t1, Data1_raw, 0, len(Data1_raw['time']))
    Data2_int = interpolation_simple(t0, t1, Data2_raw)
    
    # 计算气动力（需要你的无人机质量和推力模型参数）
    g = 9.81
    m = 756.5 / 1000  # 千克
    
    # 推力拟合参数（基于RPM，单位：krpm -> g）
    # 模型: force(g) = ka*rpm^2 + kb*rpm + kc
    ka = 0.6612654
    kb = 1.0575438
    kc = 4.1059176
    # Fa() 内部期望推力单位为 N，将系数从 g 换算为 N（除以1000再乘g）
    p_0 = kc * g / 1000
    p_1 = kb * g / 1000
    p_2 = ka * g / 1000
    
    Data1 = Fa(Data1_int, m, g, p_0, p_1, p_2)
    # 上层飞机只需 pos/vel，无需计算气动力
    Data2 = Data2_int

    # 生成训练数据对（下方飞机受上方飞机气流干扰）
    data_input, data_output = get_data(D1=Data1, D2=Data2, typ='fa_imu')
    
    # 保存为numpy格式
    np.save(f'{save_path}/data_input_true.npy', data_input)
    np.save(f'{save_path}/data_output_true.npy', data_output)
    
    print(f'✓ 预处理完成!')
    print(f'  输入形状: {data_input.shape}')
    print(f'  输出形状: {data_output.shape}')
    print(f'  保存位置: {save_path}')
    
    return data_input, data_output

# 3. 使用示例
if __name__ == '__main__':
    # uav1_csv: 下方飞机（受扰动），需含完整飞控日志
    # uav2_csv: 上方飞机（产生扰动），只需含 timestamp/x/y/z/vx/vy/vz
    uav_lower_file = 'data/drone1.csv'  # drone1 在下面（完整日志）
    uav_upper_file = 'data/drone2.csv'  # drone2 在上面（简化日志）
    
    data_input, data_output = preprocess_two_uav_data(
        uav1_csv=uav_lower_file,   # D1: 下方飞机，输出其受到的气动力
        uav2_csv=uav_upper_file,   # D2: 上方飞机，计算相对位置和速度
        save_path='data'
    )