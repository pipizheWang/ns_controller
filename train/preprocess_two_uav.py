import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import signal

# Convert quaternion to rotation matrix
def rotation_matrix(quat):
    rot_mat = np.ones([3,3])
    a = quat[0]**2
    b = quat[1]**2
    c = quat[2]**2
    d = quat[3]**2
    e = quat[0]*quat[1]
    f = quat[0]*quat[2]
    g = quat[0]*quat[3]
    h = quat[1]*quat[2]
    i = quat[1]*quat[3]
    j = quat[2]*quat[3]
    rot_mat[0,0] = a - b - c + d
    rot_mat[0,1] = 2 * (e - j)
    rot_mat[0,2] = 2 * (f + i)
    rot_mat[1,0] = 2 * (e + j)
    rot_mat[1,1] = -a + b - c + d
    rot_mat[1,2] = 2 * (h - g)
    rot_mat[2,0] = 2 * (f - i)
    rot_mat[2,1] = 2 * (h + g)
    rot_mat[2,2] = -a - b + c + d
    return rot_mat

# Convert quaternion to Euler angle
def qua2euler(qua):
    euler = np.zeros(3)
    q0 = qua[3]
    q1 = qua[0]
    q2 = qua[1]
    q3 = qua[2]
    euler[0] = np.degrees(np.arctan2(2*(q0*q1+q2*q3), 1-2*(q1**2+q2**2)))
    euler[1] = np.degrees(np.arcsin(2*(q0*q2-q3*q1)))
    euler[2] = np.degrees(np.arctan2(2*(q0*q3+q1*q2), 1-2*(q2**2+q3**2)))
    return euler

# cubic (or linear) interpolation
def cubic(x, y, xnew, kind='linear'):
    f = interp1d(x, y, kind=kind)
    return f(xnew)

# data interpolation
def interpolation_cubic(t0, t1, Data, ss, ee):
    time = np.linspace(t0, t1, int(100*(t1-t0)+1))
    pos = np.zeros((time.shape[0], 3))
    qua = np.zeros((time.shape[0], 4))
    pwm = np.zeros((time.shape[0], 4))
    vel = np.zeros((time.shape[0], 3))
    vel_num = np.zeros((time.shape[0], 3))
    acc_num = np.zeros((time.shape[0], 3))
    vol = np.zeros((time.shape[0]))
    acc_imu = np.zeros((time.shape[0], 3))
    tau_u = np.zeros((time.shape[0], 3))
    omega = np.zeros((time.shape[0], 3))
    omega_dot = np.zeros((time.shape[0], 3))
    euler = np.zeros((time.shape[0], 3))
    acc_filter = np.zeros((time.shape[0], 3))
    acc_smooth = np.zeros((time.shape[0], 3))
    
    x = Data['time'][ss:ee]
    for i in range(3):
        pos[:, i] = cubic(x, Data['pos'][ss:ee, i], time)
        vel[:, i] = cubic(x, Data['vel'][ss:ee, i], time)
        acc_imu[:, i] = cubic(x, Data['acc'][ss:ee, i], time)
        tau_u[:, i] = cubic(x, Data['tau_u'][ss:ee, i], time)
        omega[:, i] = cubic(x, Data['omega'][ss:ee, i], time)
 
    for i in range(3):
        acc_num[2:-2,i] = (-vel[4:,i] + 8 * vel[3:-1,i] - 8 * vel[1:-3,i] + vel[:-4,i]) / 12 * 100
        vel_num[2:-2,i] = (-pos[4:,i] + 8 * pos[3:-1,i] - 8 * pos[1:-3,i] + pos[:-4,i]) / 12 * 100        
        omega_dot[2:-2,i] = (-omega[4:,i] + 8 * omega[3:-1,i] - 8 * omega[1:-3,i] + omega[:-4,i]) / 12 * 100
    
    for i in range(4):
        qua[:, i] = cubic(x, Data['qua'][ss:ee, i], time)
        pwm[:, i] = cubic(x, Data['pwm'][ss:ee, i], time)
    vol[:] = cubic(x, Data['vol'][ss:ee], time)
    
    for j in range(time.shape[0]):
        euler[j, :] = qua2euler(qua[j, :])
    
    # Filter on acc
    b, a = signal.butter(1, 0.1)
    for i in range(3):
        acc_filter[:, i] = signal.filtfilt(b, a, acc_num[:, i])
    
    # Moving average smoothing
    n = 5
    l = int((n-1) / 2)
    for i in range(3):
        for j in range(n):
            if j == n-1:
                temp = acc_num[j:, i]
            else:
                temp = acc_num[j:-(n-1-j), i]
            acc_smooth[l:-l, i] = acc_smooth[l:-l, i] + temp
        acc_smooth[l:-l, i] = acc_smooth[l:-l, i] / n
        
    Data_int = {'time': time, 'pos': pos, 'vel': vel, 'acc_imu': acc_imu, 'vel_num': vel_num, \
                'qua': qua, 'pwm': pwm, 'vol': vol, 'acc_num': acc_num, 'euler': euler, \
               'tau_u': tau_u, 'omega': omega, 'omega_dot': omega_dot, 'acc_filter': acc_filter, 'acc_smooth': acc_smooth}
    return Data_int

# Compute Fa
def Fa(Data, m, g, p_00, p_10, p_01, p_20, p_11):
    R = np.zeros([Data['time'].shape[0], 3, 3])
    for i in range(Data['time'].shape[0]):
        R[i, :, :] = rotation_matrix(Data['qua'][i, :])
        
    force_pwm_1 = p_00 + p_10 * Data['pwm'][:, 0] + p_01 * Data['vol'] + p_20 * Data['pwm'][:, 0]**2 + p_11 * Data['vol'] * Data['pwm'][:, 0]
    force_pwm_2 = p_00 + p_10 * Data['pwm'][:, 1] + p_01 * Data['vol'] + p_20 * Data['pwm'][:, 1]**2 + p_11 * Data['vol'] * Data['pwm'][:, 1]
    force_pwm_3 = p_00 + p_10 * Data['pwm'][:, 2] + p_01 * Data['vol'] + p_20 * Data['pwm'][:, 2]**2 + p_11 * Data['vol'] * Data['pwm'][:, 2]
    force_pwm_4 = p_00 + p_10 * Data['pwm'][:, 3] + p_01 * Data['vol'] + p_20 * Data['pwm'][:, 3]**2 + p_11 * Data['vol'] * Data['pwm'][:, 3]
    thrust_pwm = force_pwm_1 + force_pwm_2 + force_pwm_3 + force_pwm_4 # gram
    # consider delay
    thrust_pwm_delay = np.zeros(len(thrust_pwm))
    thrust_pwm_delay[0] = thrust_pwm[0]
    for i in range(len(thrust_pwm)-1):
        thrust_pwm_delay[i+1] = (1-0.16)*thrust_pwm_delay[i] + 0.16*thrust_pwm[i] 

    Fa = np.zeros([Data['time'].shape[0], 3])
    Fa_delay = np.zeros([Data['time'].shape[0], 3])
    Fa_num = np.zeros([Data['time'].shape[0], 3])
    Fa_filter = np.zeros([Data['time'].shape[0], 3])
    Fa_smooth = np.zeros([Data['time'].shape[0], 3])
    force_world = np.zeros([Data['time'].shape[0], 3])
    for i in range(Data['time'].shape[0]):
        Fa[i, :] = m * Data['acc_imu'][i, :] / 1000 - thrust_pwm[i] / 1000 * g * R[i, :, 2] # Newton
        Fa_delay[i, :] = m * Data['acc_imu'][i, :] / 1000 - thrust_pwm_delay[i] / 1000 * g * R[i, :, 2] # Newton
        Fa_num[i, :] = m * np.array([0, 0, g]) / 1000 + m * Data['acc_num'][i, :] / 1000 - thrust_pwm[i] / 1000 * g * R[i, :, 2] # Newton
        Fa_filter[i, :] = m * np.array([0, 0, g]) / 1000 + m * Data['acc_filter'][i, :] / 1000 - thrust_pwm[i] / 1000 * g * R[i, :, 2] # Newton
        Fa_smooth[i, :] = m * np.array([0, 0, g]) / 1000 + m * Data['acc_smooth'][i, :] / 1000 - thrust_pwm[i] / 1000 * g * R[i, :, 2] # Newton
        force_world[i, :] = thrust_pwm[i] / 1000 * g * R[i, :, 2] # Newton
    
    Data['fa_imu'] = Fa
    Data['fa_delay'] = Fa_delay
    Data['fa_num'] = Fa_num
    Data['fa_filter'] = Fa_filter
    Data['fa_smooth'] = Fa_smooth
    
    return Data

# Get numpy data input and output pair for L2L scenario
def get_data(D1, D2, typ='fa_delay'):

    g = 9.81
    L = D1['time'].shape[0]
    
    # 13维输入: [相对位置(3), 相对速度(3), 保留(6), 场景编码(1)]
    data_input = np.zeros([L, 13], dtype=np.float32)
    data_output = np.zeros([L, 3], dtype=np.float32)
    
    # 计算相对位置和速度
    data_input[:, :3] = D2['pos'] - D1['pos']
    data_input[:, 3:6] = D2['vel'] - D1['vel']
    # data_input[:, 6:12] 保持为0（预留给三机场景）
    data_input[:, -1] = 2  # L2L场景编码
    
    # 输出: D1受到的气动力 (Newton -> gram)
    data_output[:, :] = D1[typ] / g * 1000
    
    return data_input, data_output

# ===== 原有函数 =====

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
    m = 2000  # 克
    factor = 1000 / g  # 单位转换系数：N -> gram (utils.py会做 /1000*g 转回N)
    
    # 原始拟合参数（单位：N）
    p_00_N = 354.716813
    p_10_N = -0.483469
    p_01_N = -19.170488
    p_20_N = 8.183327e-05
    p_11_N = 0.022760
    
    # 转换为utils.py期望的参数（单位：gram）
    C_00 = p_00_N * factor
    C_10 = p_10_N * factor
    C_01 = p_01_N * factor
    C_20 = p_20_N * factor
    C_11 = p_11_N * factor
    
    Data1 = Fa(Data1_int, m, g, C_00, C_10, C_01, C_20, C_11)
    Data2 = Fa(Data2_int, m, g, C_00, C_10, C_01, C_20, C_11)
    
    # 生成L2L训练数据对（下方飞机受上方飞机干扰）
    data_input, data_output = get_data(D1=Data1, D2=Data2, typ='fa_delay')
    
    # 保存为numpy格式
    np.save(f'{save_path}/data_input_L2L.npy', data_input)
    np.save(f'{save_path}/data_output_L2L.npy', data_output)
    
    print(f'✓ 预处理完成!')
    print(f'  输入形状: {data_input.shape}')
    print(f'  输出形状: {data_output.shape}')
    print(f'  保存位置: {save_path}')
    
    return data_input, data_output

# 3. 使用示例
if __name__ == '__main__':
    # 注意：uav1_csv是下方飞机（受扰动的飞机），uav2_csv是上方飞机（产生扰动的飞机）
    uav_lower_file = '/home/zhe/px4_ws/src/ns_controller/log/uav0_20260205_000450.csv'  # UAV0在下面
    uav_upper_file = '/home/zhe/px4_ws/src/ns_controller/log/uav1_20260205_000446.csv'  # UAV1在上面
    
    data_input, data_output = preprocess_two_uav_data(
        uav1_csv=uav_lower_file,   # D1: 下方飞机，输出其受到的气动力
        uav2_csv=uav_upper_file,   # D2: 上方飞机，计算相对位置
        save_path='/home/zhe/px4_ws/src/ns_controller/train/log/'
    )