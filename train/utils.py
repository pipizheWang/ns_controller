"""
通用工具函数模块
包含数据预处理、训练辅助等功能
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt
import random


# ==================== 数据预处理函数 ====================

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
def Fa(Data, m, g, p_0, p_1, p_2):
    R = np.zeros([Data['time'].shape[0], 3, 3])
    for i in range(Data['time'].shape[0]):
        R[i, :, :] = rotation_matrix(Data['qua'][i, :])
        
    force_pwm_1 = p_0 + p_1 * Data['pwm'][:, 0] + p_2 * Data['pwm'][:, 0]**2
    force_pwm_2 = p_0 + p_1 * Data['pwm'][:, 1] + p_2 * Data['pwm'][:, 1]**2
    force_pwm_3 = p_0 + p_1 * Data['pwm'][:, 2] + p_2 * Data['pwm'][:, 2]**2
    force_pwm_4 = p_0 + p_1 * Data['pwm'][:, 3] + p_2 * Data['pwm'][:, 3]**2
    thrust_pwm = force_pwm_1 + force_pwm_2 + force_pwm_3 + force_pwm_4 # N

    Fa = np.zeros([Data['time'].shape[0], 3])
    for i in range(Data['time'].shape[0]):
        Fa[i, :] = m * Data['acc_imu'][i, :] - thrust_pwm[i]  * R[i, :, 2] # Newton
    
    Data['fa_imu'] = Fa
    
    return Data

def get_data(D1, D2, typ='fa_imu'):

    g = 9.81
    L = D1['time'].shape[0]
    
    # 6维输入: [相对位置(3), 相对速度(3)]
    data_input = np.zeros([L, 6], dtype=np.float32)
    data_output = np.zeros([L, 1], dtype=np.float32)
    
    # 计算相对位置和速度
    data_input[:, :3] = D2['pos'] - D1['pos']
    data_input[:, 3:6] = D2['vel'] - D1['vel']
    
    # 输出: D1受到的气动力的第三列Faz (Newton -> gram)
    data_output[:, 0] = D1[typ][:, 2] / g * 1000
    
    return data_input, data_output

# ==================== 迁移的工具函数 ====================

# Dataset in torch
class MyDataset(Dataset):
    def __init__(self, inputs, outputs, type):
        self.inputs = inputs
        self.outputs = outputs
        self.type = type

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        Input = self.inputs[idx,]
        output = self.outputs[idx,]
        sample = {'input': Input, 'output': output, 'type': self.type}
        return sample


def split(data, type, val_num=10, num=50):
    """将数据分割为训练集和验证集"""
    # 50 pieces together, and 10 pieces for validation
    L = len(data)
    temp = np.split(data[:L-np.mod(L,num),:], num, axis=0)
    code = int(np.sum([ord(s) for s in type]))
    random.seed(code) # fix the seed for each scenario
    val_index = random.sample([i for i in range(num)], val_num)
    val_index_sorted = sorted(val_index)
    train_index_sorted = sorted([i for i in range(num) if i not in val_index])
    val_data = np.concatenate([temp[i] for i in val_index_sorted], axis=0)
    train_data = np.concatenate([temp[i] for i in train_index_sorted], axis=0)
    return val_data, train_data


def set_generate(data_input, data_output, type, device, batch_size):
    """生成训练集和验证集"""
    # 20% for validation
    val_input, data_input = split(data_input, type=type)
    val_output, data_output = split(data_output, type=type)

    Data_input = torch.from_numpy(data_input[:, :]).float().to(device) # 13维输入
    Data_output = torch.from_numpy(data_output[:, :]).float().to(device) # 3维输出 (Fax, Fay, Faz)
    Val_input = torch.from_numpy(val_input[:, :]).float().to(device)
    Val_output = torch.from_numpy(val_output[:, :]).float().to(device)
    trainset = MyDataset(Data_input, Data_output, type)
    valset = MyDataset(Val_input, Val_output, type)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainset, trainloader, valset, val_input, val_output


def heatmap(phi_1_net, rho_net, phi_2_net=None, pos1=[0,0,0.5], vel1=[0,0,0], pos2=None, vel2=None, GE=False, pos3=None, phi_3_net=None, vel3=None, pos4=None, phi_4_net=None, vel4=None):
    """生成热力图数据"""
    z_min = 0.0
    z_max = 0.7
    y_min = -0.4
    y_max = 0.4
    
    z_length = int((z_max-z_min)*200) + 1
    y_length = int((y_max-y_min)*200) + 1
    
    z = np.linspace(z_max, z_min, z_length)
    y = np.linspace(y_min, y_max, y_length)
    
    fa_heatmap = np.zeros([1, z_length, y_length], dtype=np.float32)
    
    for j in range(y_length):
        for i in range(z_length):
            c = np.zeros([1, 6], dtype=np.float32)
            c[0, 0] = pos1[0] - 0
            c[0, 1] = pos1[1] - y[j]
            c[0, 2] = pos1[2] - z[i]
            c[0, 3] = vel1[0]
            c[0, 4] = vel1[1]
            c[0, 5] = vel1[2]
            cc1 = torch.from_numpy(c)
            if GE:
                cc1 = torch.from_numpy(c[:, 2:])
            if pos2 is not None:
                c = np.zeros([1, 6], dtype=np.float32)
                c[0, 0] = pos2[0] - 0
                c[0, 1] = pos2[1] - y[j]
                c[0, 2] = pos2[2] - z[i]
                c[0, 3] = vel2[0]
                c[0, 4] = vel2[1]
                c[0, 5] = vel2[2]
                cc2 = torch.from_numpy(c)
            if pos3 is not None:
                c = np.zeros([1, 6], dtype=np.float32)
                c[0, 0] = pos3[0] - 0
                c[0, 1] = pos3[1] - y[j]
                c[0, 2] = pos3[2] - z[i]
                c[0, 3] = vel3[0]
                c[0, 4] = vel3[1]
                c[0, 5] = vel3[2]
                cc3 = torch.from_numpy(c)
            if pos4 is not None:
                c = np.zeros([1, 6], dtype=np.float32)
                c[0, 0] = pos4[0] - 0
                c[0, 1] = pos4[1] - y[j]
                c[0, 2] = pos4[2] - z[i]
                c[0, 3] = vel4[0]
                c[0, 4] = vel4[1]
                c[0, 5] = vel4[2]
                cc4 = torch.from_numpy(c)
            with torch.no_grad():
                if pos2 is None:
                    fa_heatmap[0, i, j] = rho_net(phi_1_net(cc1[:, :]))[0, 0].item() # f_a_z
                else:
                    if pos3 is None:
                        fa_heatmap[0, i, j] = rho_net(phi_1_net(cc1[:, :]) + phi_2_net(cc2[:, :]))[0, 0].item() # f_a_z
                    else:
                        if pos4 is None:
                            fa_heatmap[0, i, j] = rho_net(phi_1_net(cc1[:, :]) + phi_2_net(cc2[:, :]) + phi_3_net(cc3[:, :]))[0, 0].item() # f_a_z
                        else:
                            fa_heatmap[0, i, j] = rho_net(phi_1_net(cc1[:, :]) + phi_2_net(cc2[:, :]) + phi_3_net(cc3[:, :]) + phi_4_net(cc4[:, :]))[0, 0].item() # f_a_z

    return y, z, fa_heatmap[0, :, :]


def vis(pp, phi_G_net, phi_L_net, rho_L_net, phi_S_net, rho_S_net, rasterized):
    """可视化神经网络输出"""
    # visualization
    vmin = -20
    vmax = 5
    plt.figure(figsize=(20,16))
    plt.subplot(5, 4, 1, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_G_net, rho_L_net, pos1=[0,0,0], GE=True)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (Ge2L)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)

    plt.subplot(5, 4, 2, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_G_net, rho_S_net, pos1=[0,0,0], GE=True)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (Ge2S)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)

    plt.subplot(5, 4, 5, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_L_net, rho_L_net)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (L2L)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([0], [0.5], marker='*', markersize=10, color="black")

    plt.subplot(5, 4, 6, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_S_net, rho_S_net)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (S2S)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([0], [0.5], marker='*', markersize=10, color="black")

    plt.subplot(5, 4, 7, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_L_net, rho_S_net)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (L2S)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([0], [0.5], marker='*', markersize=10, color="black")

    plt.subplot(5, 4, 8, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_S_net, rho_L_net)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (S2L)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([0], [0.5], marker='*', markersize=10, color="black")
    
    plt.subplot(5, 4, 9, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_G_net, rho_L_net, phi_2_net=phi_L_net, pos1=[0,0,0], pos2=[0,0,0.5], vel2=[0,0,0], GE=True)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (GeL2L)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([0], [0.5], marker='*', markersize=20, color="black")

    plt.subplot(5, 4, 10, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_G_net, rho_L_net, phi_2_net=phi_S_net, pos1=[0,0,0], pos2=[0,0,0.5], vel2=[0,0,0], GE=True)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (GeS2L)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([0], [0.5], marker='*', markersize=10, color="black")

    plt.subplot(5, 4, 11, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_G_net, rho_S_net, phi_2_net=phi_L_net, pos1=[0,0,0], pos2=[0,0,0.5], vel2=[0,0,0], GE=True)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (GeL2S)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([0], [0.5], marker='*', markersize=20, color="black")

    plt.subplot(5, 4, 12, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_G_net, rho_S_net, phi_2_net=phi_S_net, pos1=[0,0,0], pos2=[0,0,0.5], vel2=[0,0,0], GE=True)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (GeS2S)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([0], [0.5], marker='*', markersize=10, color="black")

    pos1 = [0, -0.1, 0.5]
    pos2 = [0, 0.1, 0.5]
    vel1 = [0, 0, 0]
    vel2 = [0 ,0, 0]
    plt.subplot(5, 4, 13, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_S_net, rho_L_net, phi_2_net=phi_S_net, pos1=pos1, vel1=vel1, pos2=pos2, vel2=vel2)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (SS2L)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([pos1[1]], [pos1[2]], marker='*', markersize=10, color="black")
    plt.plot([pos2[1]], [pos2[2]], marker='*', markersize=10, color="black")

    plt.subplot(5, 4, 14, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_L_net, rho_L_net, phi_2_net=phi_L_net, pos1=pos1, vel1=vel1, pos2=pos2, vel2=vel2)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (LL2L)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([pos1[1]], [pos1[2]], marker='*', markersize=20, color="black")
    plt.plot([pos2[1]], [pos2[2]], marker='*', markersize=20, color="black")

    plt.subplot(5, 4, 15, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_L_net, rho_L_net, phi_2_net=phi_S_net, pos1=pos1, vel1=vel1, pos2=pos2, vel2=vel2)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (LS2L)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([pos1[1]], [pos1[2]], marker='*', markersize=20, color="black")
    plt.plot([pos2[1]], [pos2[2]], marker='*', markersize=10, color="black")

    plt.subplot(5, 4, 16, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_S_net, rho_L_net, phi_2_net=phi_L_net, pos1=pos1, vel1=vel1, pos2=pos2, vel2=vel2)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (SL2L)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([pos1[1]], [pos1[2]], marker='*', markersize=10, color="black")
    plt.plot([pos2[1]], [pos2[2]], marker='*', markersize=20, color="black")

    plt.subplot(5, 4, 17, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_S_net, rho_S_net, phi_2_net=phi_S_net, pos1=pos1, vel1=vel1, pos2=pos2, vel2=vel2)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (SS2S)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([pos1[1]], [pos1[2]], marker='*', markersize=10, color="black")
    plt.plot([pos2[1]], [pos2[2]], marker='*', markersize=10, color="black")

    plt.subplot(5, 4, 18, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_L_net, rho_S_net, phi_2_net=phi_L_net, pos1=pos1, vel1=vel1, pos2=pos2, vel2=vel2)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (LL2S)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([pos1[1]], [pos1[2]], marker='*', markersize=20, color="black")
    plt.plot([pos2[1]], [pos2[2]], marker='*', markersize=20, color="black")

    plt.subplot(5, 4, 19, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_L_net, rho_S_net, phi_2_net=phi_S_net, pos1=pos1, vel1=vel1, pos2=pos2, vel2=vel2)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (LS2S)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([pos1[1]], [pos1[2]], marker='*', markersize=20, color="black")
    plt.plot([pos2[1]], [pos2[2]], marker='*', markersize=10, color="black")

    plt.subplot(5, 4, 20, rasterized=rasterized)
    y, z, fa_heatmap = heatmap(phi_S_net, rho_S_net, phi_2_net=phi_L_net, pos1=pos1, vel1=vel1, pos2=pos2, vel2=vel2)
    plt.ylabel(r'$z$ (m)', fontsize = 15)
    plt.xlabel(r'$y$ (m)', fontsize = 15)
    plt.title(r'${F_a}_z$ (SL2S)', fontsize = 15)
    plt.pcolor(y, z, fa_heatmap, cmap='Reds_r', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tick_params(labelsize = 13)
    plt.plot([pos1[1]], [pos1[2]], marker='*', markersize=10, color="black")
    plt.plot([pos2[1]], [pos2[2]], marker='*', markersize=20, color="black")
    plt.tight_layout()
    pp.savefig()
    plt.close()