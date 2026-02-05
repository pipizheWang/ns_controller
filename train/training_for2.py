"""
双机训练脚本 (Two UAV Training Script)
专门用于训练双机场景的神经网络

使用方法:
1. 先使用 preprocess_two_uav.py 预处理数据，生成 .npy 文件
2. 然后运行本脚本进行训练:
   python training_for2.py --data_input path/to/data_input.npy --data_output path/to/data_output.npy --uav_type L
"""

from nns import phi_Net, rho_Net
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import random


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
    code = np.sum([ord(s) for s in type])
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

    Data_input = torch.from_numpy(data_input[:, :]).float().to(device) # 7x1
    Data_output = torch.from_numpy(data_output[:, 2:]).float().to(device) # 1x1
    Val_input = torch.from_numpy(val_input[:, :]).float().to(device)
    Val_output = torch.from_numpy(val_output[:, 2:]).float().to(device)
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

# ==================== 主程序 ====================

parser = argparse.ArgumentParser(description='Two UAV Training Script')

parser.add_argument('--data_input', type=str, required=True, help='Path to preprocessed input data (.npy)')
parser.add_argument('--data_output', type=str, required=True, help='Path to preprocessed output data (.npy)')
parser.add_argument('--uav_type', type=str, default='L', choices=['L', 'S'], 
                    help='UAV type: L (Large) or S (Small)')
parser.add_argument('--path', default='models/output_two_uav', help='Output path')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--hidden_dim', type=int, default=20, help='Hidden dimension')

opt = parser.parse_args()

# 训练参数
output_name = opt.path
num_epochs = opt.num_epochs
hidden_dim = opt.hidden_dim
batch_size = opt.batch_size
rasterized = True

# 场景编码
scenario = 'L2L' if opt.uav_type == 'L' else 'S2S'

# 创建输出目录
if os.path.isdir(f'../data/models/{output_name}'):
    print(f'../data/models/{output_name} exists and will be rewritten!')
else:
    os.makedirs(f'../data/models/{output_name}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

torch.set_default_tensor_type('torch.DoubleTensor')
torch.multiprocessing.set_sharing_strategy('file_system')
pp = PdfPages(f'../data/models/{output_name}/output.pdf')


##### Part I: 加载预处理好的数据 #####
print('=' * 60)
print('PART I: Loading Preprocessed Data')
print('=' * 60)

# 加载预处理好的 numpy 数据
print(f'Loading data from: {opt.data_input}')
print(f'Loading data from: {opt.data_output}')

data_input_all = np.load(opt.data_input)
data_output_all = np.load(opt.data_output)

print(f'✓ Loaded input data: {data_input_all.shape}')
print(f'✓ Loaded output data: {data_output_all.shape}')
print(f'✓ Total training samples: {data_input_all.shape[0]}')

##### Part II: 生成训练集和验证集 #####
print('\n' + '=' * 60)
print('PART II: Generate Training and Validation Sets')
print('=' * 60)

trainset, trainloader, valset, val_input, val_output = set_generate(
    data_input_all, data_output_all, scenario, device, batch_size
)

print(f'Training samples: {len(trainset)}')
print(f'Validation samples: {len(valset)}')
print(f'Batch size: {batch_size}')

##### Part III: 初始化神经网络 #####
print('\n' + '=' * 60)
print('PART III: Initialize Neural Networks')
print('=' * 60)

# 创建网络
phi_G_net = phi_Net(inputdim=4, hiddendim=hidden_dim).to(device, dtype=torch.float32)
if opt.uav_type == 'L':
    phi_net = phi_Net(inputdim=6, hiddendim=hidden_dim).to(device, dtype=torch.float32)
    rho_net = rho_Net(hiddendim=hidden_dim).to(device, dtype=torch.float32)
    print('Created networks for Large UAV')
else:
    phi_net = phi_Net(inputdim=6, hiddendim=hidden_dim).to(device, dtype=torch.float32)
    rho_net = rho_Net(hiddendim=hidden_dim).to(device, dtype=torch.float32)
    print('Created networks for Small UAV')

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer_phi_G = optim.Adam(phi_G_net.parameters(), lr=1e-3)
optimizer_phi = optim.Adam(phi_net.parameters(), lr=1e-3)
optimizer_rho = optim.Adam(rho_net.parameters(), lr=1e-3)

##### Part IV: 训练 #####
print('\n' + '=' * 60)
print('PART IV: Training')
print('=' * 60)

def compute_loss(data_batch, phi_G_net, phi_net, rho_net, criterion):
    """计算一个batch的损失"""
    inputs = data_batch['input'].to(device)
    labels = data_batch['output'].to(device)
    
    # 前向传播
    # 地面效应特征
    ground_features = phi_G_net(inputs[:, [2, 5, -1, -1]])  # z, vz, 0, 0
    # 相对状态特征
    relative_features = phi_net(inputs[:, :6])  # 相对位置和速度
    # 聚合
    aggregated = ground_features + relative_features
    # 输出预测
    predictions = rho_net(aggregated)
    
    # 计算损失
    loss = criterion(predictions, labels)
    return loss, predictions


# 训练前损失
print('Computing initial loss...')
phi_G_net.eval()
phi_net.eval()
rho_net.eval()
with torch.no_grad():
    total_loss = 0.0
    for batch in trainloader:
        loss, _ = compute_loss(batch, phi_G_net, phi_net, rho_net, criterion)
        total_loss += loss.item()
    initial_loss = total_loss / len(trainloader)
    print(f'Initial training loss: {initial_loss:.6f}')

# 开始训练
Loss_history = []

for epoch in range(num_epochs):
    phi_G_net.train()
    phi_net.train()
    rho_net.train()
    
    epoch_loss = 0.0
    
    for batch_idx, batch in enumerate(trainloader):
        # 前向传播
        loss, _ = compute_loss(batch, phi_G_net, phi_net, rho_net, criterion)
        
        # 反向传播
        optimizer_phi_G.zero_grad()
        optimizer_phi.zero_grad()
        optimizer_rho.zero_grad()
        loss.backward()
        optimizer_phi_G.step()
        optimizer_phi.step()
        optimizer_rho.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(trainloader)
    Loss_history.append(avg_loss)
    
    # 每5个epoch打印一次
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

print('Training finished!')

# 绘制训练曲线
plt.figure(figsize=(10, 6))
plt.plot(Loss_history, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True, alpha=0.3)
pp.savefig()
plt.close()

##### Part V: 评估和保存 #####
print('\n' + '=' * 60)
print('PART V: Evaluation and Saving')
print('=' * 60)

# 训练后损失
phi_G_net.eval()
phi_net.eval()
rho_net.eval()
with torch.no_grad():
    # 训练集损失
    total_loss = 0.0
    for batch in trainloader:
        loss, _ = compute_loss(batch, phi_G_net, phi_net, rho_net, criterion)
        total_loss += loss.item()
    train_loss = total_loss / len(trainloader)
    print(f'Final training loss: {train_loss:.6f}')
    
    # 验证集损失
    total_loss = 0.0
    for batch in valset:
        val_batch = {
            'input': torch.from_numpy(batch['input'].reshape(1, -1)).float(),
            'output': torch.from_numpy(batch['output'].reshape(1, -1)).float()
        }
        loss, _ = compute_loss(val_batch, phi_G_net, phi_net, rho_net, criterion)
        total_loss += loss.item()
    val_loss = total_loss / len(valset)
    print(f'Validation loss: {val_loss:.6f}')

# 保存模型
phi_G_net.cpu()
phi_net.cpu()
rho_net.cpu()

torch.save(phi_G_net.state_dict(), f'../data/models/{output_name}/phi_G.pth')
torch.save(phi_net.state_dict(), f'../data/models/{output_name}/phi_{opt.uav_type}.pth')
torch.save(rho_net.state_dict(), f'../data/models/{output_name}/rho_{opt.uav_type}.pth')
print(f'\n✓ Models saved to: ../data/models/{output_name}/')

##### Part VI: 可视化 #####
print('\n' + '=' * 60)
print('PART VI: Visualization')
print('=' * 60)

# 加载模型进行可视化
phi_G_net.load_state_dict(torch.load(f'../data/models/{output_name}/phi_G.pth'))
phi_net.load_state_dict(torch.load(f'../data/models/{output_name}/phi_{opt.uav_type}.pth'))
rho_net.load_state_dict(torch.load(f'../data/models/{output_name}/rho_{opt.uav_type}.pth'))

# 可视化网络输出
if opt.uav_type == 'L':
    vis(pp, phi_G_net, phi_net, rho_net, phi_net, rho_net, rasterized)
else:
    vis(pp, phi_G_net, phi_net, rho_net, phi_net, rho_net, rasterized)

# 验证预测
print('Generating validation plots...')
val_input_np = val_input[:, :]
val_output_np = val_output[:, :]

# 计算预测值
with torch.no_grad():
    inputs_torch = torch.from_numpy(val_input_np).float()
    ground_features = phi_G_net(inputs_torch[:, [2, 5, -1, -1]])
    relative_features = phi_net(inputs_torch[:, :6])
    aggregated = ground_features + relative_features
    predictions = rho_net(aggregated).numpy()

# 绘制预测vs真实值
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
titles = ['Fax', 'Fay', 'Faz']
for i, (ax, title) in enumerate(zip(axes, titles)):
    ax.scatter(val_output_np[:, i], predictions[:, i], alpha=0.5, s=1)
    ax.plot([val_output_np[:, i].min(), val_output_np[:, i].max()],
            [val_output_np[:, i].min(), val_output_np[:, i].max()],
            'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel(f'True {title} (g)')
    ax.set_ylabel(f'Predicted {title} (g)')
    ax.set_title(f'{title} Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
pp.savefig()
plt.close()

pp.close()
print(f'\n✓ PDF report saved to: ../data/models/{output_name}/output.pdf')

print('\n' + '=' * 60)
print('TRAINING COMPLETE!')
print('=' * 60)
print(f'Summary:')
print(f'  - Scenario: {scenario}')
print(f'  - Training samples: {len(trainset)}')
print(f'  - Validation samples: {len(valset)}')
print(f'  - Initial loss: {initial_loss:.6f}')
print(f'  - Final training loss: {train_loss:.6f}')
print(f'  - Validation loss: {val_loss:.6f}')
print(f'  - Models saved to: ../data/models/{output_name}/')
