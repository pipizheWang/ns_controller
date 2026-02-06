"""
双机训练脚本 (Two UAV Training Script)
专门用于训练双机场景的神经网络

使用方法:
1. 先使用 preprocess_two_uav.py 预处理数据，生成 .npy 文件
2. 然后运行本脚本进行训练:
   python training_for2.py
"""

from nns import phi_Net, rho_Net
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

# 从 utils 模块导入所有辅助函数
from utils import set_generate, heatmap, vis


# ==================== 主程序 ====================

parser = argparse.ArgumentParser(description='Two UAV Training Script')

parser.add_argument('--data_input', type=str, default='data/data_input_L2L.npy', help='Path to preprocessed input data (.npy)')
parser.add_argument('--data_output', type=str, default='data/data_output_L2L.npy', help='Path to preprocessed output data (.npy)')
parser.add_argument('--output_path', default='model', help='Output path')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--hidden_dim', type=int, default=20, help='Hidden dimension')

opt = parser.parse_args()

# 训练参数
output_name = opt.output_path
num_epochs = opt.num_epochs
hidden_dim = opt.hidden_dim
batch_size = opt.batch_size
rasterized = True

# 场景编码
scenario = 'L2L'

# 创建输出目录
if os.path.isdir(output_name):
    print(f'{output_name} exists and will be rewritten!')
else:
    os.makedirs(output_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

torch.set_default_tensor_type('torch.DoubleTensor')
torch.multiprocessing.set_sharing_strategy('file_system')

# 创建可视化输出目录
import os
vis_dir = f'{output_name}/visualizations'
os.makedirs(vis_dir, exist_ok=True)


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
phi_net = phi_Net(inputdim=6, hiddendim=hidden_dim).to(device, dtype=torch.float32)
rho_net = rho_Net(hiddendim=hidden_dim).to(device, dtype=torch.float32)
print('Created networks: phi_net and rho_net')

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer_phi = optim.Adam(phi_net.parameters(), lr=1e-3)
optimizer_rho = optim.Adam(rho_net.parameters(), lr=1e-3)

##### Part IV: 训练 #####
print('\n' + '=' * 60)
print('PART IV: Training')
print('=' * 60)

def compute_loss(data_batch, phi_net, rho_net, criterion):
    """计算一个batch的损失"""
    inputs = data_batch['input'].to(device)
    labels = data_batch['output'].to(device)
    
    # 前向传播
    # 相对状态特征
    features = phi_net(inputs)  # 相对位置和速度 (6维)
    # 输出预测
    predictions = rho_net(features)
    
    # 计算损失
    loss = criterion(predictions, labels)
    return loss, predictions


# 训练前损失
print('Computing initial loss...')
phi_net.eval()
rho_net.eval()
with torch.no_grad():
    total_loss = 0.0
    for batch in trainloader:
        loss, _ = compute_loss(batch, phi_net, rho_net, criterion)
        total_loss += loss.item()
    initial_loss = total_loss / len(trainloader)
    print(f'Initial training loss: {initial_loss:.6f}')

# 开始训练
Train_loss_history = []
Val_loss_history = []

for epoch in range(num_epochs):
    phi_net.train()
    rho_net.train()
    
    epoch_loss = 0.0
    
    for batch_idx, batch in enumerate(trainloader):
        # 前向传播
        loss, _ = compute_loss(batch, phi_net, rho_net, criterion)
        
        # 反向传播
        optimizer_phi.zero_grad()
        optimizer_rho.zero_grad()
        loss.backward()
        optimizer_phi.step()
        optimizer_rho.step()
        
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(trainloader)
    Train_loss_history.append(avg_train_loss)
    
    # 计算验证损失
    phi_net.eval()
    rho_net.eval()
    with torch.no_grad():
        val_epoch_loss = 0.0
        for batch in valset:
            val_batch = {
                'input': batch['input'].reshape(1, -1),
                'output': batch['output'].reshape(1, -1)
            }
            loss, _ = compute_loss(val_batch, phi_net, rho_net, criterion)
            val_epoch_loss += loss.item()
        avg_val_loss = val_epoch_loss / len(valset)
        Val_loss_history.append(avg_val_loss)
    
    # 每5个epoch打印一次
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

print('Training finished!')

# 1. 绘制训练和验证损失曲线对比
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(Train_loss_history, label='Training Loss', linewidth=2)
plt.plot(Val_loss_history, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training vs Validation Loss', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.semilogy(Train_loss_history, label='Training Loss', linewidth=2)
plt.semilogy(Val_loss_history, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (log scale)', fontsize=12)
plt.title('Loss Curve (Log Scale)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{vis_dir}/loss_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'✓ Loss curves saved to: {vis_dir}/loss_curves.png')

##### Part V: 评估和保存 #####
print('\n' + '=' * 60)
print('PART V: Evaluation and Saving')
print('=' * 60)

# 训练后损失
phi_net.eval()
rho_net.eval()
with torch.no_grad():
    # 训练集损失
    total_loss = 0.0
    for batch in trainloader:
        loss, _ = compute_loss(batch, phi_net, rho_net, criterion)
        total_loss += loss.item()
    train_loss = total_loss / len(trainloader)
    print(f'Final training loss: {train_loss:.6f}')
    
    # 验证集损失
    total_loss = 0.0
    for batch in valset:
        val_batch = {
            'input': batch['input'].reshape(1, -1),
            'output': batch['output'].reshape(1, -1)
        }
        loss, _ = compute_loss(val_batch, phi_net, rho_net, criterion)
        total_loss += loss.item()
    val_loss = total_loss / len(valset)
    print(f'Validation loss: {val_loss:.6f}')

# 保存模型
phi_net.cpu()
rho_net.cpu()

torch.save(phi_net.state_dict(), f'{output_name}/phi_net.pth')
torch.save(rho_net.state_dict(), f'{output_name}/rho_net.pth')
print(f'\n✓ Models saved to: {output_name}/')

##### Part VI: 可视化 #####
print('\n' + '=' * 60)
print('PART VI: Visualization')
print('=' * 60)

# 加载模型进行可视化
phi_net.load_state_dict(torch.load(f'{output_name}/phi_net.pth', weights_only=True))
rho_net.load_state_dict(torch.load(f'{output_name}/rho_net.pth', weights_only=True))

print('⚠ Skipping network heatmap visualization (can be added separately if needed)')

# 验证预测
print('Generating validation plots...')
val_input_np = val_input[:, :]
val_output_np = val_output[:, :]

# 计算预测值
with torch.no_grad():
    inputs_torch = torch.from_numpy(val_input_np).float()
    features = phi_net(inputs_torch)
    predictions = rho_net(features).numpy()

# 绘制预测vs真实值
try:
    # 计算误差
    errors = predictions - val_output_np
    
    # 1. 预测vs真实值散点图 (只有Faz)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(val_output_np[:, 0], predictions[:, 0], alpha=0.6, s=10)
    min_val = min(val_output_np[:, 0].min(), predictions[:, 0].min())
    max_val = max(val_output_np[:, 0].max(), predictions[:, 0].max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel(f'True Faz (gram)', fontsize=11)
    ax.set_ylabel(f'Predicted Faz (gram)', fontsize=11)
    ax.set_title(f'Faz Prediction', fontsize=12, fontweight='bold')
    
    # 添加R²和RMSE
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(val_output_np[:, 0], predictions[:, 0])
    rmse = np.sqrt(mean_squared_error(val_output_np[:, 0], predictions[:, 0]))
    ax.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{vis_dir}/1_predictions_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Predictions scatter plot saved')
    
    # 2. 误差分布直方图
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(errors[:, 0], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel(f'Prediction Error (gram)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Faz Error Distribution', fontsize=12, fontweight='bold')
    
    # 添加统计信息
    mean_err = np.mean(errors[:, 0])
    std_err = np.std(errors[:, 0])
    ax.text(0.05, 0.95, f'Mean: {mean_err:.4f}\nStd: {std_err:.4f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{vis_dir}/2_error_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Error distribution plot saved')
    
    # 3. 残差图 (Residual Plot)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(predictions[:, 0], errors[:, 0], alpha=0.6, s=10)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel(f'Predicted Faz (gram)', fontsize=11)
    ax.set_ylabel(f'Residual (gram)', fontsize=11)
    ax.set_title(f'Faz Residual Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{vis_dir}/3_residual_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Residual plot saved')
    
    # 4. 误差箱线图
    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot([errors[:, 0]], labels=['Faz'], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.set_ylabel('Prediction Error (gram)', fontsize=12)
    ax.set_title('Faz Prediction Error Box Plot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{vis_dir}/4_error_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Error boxplot saved')
    
    # 5. 绝对误差和相对误差统计
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 绝对误差
    abs_errors = np.abs(errors[:, 0])
    axes[0].hist(abs_errors, bins=50, alpha=0.7, color='coral', edgecolor='black')
    axes[0].set_xlabel(f'Absolute Error (gram)', fontsize=10)
    axes[0].set_ylabel('Frequency', fontsize=10)
    axes[0].set_title(f'Faz Absolute Error', fontsize=11, fontweight='bold')
    mae = np.mean(abs_errors)
    axes[0].axvline(mae, color='red', linestyle='--', linewidth=2, label=f'MAE={mae:.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 相对误差百分比
    relative_errors = 100 * errors[:, 0] / (np.abs(val_output_np[:, 0]) + 1e-8)
    axes[1].hist(relative_errors, bins=50, alpha=0.7, color='mediumseagreen', edgecolor='black')
    axes[1].set_xlabel(f'Relative Error (%)', fontsize=10)
    axes[1].set_ylabel('Frequency', fontsize=10)
    axes[1].set_title(f'Faz Relative Error', fontsize=11, fontweight='bold')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{vis_dir}/5_absolute_relative_errors.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Absolute and relative error plots saved')
    
    # 6. 生成统计报告
    print('\\n' + '='*60)
    print('PREDICTION STATISTICS')
    print('='*60)
    print(f'\\nFaz:')
    print(f'  MAE:  {np.mean(np.abs(errors[:, 0])):.6f} gram')
    print(f'  RMSE: {np.sqrt(np.mean(errors[:, 0]**2)):.6f} gram')
    print(f'  Max Error: {np.max(np.abs(errors[:, 0])):.6f} gram')
    print(f'  R² Score: {r2_score(val_output_np[:, 0], predictions[:, 0]):.6f}')
    
    print(f'\\n✓ All validation plots saved to: {vis_dir}/')
except Exception as e:
    plt.close()
    print(f'⚠ Warning: Validation plots failed ({type(e).__name__}: {e}), skipping...')
    import traceback
    traceback.print_exc()
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
print(f'  - Models saved to: {output_name}/')
