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

parser.add_argument('--data_input', type=str, default='data/data_input_balanced.npy', help='Path to preprocessed input data (.npy)')
parser.add_argument('--data_output', type=str, default='data/data_output_balanced.npy', help='Path to preprocessed output data (.npy)')
parser.add_argument('--output_path', default='model', help='Output path')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs (default: 100)')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--hidden_dim', type=int, default=20, help='Hidden dimension')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--early_stop_patience', type=int, default=30, help='Early stopping patience')
parser.add_argument('--lr_patience', type=int, default=15, help='Learning rate scheduler patience')
parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')

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
optimizer_phi = optim.Adam(phi_net.parameters(), lr=opt.learning_rate)
optimizer_rho = optim.Adam(rho_net.parameters(), lr=opt.learning_rate)

# 学习率调度器 (当验证loss不再下降时降低学习率)
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler_phi = ReduceLROnPlateau(optimizer_phi, mode='min', factor=0.5,
                                  patience=opt.lr_patience, verbose=True, min_lr=1e-6)
scheduler_rho = ReduceLROnPlateau(optimizer_rho, mode='min', factor=0.5,
                                  patience=opt.lr_patience, verbose=True, min_lr=1e-6)
print(f'Optimizer: Adam with initial lr={opt.learning_rate}')
print(f'LR Scheduler: ReduceLROnPlateau (patience={opt.lr_patience}, factor=0.5)')
print(f'Gradient clipping: {opt.grad_clip}')

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
Learning_rate_history = []

# 早停和最佳模型保存
best_val_loss = float('inf')
best_epoch = 0
patience_counter = 0
best_model_state = None

import time
start_time = time.time()

print(f'\n⏱️  Training started at {time.strftime("%H:%M:%S")}')
print(f'Target: Train until convergence or {num_epochs} epochs')
print(f'Early stopping patience: {opt.early_stop_patience} epochs\n')

for epoch in range(num_epochs):
    epoch_start_time = time.time()
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

        # 梯度裁剪 (防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(phi_net.parameters(), opt.grad_clip)
        torch.nn.utils.clip_grad_norm_(rho_net.parameters(), opt.grad_clip)

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

    # 学习率调度
    current_lr_phi = optimizer_phi.param_groups[0]['lr']
    current_lr_rho = optimizer_rho.param_groups[0]['lr']
    Learning_rate_history.append(current_lr_phi)

    scheduler_phi.step(avg_val_loss)
    scheduler_rho.step(avg_val_loss)

    # 检查是否为最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        patience_counter = 0
        # 保存最佳模型状态
        best_model_state = {
            'phi_net': phi_net.state_dict(),
            'rho_net': rho_net.state_dict(),
            'epoch': best_epoch,
            'val_loss': best_val_loss
        }
        best_indicator = ' ⭐ (Best!)'
    else:
        patience_counter += 1
        best_indicator = ''

    # 计算epoch用时
    epoch_time = time.time() - epoch_start_time

    # 打印进度
    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1 or best_indicator:
        print(f'Epoch [{epoch+1:3d}/{num_epochs}] | '
              f'Train: {avg_train_loss:8.4f} | Val: {avg_val_loss:8.4f} | '
              f'LR: {current_lr_phi:.2e} | Time: {epoch_time:.1f}s{best_indicator}')

    # 早停检查
    if patience_counter >= opt.early_stop_patience:
        print(f'\n🛑 Early stopping triggered at epoch {epoch+1}')
        print(f'   No improvement for {opt.early_stop_patience} epochs')
        print(f'   Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}')
        break

training_time = time.time() - start_time
print(f'\n✅ Training finished in {training_time/60:.1f} minutes')
print(f'   Best model: Epoch {best_epoch} with Val Loss = {best_val_loss:.6f}')

# 恢复最佳模型
if best_model_state is not None:
    phi_net.load_state_dict(best_model_state['phi_net'])
    rho_net.load_state_dict(best_model_state['rho_net'])
    print(f'   ✓ Restored best model from epoch {best_epoch}')

# 1. 绘制训练和验证损失曲线对比（改进版，3个子图）
fig = plt.figure(figsize=(18, 5))

# 子图1: 线性坐标
ax1 = plt.subplot(1, 3, 1)
ax1.plot(Train_loss_history, label='Training Loss', linewidth=2, color='#4ECDC4')
ax1.plot(Val_loss_history, label='Validation Loss', linewidth=2, color='#FF6B6B')
ax1.axvline(best_epoch-1, color='green', linestyle='--', linewidth=1.5,
            alpha=0.7, label=f'Best (Epoch {best_epoch})')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 子图2: 对数坐标
ax2 = plt.subplot(1, 3, 2)
ax2.semilogy(Train_loss_history, label='Training Loss', linewidth=2, color='#4ECDC4')
ax2.semilogy(Val_loss_history, label='Validation Loss', linewidth=2, color='#FF6B6B')
ax2.axvline(best_epoch-1, color='green', linestyle='--', linewidth=1.5,
            alpha=0.7, label=f'Best (Epoch {best_epoch})')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss (log scale)', fontsize=12)
ax2.set_title('Loss Curve (Log Scale)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which='both')

# 子图3: 学习率历史
ax3 = plt.subplot(1, 3, 3)
ax3.semilogy(Learning_rate_history, label='Learning Rate', linewidth=2, color='#95E1D3')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Learning Rate (log scale)', fontsize=12)
ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, which='both')
ax3.legend(fontsize=10)

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

# 保存模型（包含完整训练信息）
phi_net.cpu()
rho_net.cpu()

torch.save(phi_net.state_dict(), f'{output_name}/phi_net.pth')
torch.save(rho_net.state_dict(), f'{output_name}/rho_net.pth')

# 保存完整的checkpoint（包含训练历史）
checkpoint = {
    'phi_net_state_dict': phi_net.state_dict(),
    'rho_net_state_dict': rho_net.state_dict(),
    'train_loss_history': Train_loss_history,
    'val_loss_history': Val_loss_history,
    'learning_rate_history': Learning_rate_history,
    'best_epoch': best_epoch,
    'best_val_loss': best_val_loss,
    'final_train_loss': train_loss,
    'final_val_loss': val_loss,
    'hyperparameters': {
        'hidden_dim': hidden_dim,
        'batch_size': batch_size,
        'learning_rate': opt.learning_rate,
        'num_epochs_trained': len(Train_loss_history),
        'input_dim': 6,
        'output_dim': 1
    }
}
torch.save(checkpoint, f'{output_name}/checkpoint_best.pth')

# 保存训练日志（Part 1 - 基本信息）
with open(f'{output_name}/training_log.txt', 'w') as f:
    f.write('=' * 70 + '\n')
    f.write('Training Log - Two UAV Downwash Force Prediction\n')
    f.write('=' * 70 + '\n\n')
    f.write(f'Training Date: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    f.write(f'Training Time: {training_time/60:.1f} minutes\n\n')

    f.write('Hyperparameters:\n')
    f.write(f'  - Hidden Dimension: {hidden_dim}\n')
    f.write(f'  - Batch Size: {batch_size}\n')
    f.write(f'  - Initial Learning Rate: {opt.learning_rate}\n')
    f.write(f'  - Epochs Trained: {len(Train_loss_history)}\n')
    f.write(f'  - Early Stop Patience: {opt.early_stop_patience}\n')
    f.write(f'  - LR Scheduler Patience: {opt.lr_patience}\n')
    f.write(f'  - Gradient Clipping: {opt.grad_clip}\n\n')

    f.write('Dataset:\n')
    f.write(f'  - Training Samples: {len(trainset)}\n')
    f.write(f'  - Validation Samples: {len(valset)}\n')
    f.write(f'  - Input Data: {opt.data_input}\n')
    f.write(f'  - Output Data: {opt.data_output}\n\n')

    f.write('Training Results:\n')
    f.write(f'  - Best Epoch: {best_epoch}\n')
    f.write(f'  - Best Val Loss: {best_val_loss:.6f}\n')
    f.write(f'  - Initial Loss: {initial_loss:.6f}\n')
    f.write(f'  - Final Train Loss: {train_loss:.6f}\n')
    f.write(f'  - Final Val Loss: {val_loss:.6f}\n')
    f.write(f'  - Loss Reduction: {100*(initial_loss-train_loss)/initial_loss:.2f}%\n\n')

print(f'✓ Models saved to: {output_name}/')
print(f'✓ Full checkpoint saved to: {output_name}/checkpoint_best.pth')

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

# 追加验证指标到训练日志（Part 2 - 验证指标）
try:
    from sklearn.metrics import r2_score, mean_squared_error
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(mean_squared_error(val_output_np[:, 0], predictions[:, 0]))
    max_error = np.max(np.abs(errors))
    r2 = r2_score(val_output_np[:, 0], predictions[:, 0])

    with open(f'{output_name}/training_log.txt', 'a') as f:
        f.write('Validation Metrics:\n')
        f.write(f'  - MAE: {mae:.6f} gram\n')
        f.write(f'  - RMSE: {rmse:.6f} gram\n')
        f.write(f'  - Max Error: {max_error:.6f} gram\n')
        f.write(f'  - R² Score: {r2:.6f}\n\n')

        # 性能评级
        f.write('Performance Rating:\n')
        if r2 > 0.99:
            f.write('  ⭐⭐⭐⭐⭐ Excellent (R² > 0.99)\n')
        elif r2 > 0.95:
            f.write('  ⭐⭐⭐⭐ Very Good (R² > 0.95)\n')
        elif r2 > 0.90:
            f.write('  ⭐⭐⭐ Good (R² > 0.90)\n')
        elif r2 > 0.80:
            f.write('  ⭐⭐ Fair (R² > 0.80)\n')
        else:
            f.write('  ⭐ Poor (R² < 0.80)\n')

    print(f'✓ Training log updated with validation metrics: {output_name}/training_log.txt')

except Exception as e:
    print(f'⚠ Warning: Could not append validation metrics to log ({e})')

print('\n' + '=' * 60)
print('TRAINING COMPLETE!')
print('=' * 60)
print(f'Summary:')
print(f'  - Scenario: {scenario}')
print(f'  - Training samples: {len(trainset)}')
print(f'  - Validation samples: {len(valset)}')
print(f'  - Best epoch: {best_epoch}')
print(f'  - Initial loss: {initial_loss:.6f}')
print(f'  - Final training loss: {train_loss:.6f}')
print(f'  - Validation loss: {val_loss:.6f}')
try:
    print(f'  - R² Score: {r2:.6f}')
    print(f'  - RMSE: {rmse:.6f} gram')
except:
    pass
print(f'  - Models saved to: {output_name}/')
