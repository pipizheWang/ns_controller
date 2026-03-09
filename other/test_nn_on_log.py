#!/usr/bin/env python3
"""
基于飞行日志的神经网络离线测试工具

读取一对 UAV0/UAV1 飞行日志 CSV，将日志中已校正的位置/速度输入神经网络，
计算估计的气动力，并与日志中已记录的 Fa_z 对比。

CSV 中的位置已经是校正后的坐标（UAV0 减去起飞偏移，UAV1 加上起飞偏移），
因此这里直接使用，不需要再做额外修正。

用法:
    python test_nn_on_log.py                          # 自动使用 log/ 下最新的一对日志
    python test_nn_on_log.py <uav0.csv> <uav1.csv>   # 手动指定日志路径
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────── 神经网络定义 ─────────────────────────────── #

class PhiNet(nn.Module):
    def __init__(self, inputdim=6, hiddendim=20):
        super().__init__()
        self.fc1 = nn.Linear(inputdim, 25)
        self.fc2 = nn.Linear(25, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, hiddendim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class RhoNet(nn.Module):
    def __init__(self, hiddendim=20):
        super().__init__()
        self.fc1 = nn.Linear(hiddendim, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# ────────────────────────────── 工具函数 ─────────────────────────────────── #

def load_csv(path):
    """读取 CSV，返回以列名为键的 numpy float64 数组字典"""
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"日志文件为空: {path}")
    keys = rows[0].keys()
    return {k: np.array([float(r[k]) for r in rows]) for k in keys}


def find_latest_log_pair(log_dir: Path):
    """在 log/ 目录中自动查找最新的 uav0_*.csv / uav1_*.csv 配对"""
    uav0_files = sorted(log_dir.glob('uav0_*.csv'))
    uav1_files = sorted(log_dir.glob('uav1_*.csv'))
    if not uav0_files or not uav1_files:
        raise FileNotFoundError(f"在 {log_dir} 中未找到 uav0_/uav1_ 开头的日志文件")
    return uav0_files[-1], uav1_files[-1]


def align_by_tick(d0, d1):
    """
    将 d1 的数据插值到 d0 的时间轴上（取两者重叠的时间段）。
    tick 单位：微秒。
    返回：(t_common, d0_aligned, d1_interp) 均基于 d0 时间轴。
    """
    t0 = d0['tick']
    t1 = d1['tick']

    t_start = max(t0[0], t1[0])
    t_end   = min(t0[-1], t1[-1])

    if t_start >= t_end:
        raise ValueError("两个日志的时间范围无重叠，请检查文件是否匹配")

    mask0 = (t0 >= t_start) & (t0 <= t_end)
    t_common = t0[mask0]

    d0_out = {k: v[mask0] for k, v in d0.items()}

    # 把 d1 的各列插值到 t_common
    d1_out = {}
    for k, v in d1.items():
        d1_out[k] = np.interp(t_common, t1, v)

    return t_common, d0_out, d1_out


def run_network(phi_net, rho_net, rel_pos, rel_vel):
    """
    批量推理。
    rel_pos: (N, 3)  upper - lower 校正后的相对位置
    rel_vel: (N, 3)  upper - lower 相对速度
    返回: Fa_z_newton (N,) float64
    """
    inp = np.concatenate([rel_pos, rel_vel], axis=1).astype(np.float32)
    t = torch.from_numpy(inp)
    with torch.no_grad():
        fa_gram = rho_net(phi_net(t)).squeeze(1)   # (N,)
    # 克力 → 牛顿
    return fa_gram.numpy().astype(np.float64) * 9.81 / 1000.0


# ─────────────────────────────────── main ────────────────────────────────── #

def main():
    # 确定日志路径
    script_dir = Path(__file__).resolve().parent
    workspace_root = next(
        (p for p in script_dir.parents if p.name == 'px4_ws'), None
    )
    log_dir = (workspace_root / 'src' / 'ns_controller' / 'log'
               if workspace_root else script_dir.parent / 'log')

    if len(sys.argv) == 3:
        path0, path1 = Path(sys.argv[1]), Path(sys.argv[2])
    elif len(sys.argv) == 1:
        path0, path1 = find_latest_log_pair(log_dir)
        print(f"自动选择日志:")
        print(f"  UAV0: {path0.name}")
        print(f"  UAV1: {path1.name}")
    else:
        print("用法: python test_nn_on_log.py [uav0.csv uav1.csv]")
        sys.exit(1)

    # 加载日志
    d0 = load_csv(path0)
    d1 = load_csv(path1)

    # 时间对齐
    t_common, d0, d1 = align_by_tick(d0, d1)
    N = len(t_common)
    t_sec = (t_common - t_common[0]) * 1e-6   # 微秒 → 秒

    # 加载神经网络
    model_dir = script_dir.parent / 'train' / 'model'
    phi_net = PhiNet(inputdim=6, hiddendim=20)
    rho_net = RhoNet(hiddendim=20)
    phi_path = model_dir / 'phi_net.pth'
    rho_path = model_dir / 'rho_net.pth'
    if not phi_path.exists() or not rho_path.exists():
        raise FileNotFoundError(f"模型文件未找到，请检查 {model_dir}")
    phi_net.load_state_dict(torch.load(phi_path, map_location='cpu', weights_only=True))
    rho_net.load_state_dict(torch.load(rho_path, map_location='cpu', weights_only=True))
    phi_net.eval(); rho_net.eval()
    print("模型加载成功")

    # 构造网络输入（日志中的位置/速度已经是校正后的坐标）
    # UAV1（upper） - UAV0（lower）
    rel_pos = np.stack([
        d1['pos_x'] - d0['pos_x'],
        d1['pos_y'] - d0['pos_y'],
        d1['pos_z'] - d0['pos_z'],
    ], axis=1)   # (N, 3)

    rel_vel = np.stack([
        d1['vel_x'] - d0['vel_x'],
        d1['vel_y'] - d0['vel_y'],
        d1['vel_z'] - d0['vel_z'],
    ], axis=1)   # (N, 3)

    rel_dist = np.linalg.norm(rel_pos, axis=1)

    # 网络推理
    fa_nn = run_network(phi_net, rho_net, rel_pos, rel_vel)   # (N,) Newton

    # 从 UAV0 日志取已记录的 Fa_z
    fa_logged = d0.get('Fa_z', np.zeros(N))   # Newton

    # ───────────────────── 打印统计 ──────────────────────
    print(f"\n{'─'*50}")
    print(f"样本数量        : {N}")
    print(f"时长            : {t_sec[-1]:.2f} s")
    print(f"相对距离 mean   : {rel_dist.mean():.3f} m  "
          f"  min: {rel_dist.min():.3f} m  max: {rel_dist.max():.3f} m")
    print(f"\n网络推理 Fa_z   : mean={fa_nn.mean()*1000/9.81:.3f} gf  "
          f"min={fa_nn.min()*1000/9.81:.3f} gf  max={fa_nn.max()*1000/9.81:.3f} gf")
    print(f"  (牛顿)          mean={fa_nn.mean():.5f} N  "
          f"min={fa_nn.min():.5f} N  max={fa_nn.max():.5f} N")
    print(f"\n日志记录 Fa_z   : mean={fa_logged.mean()*1000/9.81:.3f} gf  "
          f"min={fa_logged.min()*1000/9.81:.3f} gf  max={fa_logged.max()*1000/9.81:.3f} gf")
    print(f"  (牛顿)          mean={fa_logged.mean():.5f} N  "
          f"min={fa_logged.min():.5f} N  max={fa_logged.max():.5f} N")
    print(f"{'─'*50}\n")

    # ───────────────────── 绘图 ──────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        f"Neural Network Offline Test\n"
        f"UAV0: {path0.name}   UAV1: {path1.name}",
        fontsize=11
    )

    # --- 子图1：气动力对比 ---
    ax = axes[0]
    ax.plot(t_sec, fa_nn * 1000 / 9.81,     label='NN Estimated Fa_z', color='blue')
    ax.plot(t_sec, fa_logged * 1000 / 9.81, label='Logged Fa_z (from controller)',
            color='orange', linestyle='--', alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylabel('Fa_z (gf)')
    ax.set_title('Aerodynamic Force Comparison')
    ax.legend()
    ax.grid(True, alpha=0.4)

    # --- 子图2：相对位置三轴 ---
    ax = axes[1]
    ax.plot(t_sec, rel_pos[:, 0], label='Δx (UAV1−UAV0)', color='red')
    ax.plot(t_sec, rel_pos[:, 1], label='Δy',              color='green')
    ax.plot(t_sec, rel_pos[:, 2], label='Δz',              color='blue')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylabel('Relative Position (m)')
    ax.set_title('Relative Position (Upper − Lower)')
    ax.legend()
    ax.grid(True, alpha=0.4)

    # --- 子图3：相对距离 ---
    ax = axes[2]
    ax.plot(t_sec, rel_dist, color='purple', label='Relative Distance')
    ax.set_ylabel('Distance (m)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Inter-UAV Distance')
    ax.legend()
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
