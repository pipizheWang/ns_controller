"""
对预处理后的数据（data_input_true.npy / data_output_true.npy）进行后处理：
  1. 打乱数据顺序（保持输入-输出对应关系）
  2. 按扰动力大小分桶均衡采样，使各区间样本数尽量相等
  3. 保存为 data_input_final.npy / data_out_final.npy
"""

import numpy as np
import os

# ===== 配置 =====
DATA_DIR   = 'data'
INPUT_FILE = os.path.join(DATA_DIR, 'data_input_true.npy')
OUTPUT_FILE= os.path.join(DATA_DIR, 'data_output_true.npy')
SAVE_INPUT = os.path.join(DATA_DIR, 'data_input_final.npy')
SAVE_OUTPUT= os.path.join(DATA_DIR, 'data_out_final.npy')

N_BINS     = 10    # 按扰动力大小划分的区间数
SEED       = 42    # 随机种子，保证可复现

# ===== 加载数据 =====
data_input  = np.load(INPUT_FILE)
data_output = np.load(OUTPUT_FILE)

assert data_input.shape[0] == data_output.shape[0], \
    f"输入输出样本数不一致：{data_input.shape[0]} vs {data_output.shape[0]}"

print(f"原始输入形状 : {data_input.shape}")
print(f"原始输出形状 : {data_output.shape}")

# ===== Step 1：整体打乱（保持对应关系）=====
rng = np.random.default_rng(SEED)
perm = rng.permutation(data_input.shape[0])
data_input  = data_input[perm]
data_output = data_output[perm]
print(f"\n[Step 1] 数据已打乱（seed={SEED}）")

# ===== Step 2：按扰动力大小均衡采样 =====
# data_output shape: [L, 1]，取第0列作为扰动力幅值
force = data_output[:, 0]

# 使用等宽分桶（基于实际数据范围）
f_min, f_max = force.min(), force.max()
print(f"\n扰动力范围: [{f_min:.4f}, {f_max:.4f}]")

bin_edges  = np.linspace(f_min, f_max, N_BINS + 1)
bin_indices = np.digitize(force, bin_edges[1:-1])  # 0 ~ N_BINS-1

# 统计各桶样本数
bin_counts = np.array([np.sum(bin_indices == b) for b in range(N_BINS)])
print(f"\n各桶样本数（共 {N_BINS} 桶）:")
for b in range(N_BINS):
    lo = bin_edges[b]
    hi = bin_edges[b + 1]
    print(f"  桶 {b:2d} [{lo:8.4f}, {hi:8.4f}): {bin_counts[b]:6d} 个样本")

# 非空桶的最小样本数作为目标数量
non_empty_counts = bin_counts[bin_counts > 0]
target_per_bin   = int(non_empty_counts.min())
print(f"\n目标每桶保留样本数: {target_per_bin}")

# 对每个桶进行随机欠采样
keep_indices = []
for b in range(N_BINS):
    idx = np.where(bin_indices == b)[0]
    if len(idx) == 0:
        continue
    chosen = rng.choice(idx, size=target_per_bin, replace=False)
    keep_indices.append(chosen)

keep_indices = np.concatenate(keep_indices)

# 再次打乱（消除桶顺序）
keep_indices = rng.permutation(keep_indices)

data_input  = data_input[keep_indices]
data_output = data_output[keep_indices]

print(f"\n[Step 2] 均衡后输入形状 : {data_input.shape}")
print(f"[Step 2] 均衡后输出形状 : {data_output.shape}")

# 验证均衡效果
force_new  = data_output[:, 0]
bin_new    = np.digitize(force_new, bin_edges[1:-1])
print(f"\n均衡后各桶样本数:")
for b in range(N_BINS):
    lo = bin_edges[b]
    hi = bin_edges[b + 1]
    cnt = np.sum(bin_new == b)
    print(f"  桶 {b:2d} [{lo:8.4f}, {hi:8.4f}): {cnt:6d} 个样本")

# ===== 保存 =====
np.save(SAVE_INPUT,  data_input)
np.save(SAVE_OUTPUT, data_output)
print(f"\n✓ 已保存:")
print(f"  {SAVE_INPUT}")
print(f"  {SAVE_OUTPUT}")
