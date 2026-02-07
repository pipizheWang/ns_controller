"""
合并训练数据文件
将三个输入文件和三个输出文件分别合并成一个大文件
确保每一帧输入数据对应的输出不出错
"""

import numpy as np
import os

def merge_data():
    # 定义数据目录
    data_dir = "data"
    
    # 定义输入文件列表（按顺序）
    input_files = [
        "data_input_x.npy",
        "data_input_xy.npy",
        "data_input_y.npy"
    ]
    
    # 定义输出文件列表（按相同顺序对应）
    output_files = [
        "data_output_x.npy",
        "data_output_xy.npy",
        "data_output_y.npy"
    ]
    
    # 加载所有输入数据
    print("正在加载输入数据...")
    input_data_list = []
    for input_file in input_files:
        file_path = os.path.join(data_dir, input_file)
        data = np.load(file_path)
        print(f"  {input_file}: shape = {data.shape}")
        input_data_list.append(data)
    
    # 加载所有输出数据
    print("\n正在加载输出数据...")
    output_data_list = []
    for output_file in output_files:
        file_path = os.path.join(data_dir, output_file)
        data = np.load(file_path)
        print(f"  {output_file}: shape = {data.shape}")
        output_data_list.append(data)
    
    # 验证输入输出配对的数据量是否一致
    print("\n验证数据对应关系...")
    for i, (inp, out) in enumerate(zip(input_data_list, output_data_list)):
        if inp.shape[0] != out.shape[0]:
            print(f"警告：第{i+1}对数据的样本数不匹配！")
            print(f"  输入样本数: {inp.shape[0]}, 输出样本数: {out.shape[0]}")
            return
        else:
            print(f"  第{i+1}对数据验证通过：{inp.shape[0]} 个样本")
    
    # 沿着第0维（样本维度）合并数据
    print("\n正在合并数据...")
    merged_input = np.concatenate(input_data_list, axis=0)
    merged_output = np.concatenate(output_data_list, axis=0)
    
    print(f"合并后的输入数据 shape: {merged_input.shape}")
    print(f"合并后的输出数据 shape: {merged_output.shape}")
    
    # 保存合并后的数据
    merged_input_path = os.path.join(data_dir, "data_input_merged.npy")
    merged_output_path = os.path.join(data_dir, "data_output_merged.npy")
    
    print(f"\n正在保存合并后的数据...")
    np.save(merged_input_path, merged_input)
    print(f"  输入数据已保存到: {merged_input_path}")
    
    np.save(merged_output_path, merged_output)
    print(f"  输出数据已保存到: {merged_output_path}")
    
    # 显示合并顺序
    print("\n=== 数据合并顺序 ===")
    cumsum = 0
    for i, (inp_file, out_file) in enumerate(zip(input_files, output_files)):
        n_samples = input_data_list[i].shape[0]
        print(f"索引 [{cumsum}:{cumsum + n_samples}] : {inp_file} <-> {out_file}")
        cumsum += n_samples
    
    print("\n✓ 数据合并完成！输入输出对应关系已保持一致。")

if __name__ == "__main__":
    merge_data()
