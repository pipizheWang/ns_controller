#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot npy data visualization
Input shape: (1717, 13)
Output shape: (1717, 3)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_npy_data(input_file, output_file, save_dir):
    """
    Plot input and output npy data
    
    Args:
        input_file: Input npy file path (1717, 13)
        output_file: Output npy file path (1717, 3)
        save_dir: Image save directory
    """
    # Load data
    input_data = np.load(input_file)
    output_data = np.load(output_file)
    
    print(f"Input data shape: {input_data.shape}")
    print(f"Output data shape: {output_data.shape}")
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Get sample count
    n_samples = input_data.shape[0]
    time_steps = np.arange(n_samples)
    
    # ========== Plot 13-dimensional input data ==========
    fig1, axes1 = plt.subplots(4, 4, figsize=(20, 16))
    axes1 = axes1.flatten()
    
    for i in range(13):
        axes1[i].plot(time_steps, input_data[:, i], linewidth=1.5)
        axes1[i].set_title(f'Input Dim {i+1}', fontsize=12, fontweight='bold')
        axes1[i].set_xlabel('Time Steps')
        axes1[i].set_ylabel('Value')
        axes1[i].grid(True, alpha=0.3)
        
        # Display statistics
        mean_val = np.mean(input_data[:, i])
        std_val = np.std(input_data[:, i])
        axes1[i].text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}',
                     transform=axes1[i].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                     fontsize=8)
    
    # Remove extra subplots
    for i in range(13, 16):
        fig1.delaxes(axes1[i])
    
    plt.tight_layout()
    plt.savefig(save_path / 'input_data_13dims.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path / 'input_data_13dims.png'}")
    plt.close()
    
    # ========== Plot 3-dimensional output data ==========
    fig2, axes2 = plt.subplots(3, 1, figsize=(15, 10))
    
    output_labels = ['Output Dim 1', 'Output Dim 2', 'Output Dim 3']
    colors = ['b', 'g', 'r']
    
    for i in range(3):
        axes2[i].plot(time_steps, output_data[:, i], color=colors[i], linewidth=1.5)
        axes2[i].set_title(output_labels[i], fontsize=14, fontweight='bold')
        axes2[i].set_xlabel('Time Steps', fontsize=12)
        axes2[i].set_ylabel('Value', fontsize=12)
        axes2[i].grid(True, alpha=0.3)
        
        # Display statistics
        mean_val = np.mean(output_data[:, i])
        std_val = np.std(output_data[:, i])
        min_val = np.min(output_data[:, i])
        max_val = np.max(output_data[:, i])
        axes2[i].text(0.02, 0.98, 
                     f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMin: {min_val:.3f}\nMax: {max_val:.3f}',
                     transform=axes2[i].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                     fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path / 'output_data_3dims.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path / 'output_data_3dims.png'}")
    plt.close()
    
    print("\nAll plots generated successfully!")
    print(f"Save location: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot npy data visualization')
    parser.add_argument('--input', type=str, required=True, help='Input npy file path (shape: N, 13)')
    parser.add_argument('--output', type=str, required=True, help='Output npy file path (shape: N, 3)')
    parser.add_argument('--save_dir', type=str, 
                       default='/home/zhe/px4_ws/src/ns_controller/train/log/',
                       help='Image save directory')
    
    args = parser.parse_args()
    
    plot_npy_data(args.input, args.output, args.save_dir)


if __name__ == '__main__':
    # Example usage:
    # python train/plot_npy_data.py --input input_data.npy --output output_data.npy
    
    main()


if __name__ == '__main__':
    # 示例用法
    # python plot_npy_data.py --input /home/zhe/px4_ws/src/ns_controller/train/log/data_input_L2L.npy --output /home/zhe/px4_ws/src/ns_controller/train/log/data_output_L2L.npy
    
    main()
