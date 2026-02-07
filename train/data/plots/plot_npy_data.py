#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot npy data visualization
Automatically adapts to any input/output dimensions
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_npy_data(input_file, output_file, save_dir):
    """
    Plot input and output npy data (automatically adapts to any dimensions)
    
    Args:
        input_file: Input npy file path (N, D_in)
        output_file: Output npy file path (N, D_out)
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
    
    # Get sample count and dimensions
    n_samples = input_data.shape[0]
    n_input_dims = input_data.shape[1]
    n_output_dims = output_data.shape[1]
    time_steps = np.arange(n_samples)
    
    # ========== Plot input data ==========
    # Calculate subplot layout dynamically
    n_cols = min(4, n_input_dims)
    n_rows = (n_input_dims + n_cols - 1) // n_cols  # Ceiling division
    
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_input_dims == 1:
        axes1 = np.array([axes1])
    else:
        axes1 = axes1.flatten()
    
    for i in range(n_input_dims):
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
    total_subplots = n_rows * n_cols
    for i in range(n_input_dims, total_subplots):
        fig1.delaxes(axes1[i])
    
    plt.tight_layout()
    plt.savefig(save_path / f'input_data_{n_input_dims}dims.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path / f'input_data_{n_input_dims}dims.png'}")
    plt.close()
    
    # ========== Plot output data ==========
    fig2, axes2 = plt.subplots(n_output_dims, 1, figsize=(15, 4*n_output_dims))
    if n_output_dims == 1:
        axes2 = np.array([axes2])
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    
    for i in range(n_output_dims):
        color = colors[i % len(colors)]
        axes2[i].plot(time_steps, output_data[:, i], color=color, linewidth=1.5)
        axes2[i].set_title(f'Output Dim {i+1}', fontsize=14, fontweight='bold')
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
    plt.savefig(save_path / f'output_data_{n_output_dims}dims.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path / f'output_data_{n_output_dims}dims.png'}")
    plt.close()
    
    print("\nAll plots generated successfully!")
    print(f"Save location: {save_path}")


def main():
    # Get script directory and calculate paths
    script_dir = Path(__file__).parent.resolve()  # train/data/plots/
    data_dir = script_dir.parent  # train/data/
    
    parser = argparse.ArgumentParser(description='Plot npy data visualization')
    parser.add_argument('--input', type=str, 
                       default=str(data_dir / 'data_input_all.npy'),
                       help='Input npy file path (shape: N, D_in)')
    parser.add_argument('--output', type=str, 
                       default=str(data_dir / 'data_output_all.npy'),
                       help='Output npy file path (shape: N, D_out)')
    parser.add_argument('--save_dir', type=str, 
                       default=str(script_dir),  # Save in data/plots/
                       help='Image save directory')
    
    args = parser.parse_args()
    
    plot_npy_data(args.input, args.output, args.save_dir)


if __name__ == '__main__':
    # Example usage:
    # python plot_npy_data.py --input data_input_L2L.npy --output data_output_L2L.npy
    # Or just run directly: python plot_npy_data.py
    
    main()
