# utils.py
import matplotlib.pyplot as plt
import numpy as np

def plot_performance_results(history, baselines_res):
    """
    Vẽ 4 đồ thị đánh giá hiệu năng khoa học dựa trên số liệu CHẠY THẬT
    """
    episodes_arr = list(range(1, len(history['rewards']) + 1))
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Đồ thị 1: Training Convergence
    axs[0, 0].plot(episodes_arr, history['rewards'], color='#1f77b4', alpha=0.4, label='Raw Reward')
    window = 10
    if len(history['rewards']) >= window:
        smoothed = np.convolve(history['rewards'], np.ones(window)/window, mode='valid')
        axs[0, 0].plot(range(window, len(history['rewards'])+1), smoothed, color='#d62728', linewidth=2, label='Smoothed Trend')
    axs[0, 0].set_title("1. MARL Training Convergence (Credit Assignment Reward)")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Total Reward")
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].legend()

    # Đồ thị 2: Makespan Comparison (Chạy THẬT)
    methods = ['FIFO', 'SJF', 'DRF', 'MARL (Ours)']
    marl_makespan = np.mean(history['makespan'][-20:])
    makespans = [baselines_res['FIFO'][0], baselines_res['SJF'][0], baselines_res['DRF'][0], marl_makespan]
    colors = ['#7f7f7f', '#ff7f0e', '#9467bd', '#2ca02c']
    bars = axs[0, 1].bar(methods, makespans, color=colors)
    axs[0, 1].set_title("2. True Workload Makespan Comparison (Empirical Run)")
    axs[0, 1].set_ylabel("Time Steps (Lower is Better)")
    axs[0, 1].grid(axis='y', linestyle='--', alpha=0.6)
    for bar in bars:
        yval = bar.get_height()
        axs[0, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}", ha='center', va='bottom')

    # Đồ thị 3: Cluster GPU Utilization
    axs[1, 0].plot(episodes_arr, history['utilization'], color='#2ca02c', linewidth=1.5)
    final_util = np.mean(history['utilization'][-20:])
    axs[1, 0].axhline(y=final_util, color='red', linestyle='--', label=f'MARL Final Avg: {final_util:.1f}%')
    axs[1, 0].set_title("3. Average GPU Cluster Utilization (%)")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Utilization Rate (%)")
    axs[1, 0].set_ylim(0, 100)
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    axs[1, 0].legend()

    # Đồ thị 4: High-Priority SLA Violations (Đếm ĐÚNG Priority = 3)
    axs[1, 1].plot(episodes_arr, history['sla_violations'], color='#d62728', linewidth=1.5)
    axs[1, 1].set_title("4. High-Priority Job SLA Violations (Priority = 3)")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Violation Count")
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()