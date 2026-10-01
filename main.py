# main.py
import numpy as np
from env import HyperscaleGPUClusterEnv
from baselines import run_fifo_baseline, run_sjf_baseline, run_drf_baseline
from agents import MARLAgents
from utils import plot_performance_results

def main():
    print("=== 1. CHẠY THỬ NGHIỆM BASELINES (FIFO, SJF, DRF) THỰC TẾ ===")
    env_eval = HyperscaleGPUClusterEnv(num_nodes=4, gpus_per_node=8)
    
    fifo_m, fifo_u, fifo_s = run_fifo_baseline(env_eval)
    sjf_m, sjf_u, sjf_s = run_sjf_baseline(env_eval)
    drf_m, drf_u, drf_s = run_drf_baseline(env_eval)
    
    baselines_res = {
        'FIFO': (fifo_m, fifo_u, fifo_s),
        'SJF': (sjf_m, sjf_u, sjf_s),
        'DRF': (drf_m, drf_u, drf_s)
    }
    
    print(f"[FIFO Baseline] Makespan: {fifo_m} steps | GPU Util: {fifo_u:.1f}% | High-Prio SLA Violations: {fifo_s}")
    print(f"[SJF Baseline]  Makespan: {sjf_m} steps | GPU Util: {sjf_u:.1f}% | High-Prio SLA Violations: {sjf_s}")
    print(f"[DRF Baseline]  Makespan: {drf_m} steps | GPU Util: {drf_u:.1f}% | High-Prio SLA Violations: {drf_s}")

    print("\n=== 2. HUẤN LUYỆN HUẤN LUYỆN MARL AGENTS (DQN) ===")
    episodes = 150
    env = HyperscaleGPUClusterEnv(num_nodes=4, gpus_per_node=8, lookahead_window=5)
    
    # Input dimension: 8 GPUs * 2 features + 5 Lookahead Jobs * 4 features = 36
    obs_dim = 8 * 2 + (5 * 4) 
    marl_system = MARLAgents(num_nodes=4, obs_dim=obs_dim)
    
    epsilon = 0.4
    epsilon_decay = 0.98
    epsilon_min = 0.05
    target_update_freq = 10 # Đồng bộ Target Network mỗi 10 episode

    history = {'rewards': [], 'makespan': [], 'utilization': [], 'sla_violations': []}

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_ep_reward = 0
        steps = 0
        ep_utils = []

        while not done:
            actions = marl_system.select_actions(obs, epsilon)
            next_obs, rewards, done, info = env.step(actions)
            
            # Lưu vào Replay Buffer với reward dạng Vector (Credit Assignment)
            marl_system.buffer.push(obs, actions, rewards, next_obs, done)
            marl_system.train_step(batch_size=32)
            
            obs = next_obs
            total_ep_reward += sum(rewards)
            steps += 1
            ep_utils.append(info['gpu_utilization'])

        # Cập nhật Epsilon và Target Network
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if (ep + 1) % target_update_freq == 0:
            marl_system.update_target_network()

        history['rewards'].append(total_ep_reward)
        history['makespan'].append(steps)
        history['utilization'].append(np.mean(ep_utils))
        history['sla_violations'].append(info['high_priority_sla_violations'])

        if (ep + 1) % 25 == 0:
            print(f"Episode {ep+1}/{episodes} | Total Reward: {total_ep_reward:.1f} | Makespan: {steps} | GPU Util: {np.mean(ep_utils):.1f}%")

    print("\n=== 3. TRỰC QUAN HÓA KẾT QUẢ VÀ XUẤT ĐỒ THỊ ===")
    plot_performance_results(history, baselines_res)

if __name__ == "__main__":
    main()