# baselines.py
import numpy as np
import copy

def run_fifo_baseline(env_orig):
    env = copy.deepcopy(env_orig)
    obs = env.reset()
    done = False
    total_makespan = 0
    
    while not done:
        actions = [0] * env.num_nodes
        # FIFO Logic: Cố gắng xếp job đầu hàng vào Node đầu tiên thỏa điều kiện
        if len(env.job_queue) > 0:
            job = env.job_queue[0]
            for n in range(env.num_nodes):
                idle_gpus = np.where((env.gpu_states[n, :, 0] == 0) & (job['vram'] <= env.gpu_vram_capacity))[0]
                if len(idle_gpus) >= job['req_gpus']:
                    actions[n] = 1
                    break
        obs, rewards, done, info = env.step(actions)
        total_makespan += 1

    return total_makespan, info['gpu_utilization'], info['high_priority_sla_violations']

def run_sjf_baseline(env_orig):
    env = copy.deepcopy(env_orig)
    obs = env.reset()
    # Thao tác sắp xếp lại Job Queue theo Duration (SJF)
    env.job_queue.sort(key=lambda x: x['duration'])
    done = False
    total_makespan = 0
    
    while not done:
        actions = [0] * env.num_nodes
        if len(env.job_queue) > 0:
            job = env.job_queue[0]
            for n in range(env.num_nodes):
                idle_gpus = np.where((env.gpu_states[n, :, 0] == 0) & (job['vram'] <= env.gpu_vram_capacity))[0]
                if len(idle_gpus) >= job['req_gpus']:
                    actions[n] = 1
                    break
        obs, rewards, done, info = env.step(actions)
        total_makespan += 1

    return total_makespan, info['gpu_utilization'], info['high_priority_sla_violations']

def run_drf_baseline(env_orig):
    env = copy.deepcopy(env_orig)
    obs = env.reset()
    done = False
    total_makespan = 0
    
    while not done:
        actions = [0] * env.num_nodes
        if len(env.job_queue) > 0:
            # DRF Logic: Ưu tiên chọn Node có dominant resource utilization ít nhất
            job = env.job_queue[0]
            best_node = -1
            min_util = 999.0

            for n in range(env.num_nodes):
                idle_gpus = np.where((env.gpu_states[n, :, 0] == 0) & (job['vram'] <= env.gpu_vram_capacity))[0]
                if len(idle_gpus) >= job['req_gpus']:
                    node_util = np.mean(env.gpu_states[n, :, 0] > 0)
                    if node_util < min_util:
                        min_util = node_util
                        best_node = n

            if best_node != -1:
                actions[best_node] = 1

        obs, rewards, done, info = env.step(actions)
        total_makespan += 1

    return total_makespan, info['gpu_utilization'], info['high_priority_sla_violations']