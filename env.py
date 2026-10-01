# env.py
import numpy as np
import random
#initialize group of nodes and gpu per node and related things 
class HyperscaleGPUClusterEnv:
    def __init__(self, num_nodes=4, gpus_per_node=8, max_steps=100, lookahead_window=5):
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.total_gpus = num_nodes * gpus_per_node
        self.max_steps = max_steps
        self.gpu_vram_capacity = 80 # GB
        self.lookahead_window = lookahead_window # observation for next 5 jobs to avoid head of line blocking
        self.reset()

    def reset(self):
        # State: [Node, GPU, Feature] -> Feature 0: Remaining time jobs, Feature 1: Used VRAM GB
        self.gpu_states = np.zeros((self.num_nodes, self.gpus_per_node, 2)) # create 3 dimensional array with value - 0
        self.current_step = 0
        self.job_queue = self._generate_job_batch(batch_size=40)
        self.completed_jobs = 0
        self.scheduled_jobs = 0
        self.high_priority_sla_violations = 0 #review service agreement violations 
        return self._get_observation()

    def _generate_job_batch(self, batch_size=40):
        jobs = []
        for j in range(batch_size):
            req_gpus = random.choice([1, 2, 4, 8]) # the requirement for GPU quantities
            duration = random.randint(3, 10) # time working randomly 3 or 10
            vram = random.choice([16, 32, 64, 80]) # memory allocted randomly
            priority = random.choice([1, 2, 3]) # High Priority following rules 1->2-> 3 'highest' 
            jobs.append({
                'id': j, 
                'req_gpus': req_gpus, 
                'duration': duration, 
                'vram': vram, 
                'priority': priority,
                'wait_time': 0,
                'status': 'queued'
            })
        return jobs

    def _get_observation(self):
        # Quan sát bổ sung thông tin các job phía sau trong hàng chờ (Lookahead)
        observations = []
        lookahead_jobs = []
        for idx in range(min(self.lookahead_window, len(self.job_queue))):
            j = self.job_queue[idx]
            lookahead_jobs.extend([j['req_gpus'], j['duration'], j['vram'], j['priority']])
        
        # Pad nếu không đủ lookahead_window
        pad_size = (self.lookahead_window * 4) - len(lookahead_jobs)
        if pad_size > 0:
            lookahead_jobs.extend([0] * pad_size)

        for n in range(self.num_nodes):
            node_gpu_flat = self.gpu_states[n].flatten()
            obs = np.concatenate([node_gpu_flat, lookahead_jobs])
            observations.append(obs)
        return np.array(observations, dtype=np.float32)

    def step(self, actions):
        """
        actions: list các action từ từng node agent cho job hiện tại ở đầu window
        Lưu ý: Đã tích hợp phạt nếu Agent nhận sai khi không đủ GPU/VRAM
        """
        self.current_step += 1
        rewards = [0.0] * self.num_nodes
        
        if len(self.job_queue) > 0:
            current_job = self.job_queue[0]
            scheduled = False

            for node_id, action in enumerate(actions):
                if action == 1 and not scheduled:
                    # Kiểm tra cả GPU rảnh VÀ VRAM Capacity
                    idle_gpus = np.where((self.gpu_states[node_id, :, 0] == 0) & 
                                         (current_job['vram'] <= self.gpu_vram_capacity))[0]
                    
                    if len(idle_gpus) >= current_job['req_gpus']:
                        # Phân bổ GPU hợp lệ
                        for g in idle_gpus[:current_job['req_gpus']]:
                            self.gpu_states[node_id, g, 0] = current_job['duration']
                            self.gpu_states[node_id, g, 1] = current_job['vram']
                        
                        # LOCAL CREDIT ASSIGNMENT: Thưởng riêng cho Node thực hiện thành công
                        rewards[node_id] += 10.0 * current_job['priority'] # depend on weight priorty
                        scheduled = True
                        self.scheduled_jobs += 1
                        self.job_queue.pop(0)
                        break
                    else:
                        # PHẠT NẶNG nếu Agent chọn "1" (Nhận) nhưng KHÔNG ĐỦ GPU/VRAM
                        rewards[node_id] -= 5.0

            if not scheduled:
                # Phạt nhẹ tất cả agent nếu job không xếp được lịch
                for n in range(self.num_nodes):
                    rewards[n] -= 0.5 * current_job['priority']

        # Tiến trình mô phỏng & Đếm job thực sự HOÀN THÀNH (Completed Jobs)
        for n in range(self.num_nodes):
            for g in range(self.gpus_per_node):
                if self.gpu_states[n, g, 0] > 0:
                    self.gpu_states[n, g, 0] -= 1
                    if self.gpu_states[n, g, 0] == 0:
                        self.gpu_states[n, g, 1] = 0
                        self.completed_jobs += 1 # Đếm chính xác khi job chạy xong!

        # Tăng wait_time và kiểm tra SLA riêng cho HIGH-PRIORITY (Priority == 3)
        for job in self.job_queue:
            job['wait_time'] += 1
            if job['priority'] == 3 and job['wait_time'] == 10:
                self.high_priority_sla_violations += 1

        done = (len(self.job_queue) == 0) or (self.current_step >= self.max_steps)
        
        active_gpus = np.sum(self.gpu_states[:, :, 0] > 0)
        gpu_utilization = (active_gpus / self.total_gpus) * 100.0

        info = {
            'gpu_utilization': gpu_utilization,
            'completed_jobs': self.completed_jobs,
            'high_priority_sla_violations': self.high_priority_sla_violations
        }

        return self._get_observation(), rewards, done, info