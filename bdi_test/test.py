#ET 是一种“预测”或“预期”，它基于历史经验来估计。
#PT 是当前的“现实”或“实际体验”，它反映了旅行者当前所选择路径的真实旅行时间。

import random
import matplotlib.pyplot as plt
import numpy as np

# 定义路径及其基础长度和容量
paths = {
    'P1': {'free_flow_time': 3.5, 'capacity': 90},
    'P2': {'free_flow_time': 3, 'capacity': 100},
    'P3': {'free_flow_time': 2, 'capacity': 100},
    'P4': {'free_flow_time': 3.5, 'capacity': 200},
}

total_demand = 300 # 总OD需求（总代理数）
exploration_rate = 0.01  # 探索行为的概率
memory_level = 0.8  # ψ: 旅行者的记忆水平，控制权重衰减

class BDI_Agent:
    next_id = 0
    def __init__(self, learning_rate,initial_aspiration):
        self.id = BDI_Agent.next_id
        BDI_Agent.next_id += 1
        self.beliefs = {path: 1 / len(paths) for path in paths}  # 初始化为均匀概率分布
        self.intention = random.choice(list(self.beliefs.keys()))  # 初始化随机选择一条路径
        self.travel_history = {p: [] for p in paths}  # 记录每条路径的旅行时间历史
        self.learning_rate = learning_rate  # 学习速率，控制信念更新的速度
        self.aspiration = initial_aspiration  # 初始化全局 Aspiration

    def calculate_stimulus(self, PT, current_path):
        """根据全局 Aspiration 和 PT 计算刺激值"""
        diff = self.aspiration - PT  # 使用全局 Aspiration 计算差异

        if diff >= 0:
            # 正面反馈：当前路径表现优于预期，计算最大可能的收益
            max_benefit = max([self.aspiration - self.calculate_PT(p) for p in paths if p != current_path])
            stimulus = diff / max(max_benefit, 1e-10)  # 防止除以零
        else:
            # 负面反馈：当前路径表现不如预期，计算最小可能的损失
            min_loss = min([self.aspiration - self.calculate_PT(p) for p in paths if p != current_path])
            stimulus = diff / max(abs(min_loss), 1e-10)  # 防止除以零

        return stimulus

    def calculate_PT(self, path):
        """计算指定路径的感知旅行时间（PT）"""
        times = self.travel_history[path]
        if times:
            # 使用记忆衰减权重计算感知旅行时间
            weights = [memory_level ** (len(times) - 1 - j) for j in range(len(times))]
            PT = sum(t * w for t, w in zip(times, weights)) / sum(weights)
        else:
            # 如果没有历史数据，使用路径的自由流时间作为PT
            PT = paths[path]['free_flow_time']
        return PT

    def update_aspiration(self, PT):
        """基于最近的 PT 动态更新全局 Aspiration"""
        self.aspiration = self.aspiration + self.learning_rate * (PT - self.aspiration)

    def update_probabilities(self, stimulus):
        for path in self.beliefs:
            if path == self.intention:
                if stimulus >= 0:
                    self.beliefs[path] += (1 - self.beliefs[path]) * self.learning_rate * stimulus
                else:
                    self.beliefs[path] -= self.beliefs[path] * self.learning_rate * abs(stimulus)
            else:
                self.beliefs[path] -= self.learning_rate * abs(stimulus) * self.beliefs[path]

            # 避免信念值为负
            self.beliefs[path] = max(0, self.beliefs[path])

        # 确保信念总和为正
        total_belief = sum(self.beliefs.values())
        if total_belief == 0:
            # 如果总和为零，重新分配信念值
            self.beliefs = {path: 1 / len(paths) for path in self.beliefs}
            total_belief = 1

        # 确保概率总和为1
        for path in self.beliefs:
            self.beliefs[path] /= total_belief

    def choose_path(self):
        # 引入探索行为
        if random.random() < exploration_rate:
            self.intention = random.choice(list(self.beliefs.keys()))
        else:
            self.intention = random.choices(list(self.beliefs.keys()), weights=list(self.beliefs.values()))[0]
        return self.intention


# 初始化代理，反映总OD需求
agents = [BDI_Agent(learning_rate=0.1,initial_aspiration=4) for _ in range(total_demand)]

def calculate_travel_times(agents):
    path_counts = {path: 0 for path in paths}
    for agent in agents:
        path_counts[agent.intention] += 1

    path_times = {}
    for path, info in paths.items():
        x_a = path_counts[path]
        c_a = info['capacity']
        t0_a = info['free_flow_time']
        path_times[path] = t0_a * (1 + 0.15 * (x_a / c_a) ** 4)  # BPR函数

    return path_times, path_counts

# 保存每轮的路径选择结果
all_path_counts = []
all_path_times = []

# 多轮模拟
target_agent_id = 0  # 目标代理的ID

for round_num in range(700):
    print("--------------------")
    print(f"\nRound {round_num + 1}")

    path_times, current_path_counts = calculate_travel_times(agents)
    print(f"Road times: {path_times}")

    for i, agent in enumerate(agents):
        # 更新历史旅行时间
        agent.travel_history[agent.intention].append(path_times[agent.intention])

        PT = path_times[agent.intention]
        stimulus = agent.calculate_stimulus(PT, agent.intention)
        agent.update_probabilities(stimulus)  # 更新信念
        agent.choose_path()  # 选择路径

        if i == target_agent_id:
            print(f"aspirations: {agent.aspiration}")
            print(f"Agent {i} chose path {agent.intention} with stimulus {stimulus}")
            print(f"Agent {i} beliefs: {agent.beliefs}")

    all_path_counts.append(list(current_path_counts.values()))
    all_path_times.append(list(path_times.values()))

# 转换为 NumPy 数组以便于绘图
all_path_counts = np.array(all_path_counts)
all_path_times = np.array(all_path_times)

# 绘制两张图
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for i, path in enumerate(paths):
    plt.plot(all_path_counts[:, i], label=path)  # 确保 all_path_counts 是 NumPy 数组
plt.title("Path Counts")
plt.xlabel("Round")
plt.ylabel("Count")
plt.legend(paths.keys())

plt.subplot(1, 2, 2)
for i, path in enumerate(paths):
    plt.plot(all_path_times[:, i], label=path)  # 确保 all_path_times 是 NumPy 数组
plt.title("Path Times")
plt.xlabel("Round")
plt.ylabel("Time")
plt.legend(paths.keys())

plt.tight_layout()
plt.show()
