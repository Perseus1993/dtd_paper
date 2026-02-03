from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
import random
from matplotlib import pyplot as plt
import pandas as pd



# 检查 GPU 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.output = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.output(x)


class Agent:
    def __init__(self, agent_id, traits, paths, state_size, action_size, learning_rate=0.001, gamma=0.9, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01):
        self.agent_id = agent_id
        self.traits = traits
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # 初始化 belief, 记录池 和 DQN
        self.belief = self.initialize_belief(paths)
        self.record_pool = self.initialize_record_pool()
        self.memory = deque(maxlen=2000)

        # 初始化 Q 网络和目标网络
        self.model = DQN(state_size, action_size).to(device)  # 将模型放到GPU上
        self.target_model = DQN(state_size, action_size).to(device)  # 将目标模型放到GPU上
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def initialize_belief(self, paths):
        belief = {}
        for path_id, path_info in paths.items():
            belief[path_id] = {
                # 'mean_time': path_info['free_flow_time'],
                'mean_time': 0,
                'variance': 0.5
            }
        return belief

    def initialize_record_pool(self):
        return defaultdict(list)

    def add_observation(self, path, observed_time):
        self.record_pool[path].append(observed_time)
        self.update_belief(path)

    def update_belief(self, path, memory_window=None):
        times = self.record_pool[path][-memory_window:] if memory_window else self.record_pool[path]

        if len(times) > 0:
            mean_time = np.mean(times)
            variance = np.var(times)
        else:
            mean_time, variance = self.belief[path]['mean_time'], self.belief[path]['variance']

        self.belief[path]['mean_time'] = mean_time
        self.belief[path]['variance'] = variance

    def get_state(self):
        state = []
        for path in self.belief:
            state.append(self.belief[path]['mean_time'])
            state.append(self.belief[path]['variance'])
        return np.array(state, dtype=np.float32)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)

        # 将状态转到GPU上
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward

            # 将next_state转移到GPU上
            if next_state is not None:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.target_model(next_state_tensor)).item()

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(state_tensor)
            q_value = q_values[0][action]

            # 将target转移到GPU上
            loss = self.criterion(q_value, torch.tensor(target).to(device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_q_values(self):
        state = self.get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            return self.model(state_tensor).squeeze().cpu().numpy()  # 将结果转回CPU进行输出

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)


# 初始化100个代理人等后续代码保持不变

paths = {
    'P1': {'free_flow_time': 3.5, 'capacity': 9},  # 主干道，适合高流量
    'P2': {'free_flow_time': 3, 'capacity': 4},  # 辅助路径，适合中等流量
    'P3': {'free_flow_time': 2, 'capacity': 3},  # 快速但易拥挤的小路
    # 'P4': {'free_flow_time': 3.5, 'capacity': 4},   # 备用路径
    'P5': {'free_flow_time': 30, 'capacity': 1},
}


# BPR 函数
def bpr_function(free_flow_time, volume, capacity, alpha=0.15, beta=4):
    return free_flow_time * (1 + alpha * (volume / capacity) ** beta)


# 初始化100个代理人
num_agents = 20
agents = [
    Agent(
        agent_id=i,
        traits={'info_processing': 1.0},
        paths=paths,
        state_size=8,
        action_size=4,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    ) for i in range(num_agents)
]

num_episodes = 400
path_travel_times = {path: [] for path in paths.keys()}  # 存储每回合每条路径的通行时间

# 模拟多个代理人并行学习
batch_size = 32

# 用于存储每隔10个回合的Q值平均变化
q_values_over_time = {path: [] for path in paths.keys()}

# 用于记录每个回合每个代理人的路径选择
choices_over_time = []  # 每个元素是一个列表，记录每个回合所有代理人的选择

# 修改主循环，记录每隔10回合的Q值
for episode in tqdm(range(num_episodes)):
    # 重置路径流量计数
    path_volumes = {path: 0 for path in paths.keys()}

    # 记录当前回合的所有代理人选择
    current_choices = []

    # 每个代理人选择路径并更新流量
    for agent in agents:
        state = agent.get_state()
        action = agent.choose_action(state)

        chosen_path = list(paths.keys())[action]
        path_volumes[chosen_path] += 1  # 统计选择该路径的流量

        # 记录该代理人的选择
        current_choices.append(chosen_path)

    # 将当前回合的选择记录添加到总选择记录中
    choices_over_time.append(current_choices)

    # 计算并记录每条路径的实际通行时间（基于 BPR 函数）
    current_travel_times = {}
    for path, volume in path_volumes.items():
        travel_time = bpr_function(
            paths[path]['free_flow_time'],
            volume,
            paths[path]['capacity']
        )
        current_travel_times[path] = travel_time
        path_travel_times[path].append(travel_time)  # 存储到可视化数据中

    # 每个代理人观察基于流量的实际通行时间
    for agent in agents:
        state = agent.get_state()
        action = agent.choose_action(state)
        chosen_path = list(paths.keys())[action]

        # 使用当前记录的通行时间
        actual_time = current_travel_times[chosen_path]

        # 更新代理人的记录池和 belief
        agent.add_observation(chosen_path, actual_time)
        next_state = agent.get_state()

        # 计算奖励并存储经验
        reward = -actual_time  # 使用负通行时间作为奖励
        agent.store_transition(state, action, reward, next_state)

        # 回放训练
        agent.replay(batch_size)

    # 每 10 回合更新所有代理人的目标网络
    if (episode + 1) % 10 == 0:
        for agent in agents:
            agent.update_target_model()

    # 每隔 10 回合记录所有代理人的Q值
    if (episode + 1) % 10 == 0:
        average_q_values = {path: 0 for path in paths.keys()}

        # 累积每个代理人的Q值
        for agent in agents:
            q_values = agent.get_q_values()
            for i, path in enumerate(paths.keys()):
                average_q_values[path] += q_values[i]

        # 计算平均Q值
        for path in paths.keys():
            average_q_values[path] /= num_agents
            q_values_over_time[path].append(average_q_values[path])

    # 衰减每个代理人的 epsilon
    for agent in agents:
        agent.decay_epsilon()

q_values_over_time = {path: np.round(values, 2).tolist() for path, values in q_values_over_time.items()}

# 绘制Q值随时间的变化
for path in paths.keys():
    print(path)
    x = range(len(q_values_over_time[path]))
    y = q_values_over_time[path]
    print(x)
    print(y)

for i, agent in enumerate(agents):
    print(f"Agent {i} Q-values: {agent.get_q_values()}")

# 将 choices_over_time 转换为 DataFrame，每列代表一个代理人的选择，每行代表一个回合
choices_df = pd.DataFrame(choices_over_time).T  # 转置，行表示回合，列表示代理人

# 选择要可视化的代理人，例如前5个代理人
num_agents_to_plot = 10
plt.figure(figsize=(12, 6))

for agent_id in range(num_agents_to_plot):
    plt.plot(choices_df[agent_id], label=f'Agent {agent_id}')

plt.xlabel('Episode')
plt.ylabel('Path Chosen')
plt.title("Path Choice per Agent over Episodes")
plt.show()
