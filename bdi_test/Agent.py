import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
import random
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class DQN(nn.Module):
    def __init__(self, input_size, action_size,hidden_size=32):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.output(x)

class Belief:
    def __init__(self, max_history=1000, mode="history"):
        """
        管理单个路径的完整历史观测值，并根据 mode 提供不同的数据。
        :param max_history: 最大历史记录长度，用于存储较大的历史数据。
        :param mode: 数据生成模式 ("history", "statistics", "peak_end")。
        """
        self.history = deque(maxlen=max_history)  # 存储完整历史数据
        self.max_time = float('-inf')
        self.min_time = float('inf')
        self.mode = mode  # 决定返回何种数据格式

    def update(self, observed_time):
        """
        更新观测值，并动态维护最大值和最小值。
        :param observed_time: 新的观测值。
        """
        if len(self.history) == self.history.maxlen:
            removed = self.history[0]
            if removed == self.max_time or removed == self.min_time:
                self._recalculate_min_max()

        self.history.append(observed_time)
        self.max_time = max(self.max_time, observed_time)
        self.min_time = min(self.min_time, observed_time)

    def _recalculate_min_max(self):
        """
        当最大值或最小值被移除时，重新计算 max_time 和 min_time。
        """
        if len(self.history) > 0:
            self.max_time = max(self.history)
            self.min_time = min(self.history)
        else:
            self.max_time = float('-inf')
            self.min_time = float('inf')

    def get_data(self, history_window=None):
        """
        根据模式和指定的窗口返回数据。
        :param history_window: 可选，决定返回的历史窗口大小。
        :return: 数据格式取决于 self.mode。
        """
        if history_window is None:
            history_window = self.history_window  # 如果未指定，则使用默认窗口大小

        # 获取最近的窗口数据
        window_data = list(self.history)[-history_window:]

        # 填充不足部分，确保长度始终为 history_window
        if len(window_data) < history_window:
            fill_value = self.history[0] if len(self.history) > 0 else 0
            window_data = [fill_value] * (history_window - len(window_data)) + window_data

        if self.mode == "history":
            return window_data

        elif self.mode == "statistics":
            # 计算窗口内均值和方差
            mean_time = np.mean(window_data) if len(window_data) > 0 else 0
            variance = np.var(window_data) if len(window_data) > 0 else 0
            return [mean_time, variance]

        elif self.mode == "peak_end":
            # 返回全局最大值、全局最小值和窗口数据
            max_time = self.max_time
            min_time = self.min_time
            return [max_time, min_time] + window_data

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")


class Agent:
    def __init__(self, agent_id, traits, paths, action_size, learning_rate=0.001, gamma=0.9, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, belief_mode="history", history_window=5):
        """
        初始化智能体。
        :param agent_id: 智能体 ID。
        :param traits: 智能体的特征（可扩展）。
        :param paths: 路径列表。
        :param action_size: 动作空间大小。
        :param learning_rate: 学习率。
        :param gamma: 折扣因子。
        :param epsilon: 初始探索率。
        :param epsilon_decay: 探索率衰减。
        :param epsilon_min: 最小探索率。
        :param belief_mode: 信念生成模式 ("history", "statistics", "peak_end")。
        :param history_window: 信念的历史窗口大小。
        """
        self.agent_id = agent_id
        self.traits = traits
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.belief_mode = belief_mode
        self.history_window = history_window

        # 管理多个路径的 Belief
        self.beliefs = {path: Belief(max_history=1000, mode=belief_mode) for path in paths}

        # 计算输入维度
        if self.belief_mode == "history":
            self.input_size = len(paths) * history_window
        elif self.belief_mode == "statistics":
            self.input_size = len(paths) * 2
        elif self.belief_mode == "peak_end":
            self.input_size = len(paths) * (history_window + 2)

        # 初始化 DQN 模型
        self.model = DQN(self.input_size, action_size).to(device)
        self.target_model = DQN(self.input_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=2000)

    def store_transition(self, state, action, reward, next_state):
        """将 (state, action, reward, next_state) 存储到经验池中"""
        self.memory.append((state, action, reward, next_state))

    def replay(self, batch_size):
        """从经验池中随机采样一个批次的经验进行训练"""
        if len(self.memory) < batch_size:
            return  # 如果经验池中的样本数不足一个批次，直接返回

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward
            if next_state is not None:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.target_model(next_state_tensor)).item()

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(state_tensor)
            q_value = q_values[0][action]

            target_tensor = torch.tensor(target).to(device)
            loss = self.criterion(q_value, target_tensor)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def decay_epsilon(self):
        """对 epsilon 进行衰减"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        """将主 Q 网络的权重复制到目标 Q 网络"""
        self.target_model.load_state_dict(self.model.state_dict())

    def add_observation(self, path, observed_time):
        """
        更新指定路径的信念。
        :param path: 路径名称。
        :param observed_time: 新的观测值。
        """
        if path not in self.beliefs:
            raise ValueError(f"Path {path} not found in beliefs.")
        self.beliefs[path].update(observed_time)

    def get_state(self):
        """
        生成当前状态向量。
        """
        state = []
        for belief in self.beliefs.values():
            state.extend(belief.get_data(history_window=self.history_window))
        return np.array(state, dtype=np.float32)

    def choose_action(self, state):
        """
        基于当前状态选择动作。
        :param state: 当前状态向量。
        :return: 选择的动作索引。
        """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def get_q_values(self):
        """
        获取当前状态下的 Q 值。
        """
        state = self.get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values[0].cpu().numpy()
