from bdi_test.Agent import Agent

if __name__ == '__main__':

    # BPR 函数
    def bpr_function(free_flow_time, volume, capacity, alpha=0.15, beta=4):
        return free_flow_time * (1 + alpha * (volume / capacity) ** beta)

    paths = {
        'P1': {'free_flow_time': 3.5, 'capacity': 9},   # 主干道，适合高流量
        'P2': {'free_flow_time': 3, 'capacity': 4},     # 辅助路径，适合中等流量
        'P3': {'free_flow_time': 2, 'capacity': 3},     # 快速但易拥挤的小路
        'P4': {'free_flow_time': 5, 'capacity': 6},     # 中等通行时间、较大容量的路径
        'P5': {'free_flow_time': 10, 'capacity': 2},    # 备用路径，适中的通行时间和容量，增加选择的合理性
    }

    num_episodes = 100
    batch_size = 32
    history_window = 10

    # 初始化路径的通行时间记录
    path_travel_times = {path: [] for path in paths.keys()}

    # 初始化 Agent
    agent = Agent(
        agent_id=0,
        traits={'info_processing': 1.0},
        paths=paths,
        action_size=len(paths),
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        belief_mode="history",  # 或 "statistics"
        history_window=history_window
    )

    agents = [agent]  # 只有一个代理
    choices_over_time = []  # 记录每回合代理的路径选择
    q_values_over_time = {path: [] for path in paths.keys()}  # 每条路径的 Q 值随时间变化

    for episode in range(num_episodes):
        # 路径流量统计
        path_volumes = {path: 0 for path in paths.keys()}
        current_choices = []

        # 每个代理选择路径并更新流量统计
        for agent in agents:
            state = agent.get_state()
            action = agent.choose_action(state)
            chosen_path = list(paths.keys())[action]
            path_volumes[chosen_path] += 1
            current_choices.append(chosen_path)

        choices_over_time.append(current_choices)  # 记录本轮的选择

        # 使用 BPR 函数计算每条路径的实际通行时间
        current_travel_times = {}
        for path, volume in path_volumes.items():
            travel_time = bpr_function(
                paths[path]['free_flow_time'], volume, paths[path]['capacity']
            )
            current_travel_times[path] = travel_time
            path_travel_times[path].append(travel_time)  # 记录通行时间

        # 更新代理的信念和经验池
        for agent in agents:
            state = agent.get_state()
            action = agent.choose_action(state)
            chosen_path = list(paths.keys())[action]

            # 获取实际的通行时间作为奖励的依据
            actual_time = current_travel_times[chosen_path]
            agent.add_observation(chosen_path, actual_time)  # 更新 Belief
            next_state = agent.get_state()

            # 计算奖励，通行时间越短奖励越高
            reward = -actual_time
            agent.store_transition(state, action, reward, next_state)
            agent.replay(batch_size)  # 使用经验回放更新模型

        # 每隔 10 回合更新目标网络
        if (episode + 1) % 10 == 0:
            for agent in agents:
                agent.update_target_model()
                q_values = agent.get_q_values()
                for i, path in enumerate(paths.keys()):
                    q_values_over_time[path].append(q_values[i])

        # 更新每个代理的 epsilon
        for agent in agents:
            agent.decay_epsilon()

    # 输出最终的 Q 值和选择记录
    print("Q-values over time:", q_values_over_time)
    print("Choices over time:", choices_over_time)
