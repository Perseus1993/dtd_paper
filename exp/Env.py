import logging
import sys
from collections import deque
import random

import networkx as nx
import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from exp.Node import NodeType
from exp.agent import Agent, AgentStatus, Action
from exp.traffic import Traffic_State
from exp.util import print_banner
from exp.dqn import DQNNetwork
import torch.nn.functional as F


class RFState:
    def __init__(self, cur_loc, cur_act_order, cur_act_start_time, cur_time):
        self.cur_loc = cur_loc
        self.cur_act_order = cur_act_order
        self.cur_act_start_time = cur_act_start_time
        self.cur_time = cur_time

    def __repr__(self):
        return f"RFState(cur_loc={self.cur_loc},cur_act_order={self.cur_act_order},cur_act_start_time={self.cur_act_start_time},cur_time={self.cur_time})"

    def get_info(self):
        return self.cur_loc, self.cur_act_order, self.cur_act_start_time, self.cur_time


def get_action_space(G: nx.Graph):
    node_num = []
    for node in G.nodes:
        node_num.append(node)
    return node_num


class Env:
    def __init__(self, G: nx.Graph, nodes_dict: dict, max_schedule_num, mode='single', num_agents=1,
                 method='q_learning'):
        self.mode = mode
        self.G = G
        self.nodes_dict = nodes_dict
        self.max_schedule_num = max_schedule_num
        self.current_time = 0
        self.action_space = get_action_space(G)
        self.reward_cur_epoch = {}

        # 找到所有的HOME节点
        self.home_nodes = [node for node in self.nodes_dict.values() if node.node_type == NodeType.HOME]
        # print(f"HOME nodes: {self.home_nodes}")
        for node in self.home_nodes:
            print(f"Home Node: {node} generated.")
        self.agents = []

        print(f"Action space: {self.action_space}")

        self.Q_table = {}
        self.observation_space = {}
        print_banner()

        print("___________________env created__________________________")
        self.generate_rl_env()
        print("___________________rl initiate__________________________")
        self.traffic_state = Traffic_State(G)
        print("___________________traffic state initiate________________")

    def generate_rl_env(self):
        # assert self.agents != [], "agents列表不能为空"
        # 时间戳 * 位置
        self.observation_space['timestamp'] = [i for i in range(241)]
        self.observation_space['location'] = [node.id for node in self.nodes_dict.values()]
        self.observation_space['current_activity_index'] = [i for i in range(self.max_schedule_num)]
        self.observation_space['current_activity_start_time'] = [i for i in range(240)]

        possible_states = []

        for cur_loc in self.observation_space['location']:
            for cur_act_order in self.observation_space['current_activity_index']:
                for start_time in self.observation_space['current_activity_start_time']:
                    for cur_time in self.observation_space['timestamp']:
                        possible_states.append((cur_loc, cur_act_order, start_time, cur_time))
        logging.debug(f"Possible states: {len(possible_states)}")

        for state in tqdm(possible_states):
            self.Q_table[state] = {}
            for action in self.action_space:
                self.Q_table[state][action] = 0.0  # 初始化Q值为0

    def generate_agent(self, default_schedule, num_agents, home):
        if self.mode == 'single':
            self.agents.append(Agent(default_schedule, home, self.G, self.nodes_dict))
            self.agents[0].available_actions = self.action_space
        else:
            # num_home_nodes = len(self.home_nodes)
            num_home_nodes = 1
            # random number of home nodes
            home_generated_num = [num_agents for _ in range(num_home_nodes)]
            for i in range(1):
                agents_in_this_home = home_generated_num[i]
                for j in range(agents_in_this_home):
                    cur_agent = Agent(default_schedule, self.home_nodes[i].id, self.G, self.nodes_dict)
                    cur_agent.available_actions = self.action_space
                    self.agents.append(cur_agent)

    def reset(self):
        self.current_time = 0
        self.reward_cur_epoch = {}
        self.traffic_state.clear_traffic()
        state_list = []
        for agent in self.agents:
            agent.reset()
            cur_loc, cur_act_order, current_activity_duration = agent.get_rf_state()
            cur_time = self.current_time
            cur_act_start_time = cur_time - current_activity_duration
            state_list.append(RFState(cur_loc, cur_act_order, cur_act_start_time, cur_time))

        return state_list, [0 for _ in range(len(self.agents))]

    def step_in_sim(self, actions):
        # todo: 为了方便测试，action是单个的，但是实际上是一个list
        logging.debug(f"Current time: {self.current_time}")
        self.current_time += 1  # 假设每次step时间增加1
        # 更新交通状态
        self.update_traffic()

        state_list = [None for _ in range(len(self.agents))]
        for index, agent in enumerate(self.agents):
            if agent.state.status == AgentStatus.TRAVELLING:
                logging.debug(f"Agent {agent.id} is travelling.")
                agent.agent_keep_travelling(self.traffic_state)
            else:
                # print(actions, index)
                if actions[index] != -1:
                    agent.step(actions[index])

            content = agent.get_rf_state()
            logging.debug(f"agent {agent.id} Content: {content}")
            if content is not None:
                cur_time = self.current_time
                cur_loc, cur_act_order, current_activity_duration = content
                cur_act_start_time = cur_time - current_activity_duration
                logging.debug(
                    f"cur_loc: {cur_loc}, cur_act_order: {cur_act_order}, cur_act_start_time: {cur_act_start_time}, "
                    f"cur_time: {cur_time}")
                state_list[index] = RFState(cur_loc, cur_act_order, cur_act_start_time, cur_time)  # 直接通过索引更新状态
        reward_list = self.get_step_reward()  # 直接通过索引更新奖励
        # 如果content为None，state_list和reward_list中对应的元素保持为None，无需再次操作

        return state_list, reward_list

    def print_state(self):
        for agent in self.agents:
            logging.debug(f"Agent {agent.id} state: {agent.state}")

    def end(self):
        def check_all_agent_activity_completed():
            for agent in self.agents:
                if not agent.check_for_end_of_schedule():
                    return False
            return True

        if not check_all_agent_activity_completed():
            logging.debug("Not all agents' activities have been completed.")

        for agent in self.agents:
            # check if the agent has completed the schedule
            if agent.state.status == AgentStatus.TRAVELLING:
                agent.agent_forcefully_end_travel()
            else:
                agent.agent_end_activity()
        self.calculate_reward()
        logging.debug("End of the day. All agents' activities have been completed.")

    def calculate_utility(self, activity_name, t_perf):
        def utility_fun(umin, umax, alpha, beta, gamma, t_perf):
            return umin + (umax - umin) / np.power((1 + gamma * np.exp(beta * (alpha - t_perf))), 1 / gamma)

        # get utility
        from exp.get_graph import get_utility
        param_dict = get_utility()
        if activity_name in param_dict:
            params = param_dict[activity_name]
            return utility_fun(params['umin'], params['umax'], params['alpha'], params['beta'], params['gamma'],
                               t_perf)
        else:
            raise ValueError(f"Unknown activity: {activity_name}")

    def calculate_reward(self):

        # decode the activity
        for agent in self.agents:
            utility_score = 0
            truncated_schedule = agent.state.schedule
            # check the schedule finished
            if len(agent.state.schedule) != len(agent.state.node_stay_record):
                logging.debug(
                    f"Agent {agent.id} schedule ( {len(agent.state.schedule)})and stay record length ({len(agent.state.node_stay_record)}) not equal.")
                # truncate the longer one
                min_len = min(len(agent.state.node_record), len(agent.state.node_stay_record))
                truncated_schedule = agent.state.schedule[:min_len]
                agent.state.node_stay_record = agent.state.node_stay_record[:min_len]
            for i in range(len(truncated_schedule)):
                activity = truncated_schedule[i]
                t_perf = agent.state.node_stay_record[i]
                utility_score += self.calculate_utility(activity.name, t_perf / 10)
            logging.debug(f"Agent {agent.born_id} reward calculated.")
            self.reward_cur_epoch[agent.id] = utility_score

    def get_step_reward(self):
        rewards = [0 for _ in range(len(self.agents))]
        for index, agent in enumerate(self.agents):
            if agent.state.status == AgentStatus.ENGAGED_IN_ACTIVITY:
                cur_activity = agent.state.schedule[agent.state.current_activity_number]
                t_perf = agent.state.current_activity_duration / 10
                last_dur = round(max(t_perf - 1 / 10, 0), 2)
                logging.debug(f"Activity: {cur_activity.name}, t_perf: {t_perf}, last_dur: {last_dur}")
                rewards[index] = self.calculate_utility(cur_activity.name, t_perf) - self.calculate_utility(
                    cur_activity.name, last_dur)

        return rewards

    def epsilon_greedy_policy(self, rf_states: [RFState], epsilon):
        # todo 分布式q-learning
        actions = [-1 for _ in range(len(self.agents))]
        # print(f"State: {rf_states}")
        for index, rf_state in enumerate(rf_states):
            possible_ids = []
            if rf_state is not None:
                agent = self.agents[index]
                cur_loc, cur_act_order, cur_act_start_time, cur_time = rf_state.get_info()
                # 如果已经是最后一个activity了，那么不需要再选择动作了
                if rf_state.cur_act_order == len(agent.state.schedule) - 1:
                    logging.debug("all activities are done, no action,just stay.")
                    actions[index] = cur_loc
                possible_next_activity_nodes = agent.get_possible_actions(cur_act_order)
                # Assuming possible_next_activity_nodes is already a list of IDs
                possible_ids.extend(possible_next_activity_nodes)  # Merge lists
                possible_ids.append(cur_loc)  # Add current location as a possible stay action

                logging.debug(f"cur_state: {rf_state}")
                logging.debug(f"possible_ids: {possible_ids}")

                if np.random.rand() < epsilon:
                    action = np.random.choice(possible_ids)
                else:
                    # 否则从possible_ids选择Q值最高的动作
                    q_values_of_state = self.Q_table[(cur_loc, cur_act_order, cur_act_start_time, cur_time)]
                    logging.debug(f"q_values_of_state: {q_values_of_state}")
                    q_values_of_possible_actions = [q_values_of_state[action_id] for action_id in possible_ids]
                    logging.debug(f"q_values_of_possible_actions: {q_values_of_possible_actions}")
                    max_q_value = np.max(q_values_of_possible_actions)
                    # 如果有多个动作对应的Q值都是最高的，那么从中随机选择一个
                    max_q_actions = [action_id for action_id in possible_ids if
                                     q_values_of_state[action_id] == max_q_value]
                    action = np.random.choice(max_q_actions)
                actions[index] = action

        return actions

    def test_q(self):
        test_episodes = 10
        num_agents = len(self.agents)
        total_rewards = [0 for _ in range(num_agents)]  # Initialize total rewards for each agent

        for _ in range(test_episodes):
            states, _ = self.reset()  # Resets and returns states for all agents
            for _ in range(239):
                actions = []
                for agent_index in range(num_agents):
                    # Assume a default or no-op action for each agent, then replace with the actual action for the current agent
                    agent_states = [None] * num_agents
                    agent_states[agent_index] = states[agent_index]  # Set the state for the current agent under test
                    action = self.epsilon_greedy_policy(agent_states, 0)[
                        agent_index]  # Use epsilon greedy policy to determine action for current agent
                    actions.append(
                        action)  # Collect actions for all agents, with the actual action for the current agent

                next_states, rewards = self.step_in_sim(actions)  # Execute actions for all agents

                for agent_index in range(num_agents):
                    if rewards[agent_index] is not None:
                        total_rewards[agent_index] += rewards[agent_index]  # Update total reward for each agent

                states = next_states  # Update states for all agents

        average_rewards = [total_reward / test_episodes for total_reward in
                           total_rewards]  # Calculate average rewards for each agent
        for agent_index, avg_reward in enumerate(average_rewards):
            logging.info(f"Agent {agent_index} 测试平均奖励: {avg_reward}")

        return average_rewards  # Return average rewards for each agent

    def q_learning(self, num_episodes, alpha, gamma, epsilon):
        qt = self.Q_table
        record = []
        test_record = []
        avg_rws = []

        # 测试q_table

        # 开始训练
        for i_episode in tqdm(range(num_episodes)):
            # print(f"Episode {i_episode + 1}/{num_episodes}")

            # 重置环境，获取初始状态
            states, _ = self.reset()
            # print(f"State: {states}")

            # 当前状态下，选择一个动作
            actions = self.epsilon_greedy_policy(states, epsilon)
            # print(f"Action: {actions}")

            for t in range(239):
                logging.debug("-------------------")
                next_states, rewards = self.step_in_sim(actions)
                for index, reward in enumerate(rewards):
                    if reward is not None:
                        logging.debug(f"Agent {self.agents[index].id} reward: {reward}")
                # 执行动作，得到下一个状态和奖励
                next_actions = self.epsilon_greedy_policy(next_states, epsilon)
                # print(f"Next Action: {next_actions}")
                for index, agent in enumerate(self.agents):
                    state = states[index]
                    next_state = next_states[index]
                    if state is not None and next_state is not None:
                        content = state.get_info()
                        next_content = next_state.get_info()
                        cur_loc, cur_act_order, start_time, cur_time = content
                        logging.debug(f"state: {cur_loc}, {cur_act_order}, {start_time}, {cur_time}")
                        state_key = (cur_loc, cur_act_order, start_time, cur_time)
                        if next_content is not None:
                            next_loc, next_act_order, next_start_time, next_time = next_content
                            logging.debug(f"next state: {next_loc}, {next_act_order}, {next_start_time}, {next_time}")
                            next_state_key = (next_loc, next_act_order, next_start_time, next_time)
                            # 更新Q表
                            logging.debug("update Q table...")
                            action = actions[index]
                            reward = rewards[index]
                            next_action = next_actions[index]
                            # print(
                            #     f"state_key: {state_key}, action: {action}, reward: {reward}, next_state_key: {next_state_key}, next_action: {next_action}")
                            old_q_value = qt[state_key][action]
                            new_q_value = qt[next_state_key][next_action]
                            qt[state_key][action] += alpha * (reward + gamma * new_q_value - old_q_value)
                            logging.info("update state: {}, next_state: {}".format(state, next_state))
                            logging.info("Q table updated: old_q_value: {}, new_q_value: {}".format(old_q_value,
                                                                                                    qt[state_key][
                                                                                                        action]))
                    else:
                        logging.debug("no update state: {}, next_state: {}".format(state, next_state))

                # 更新状态和动作
                states = next_states
                actions = next_actions

            self.calculate_reward()
            record.append(self.reward_cur_epoch[self.agents[0].id])
            logging.info("record: %s", record)
            # 每隔1000轮输出一次训练进度
            if (i_episode + 1) % 100 == 0 or i_episode == 0:
                # tqdm.write("\rEpisode {}/{}".format(i_episode + 1, num_episodes), end="")
                test_rw = self.test_q()
                avg_rw = sum(test_rw) / len(test_rw)
                tqdm.write(f"\rEpisode {i_episode + 1}/{num_episodes}, Test Reward AVG: {avg_rw}")
                # sys.stdout.flush()
                test_record.append(test_rw)
                avg_rws.append(avg_rw)

            # early stop
            if len(avg_rws) >= 1000:  # 确保有足够的数据进行检查
                last_100_test_rewards = np.array(avg_rws[-100:])
                std_dev = np.std(last_100_test_rewards)
                logging.debug(f"标准差为: {std_dev}")

                # 设定一个阈值作为判断标准，这个阈值可以根据具体情况进行调整
                threshold = 0.5  # 例如，如果标准差小于0.5，则认为收敛
                if std_dev < threshold:
                    print(f"测试奖励标准差小于{threshold}，认为已经收敛。")
                    break

        return qt, record, test_record

    def update_traffic(self):
        # 获取每个agent_state的当前segment
        for agent in self.agents:
            if agent.state.status == AgentStatus.TRAVELLING:
                cur_segment = agent.state.current_travel.get_current_road_segment()
                self.traffic_state.road_segments[cur_segment]['cur_traffic'] += 1
                logging.info(
                    f"Segment {cur_segment} traffic updated. + 1  = {self.traffic_state.road_segments[cur_segment]['cur_traffic']}")

    def dqn(self, num_episodes, gamma, epsilon):
        def state2tensor(state: RFState):
            cur_loc, cur_act_order, cur_act_start_time, cur_time = state.get_info()
            # Convert to tensors
            cur_loc = torch.FloatTensor([cur_loc])
            cur_act_order = torch.FloatTensor([cur_act_order])
            cur_act_start_time = torch.FloatTensor([cur_act_start_time])
            cur_time = torch.FloatTensor([cur_time])
            # Concatenate tensors along a new dimension
            state_tensor = torch.cat([cur_loc, cur_act_order, cur_act_start_time, cur_time], dim=0)
            return state_tensor

        state_size = 4
        action_size = 24
        print(f"State size: {state_size}, Action size: {action_size}")
        dqn_network = DQNNetwork(state_size, action_size)
        optimizer = optim.Adam(dqn_network.parameters(), lr=0.001)
        replay_memory = deque(maxlen=10000)
        batch_size = 8

        for i_episode in tqdm(range(num_episodes)):
            print(f"Episode {i_episode + 1}/{num_episodes}")
            states, _ = self.reset()
            total_reward = 0
            state_tensor = state2tensor(states[0])
            travel_happened = False
            last_real_action = None
            last_activity_flag = False
            for t in range(20):
                print(f"######  Step {t + 1}/240 ######")
                print(self.agents[0].state)
                cur_act_order = self.agents[0].state.current_activity_number
                cur_loc = self.agents[0].state.current_node.id
                possible_next_activity_nodes = self.agents[0].get_possible_actions(cur_act_order)
                possible_next_activity_nodes.append(cur_loc)
                print(f"Possible next activity nodes: {possible_next_activity_nodes}")

                if np.random.rand() < epsilon:
                    # 获取DQN网络预测的动作值
                    action_values = dqn_network(state_tensor)
                    print(f"Action values: {action_values}")

                    # 创建一个与action_values形状相同的掩码，初始值为非常小的数
                    mask = torch.full(action_values.shape, -float('inf'))

                    # 将可能的动作位置设置为0，使这些位置的动作值不受影响
                    mask[torch.tensor(possible_next_activity_nodes) - 1] = 0

                    # 应用掩码到动作值
                    adjusted_action_values = action_values + mask
                    print(f"Action values adjusted: {adjusted_action_values}")

                    # 选择调整后的最大值对应的动作
                    max_index = torch.argmax(adjusted_action_values).item()
                    print(f"Max index: {max_index}")
                    action = max_index + 1
                    print(f"Action selected: {action}")

                else:
                    action = np.random.choice(possible_next_activity_nodes)

                # action_list = [action] + [3 for _ in range(len(self.agents) - 1)]
                if last_activity_flag and self.agents[0].state.status == AgentStatus.ENGAGED_IN_ACTIVITY:
                    print("Agent's last activity, no action, just stay.")
                    action = self.agents[0].state.current_node.id
                print(f"Action generated : {action}")

                before_agent_status = self.agents[0].state.status
                next_state_list, reward_list = self.step_in_sim([action])
                after_agent_status = self.agents[0].state.status

                if before_agent_status == AgentStatus.TRAVELLING and after_agent_status == AgentStatus.ENGAGED_IN_ACTIVITY:
                    print("Agent arrived at the destination.")
                print(f"step_in_sim Next state: {next_state_list[0]}")
                print(f"step_in_sim Reward: {reward_list[0]}")

                if reward_list[0] is not None and next_state_list[0] is not None:

                    # 首先判断如果是agent最后一个activity，那么动作只能是当前位置
                    if last_activity_flag and self.agents[0].state.status == AgentStatus.ENGAGED_IN_ACTIVITY:
                        print("Agent's last activity, no action, just stay.")
                        real_action = last_real_action
                    else:
                        # 如果旅行发生，使用当前动作或者旅程前的最后一个动作
                        real_action = last_real_action if travel_happened else action
                        travel_happened = False  # 重置旅行发生标志
                    if self.agents[0].state.current_activity_number == len(self.agents[0].state.schedule) - 1:
                        last_activity_flag = True
                    next_state_tensor = state2tensor(next_state_list[0])
                    total_reward += reward_list[0]
                    # 存储转换
                    replay_memory.append((state_tensor, real_action, reward_list[0], next_state_tensor))
                    print(f">> state_tensor: {state_tensor}")
                    print(f">> action: {real_action}")
                    print(f">> reward: {reward_list[0]}")
                    print(f">> next_state_tensor: {next_state_tensor}")
                    state_tensor = next_state_tensor

                    # 学习
                    if len(replay_memory) > batch_size:
                        minibatch = random.sample(replay_memory, batch_size)
                        states, actions, rewards, next_states = zip(*minibatch)
                        # print("replay_memory ")

                        # 使用torch.stack来转换states和next_states
                        states = torch.stack(states).float()
                        next_states = torch.stack(next_states).float()

                        # 直接转换actions和rewards为张量
                        actions = torch.tensor(actions).long()  # 动作通常是整数，所以使用torch.LongTensor
                        rewards = torch.tensor(rewards).float()  # 奖励可以是浮点数

                        print(states)
                        print(actions)
                        print(rewards)
                        print(next_states)

                        Q_values = dqn_network(states)  # 计算当前状态下所有可能动作的Q值
                        print(Q_values.shape)
                        adjusted_actions = actions - 1  # 将动作编号从1开始转换为0开始
                        Q_expected = Q_values.gather(1, adjusted_actions.unsqueeze(-1)).squeeze(-1)

                        print(Q_expected)
                        next_Q_values = dqn_network(next_states).detach()  # 计算下一个状态下所有可能动作的Q值，并阻止梯度传播
                        Q_targets_next = next_Q_values.max(1)[0]  # 选择最大的Q值作为下一个状态的目标Q值

                        Q_targets = rewards + (gamma * Q_targets_next)

                        loss = F.mse_loss(Q_expected, Q_targets)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        travel_happened = False


                else:
                    # 发生了旅行，要记录旅行前最后一个action
                    if not travel_happened:
                        last_real_action = action  # 记录可能导致旅行的动作
                    travel_happened = True

                # 减少 epsilon
                epsilon = max(epsilon * 0.99, 0.01)  # 逐步减少探索率

        return dqn_network
