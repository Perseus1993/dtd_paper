import argparse
import os
import pickle

import numpy as np
from exp.get_graph import get_graph
import logging
import importlib
from exp import agent
import matplotlib.pyplot as plt
from exp.Env import Env
from exp.agent import Agent, Activity, Action, Journey

if __name__ == '__main__':
    print(os.path.dirname(__file__))
    parser = argparse.ArgumentParser(description="Run Q-learning with custom parameters.")
    # 为每个参数添加默认值
    parser.add_argument("--schedule", nargs='+', type=str, default=["HOME", "WORK", "LEISURE"],
                        help="List of activities (e.g., HOME WORK LEISURE)")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes for Q-learning")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Epsilon value for epsilon-greedy policy")
    parser.add_argument("--home", type=int, default=1, help="Home node ID")
    parser.add_argument("--method", type=str, default="q_learning", help="Method to use for training")

    args = parser.parse_args()

    # 配置日志级别和格式
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

    G, nodes_dict = get_graph()
    max_schedule_num = len(args.schedule)
    env = Env(G, nodes_dict, max_schedule_num,mode='nosingle',num_agents=1)

    # 将字符串的活动列表转换为Activity枚举
    schedule = [Activity[activity] for activity in args.schedule]

    env.generate_agent(schedule, 1, 1)
    env.generate_agent(schedule, 1, 2)
    env.generate_agent(schedule, 1, 7)

    # 使用命令行参数
    epsilon = args.epsilon
    alpha = 0.1
    gamma = 0.2
    num_episodes = args.num_episodes
    home = args.home

    q_table, record, test_record = env.q_learning(num_episodes, alpha, gamma, epsilon)

    # 调整Matplotlib的日志级别以减少调试信息的输出
    matplotlib_logger = logging.getLogger('matplotlib')
    matplotlib_logger.setLevel(logging.WARNING)

    # 绘图部分
    test_interval = 100  # 假设每10个回合记录一次test_record
    num_agents = len(test_record[0])  # Assuming all test records have the same number of agents

    # Prepare the x-axis for plotting
    test_x = [i * test_interval for i in range(1, len(test_record) + 1)]

    # Plot training rewards
    plt.plot(record, label='Training Reward')

    # Plot test rewards for each agent
    def moving_average(data, window_size):
        """计算数组的滑动平均值"""
        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)


    # 假设 test_x 和 test_record 已经定义
    # num_agents = 3  # 示例代理数量
    window_size = 10  # 滑动窗口大小，可以根据需要调整

    for agent_idx in range(num_agents):
        agent_rewards = [test[agent_idx] for test in test_record]  # 提取该代理在所有测试中的奖励
        smooth_rewards = moving_average(agent_rewards, window_size)  # 计算滑动平均值以平滑曲线

        # 为了对齐数据，我们需要调整 x 坐标
        smooth_x = test_x[window_size - 1:]

        plt.plot(smooth_x, smooth_rewards, label=f'Agent {agent_idx + 1} Test Reward')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    # 使用format方法创建文件名和文件夹名
    output_folder = 'output/{}_{}'.format(args.schedule, args.home)

    # 创建文件夹，如果存在就清空
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # 使用makedirs确保多级目录能被创建
    else:
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            if os.path.isfile(file_path):  # 确保是文件而非目录
                os.remove(file_path)

    # 使用os.path.join构建文件路径以确保跨平台兼容性
    plt.savefig(os.path.join(output_folder, 'reward_{}_{}.png'.format(args.schedule, args.home)))
    plt.close()

    # 保存q_table
    with open(os.path.join(output_folder, 'q_table_{}_{}.pkl'.format(args.schedule, args.home)), 'wb') as f:
        pickle.dump(q_table, f)

    # 保存agent的record
    with open(os.path.join(output_folder, 'agent_record_{}_{}.pkl'.format(args.schedule, args.home)), 'wb') as f:
        pickle.dump(record, f)

    # 保存agent的test_record
    with open(os.path.join(output_folder, 'agent_test_record_{}_{}.pkl'.format(args.schedule, args.home)), 'wb') as f:
        pickle.dump(test_record, f)
