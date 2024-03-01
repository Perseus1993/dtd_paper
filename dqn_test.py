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
    env = Env(G, nodes_dict, max_schedule_num, mode='single', num_agents=1)

    # 将字符串的活动列表转换为Activity枚举
    schedule = [Activity[activity] for activity in args.schedule]

    env.generate_agent(schedule, 1, 1)

    env.dqn(num_episodes=100, gamma=0.9, epsilon=0.2)
