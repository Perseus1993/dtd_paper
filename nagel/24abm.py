import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import *

import torch  # 导入torch
import torch.nn as nn  # 导入torch.nn
import torch.nn.functional as F  # 导入torch.nn.functional
import gym  # 导入gym
from tqdm.tk import tqdm

from nagel.environment import Env


def train_classical_q(env, epo):
    episodes = epo
    runs = 1
    smooth_rewards = np.zeros(episodes)
    rewards_q = np.zeros(episodes)
    test_score = []
    for r in range(runs):
        print(r)
        q_table_q = np.ones((4, 24, 12, 2)) * 0.1
        with open('data.txt', 'w') as f:
            for ep in tqdm(range(episodes)):
                # old_q_table = q_table_q.copy()
                f.writelines("------------------------------------" + str(ep) + '\n')
                rewards_q[ep] += q_learning(f, env, q_table_q, 0.2, 0.1)
                # diff = np.sum(np.abs(old_q_table - q_table_q))
                # print("diff = ",diff)
                if ep == episodes - 1:
                    test_score.append(test(q_table_q, env, True, f))
                else:
                    test_score.append(test(q_table_q, env, True, f))
                smooth_rewards[ep] += test_score[-1]
        f.close()
        if r == runs - 1:
            # print(q_table_q)
            # plt.plot(test_score)
            smooth_rewards /= runs
            plt.plot(scipy.signal.savgol_filter(smooth_rewards, 20, 3), label="smooth")
            plt.plot(smooth_rewards, label="non_smooth", alpha=0.2)
            plt.legend()
            plt.show()
            return q_table_q


def test(q_table, env, need_print, f):
    s = (0, 0, 0)
    score = 0
    while True:
        a = 0 if s[0] == 4 else s[0]
        best_act = find_action_from_q_table(q_table, (a, s[1], s[2]))
        is_finish, reward, new_s = env.step_in_sim(s, best_act)
        if need_print:
            np.argmax(q_table[a][s[1]][s[2]])
            str = "state {} qt {} best_action {} reward = {}".format(s, q_table[a][s[1]][s[2]], best_act, reward)
            f.writelines(str + '\n')
        score += reward
        if is_finish:
            break
        s = new_s
    return score


def q_learning(f, env: Env, q_table, alpha, epsilon):
    old_q_table = None
    state = env.reset()    # state = (1, np.random.randint(0, 24), np.random.randint(0, 12))
    # print("init_state : ", state)
    while True:

        if np.random.binomial(1, epsilon) == 1:
            action = np.random.choice(2)
        else:
            action = find_action_from_q_table(q_table, state)

        is_finish, reward, next_state = env.step(state, action)
        # print("action = ", action, "next state = ", next_state)
        if is_finish:
            break

        act = 0 if state[0] == 4 else state[0]
        next_activity = 0 if next_state[0] == 4 else next_state[0]
        # if next_state[0] == 1 and next_state[2] in (9, 10):
        str1 = "old {} {} {} [{}], value {} ".format(act, state[1], state[2], action,
                                                     q_table[act][state[1]][state[2]][action])
        str2 = "=> reward {} , next_state {} {} {} ,next_value {}".format(reward, next_activity, next_state[1],
                                                                          next_state[2],
                                                                          q_table[next_activity][next_state[1]][
                                                                              next_state[2]])
        # str1 = "old ", act, state[1], state[2], [action], "value ", q_table[act][state[1]][state[2]][action],
        #           " => ", "re ", reward, "next state", [next_activity], [next_state[1]], [next_state[2]],
        #           " next value ", q_table[next_activity][next_state[1]][next_state[2]]
        f.writelines(str1 + '\n')
        f.writelines(str2 + '\n')
        old_q = q_table[act][state[1]][state[2]][action]
        q_table[act][state[1]][state[2]][action] += alpha * (
                reward + 0.99 * np.max(q_table[next_activity, next_state[1], next_state[2], :]) - old_q)
        # - q_table[act][state[1]][state[2]][action])
        # if next_state[0] == 1 and next_state[2] in (9, 10):
        str3 = " new value {}".format(q_table[act][state[1]][state[2]][action])
        f.writelines(str3 + '\n')
        f.writelines("***** " + '\n')
        state = next_state
    f.writelines(str(q_table[0][0]))
    return reward


def find_action_from_q_table(q_table, state):
    activity = 0 if state[0] == 4 else state[0]
    values = q_table[activity][state[1]][state[2]]
    return np.random.choice(np.where(values == np.max(values))[0])


if __name__ == '__main__':
    env = Env()
    travel_time = [1, 1, 1, 1]
    # 构建state矩阵
    # 4 activity , 2 action state_space(activity,start_time,dur)
    state = np.zeros((1, 3))
    action = [0, 1]

    train_classical_q(env, 200)
