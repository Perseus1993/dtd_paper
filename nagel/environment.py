from nagel.reward import generate_reward_matrix
import numpy as np
travel_time = [1, 1, 1, 1]


class Env:
    def __init__(self):
        self.reward_leisure = generate_reward_matrix("leisure")
        self.reward_home = generate_reward_matrix("home")
        self.reward_shop = generate_reward_matrix("shopping")
        self.reward_work = generate_reward_matrix("work")

    def get_reward(self, activity_id, start_time, dur):
        cur_act = {
            0: self.reward_home,
            1: self.reward_work,
            2: self.reward_shop,
            3: self.reward_leisure,
            4: self.reward_home,
        }
        re_table = cur_act.get(activity_id)
        try:
            res = re_table[dur][start_time]
        except:
            print(dur, start_time)
            print(re_table)
            print(activity_id)
        return re_table[dur][start_time]

    def step(self, state, action):
        reward = 0
        activity_no = 0 if state[0] == 4 else state[0]
        start_time = state[1]
        dur = state[2]
        if action == 0:
            # 超24小时了
            if start_time + dur + 1 >= 24:
                return True, reward, None

            # dur 超过 12 * 4 自动跳转下一个行为
            if dur + 1 >= 12:
                reward -= travel_time[activity_no]
                # 加上旅程超24小时了
                if start_time + dur + travel_time[activity_no] >= 24:
                    return True, reward, None
                # 正常转换
                start_time = start_time + dur + travel_time[activity_no]
                activity_no += 1
                dur = 0
            else:
                dur += 1

            reward += self.get_reward(activity_no, start_time, dur)
        else:
            # travel cost
            reward -= travel_time[activity_no]
            start_time = start_time + dur + travel_time[activity_no]
            # 超24小时了
            if start_time >= 24:
                return True, reward, None
            if state[0] == 3:
                # go home 并且计算剩下的reward
                i = 0
                while start_time + i < 24 and i < 12:
                    reward += self.get_reward(0, start_time, i)
                    i += 1
                return True, reward, None
            else:
                activity_no += 1
                if activity_no == 4:
                    activity_no = 0
                dur = 0
                reward += self.get_reward(activity_no, start_time, dur)
        return False, reward, (activity_no, start_time, dur)

    def reset(self):
        return np.random.choice(1), np.random.randint(0, 1), np.random.randint(0, 1)
