import logging

from exp.Node import Node
from exp.get_graph import get_shortest_distance, get_shortest_path
import random
from enum import Enum, auto
import networkx as nx

from exp.traffic import Traffic_State


class Activity(Enum):
    HOME = auto()
    EDU = auto()
    SHOPPING = auto()
    WORK = auto()
    LEISURE = auto()


class AgentStatus(Enum):
    TRAVELLING = auto()
    ENGAGED_IN_ACTIVITY = auto()
    END_OF_SCHEDULE = auto()


class Action(Enum):
    STAY = auto()
    MOVE_TO_NEXT = auto()
    END = auto()


class Journey:
    def __init__(self, trace: [int], G: nx.Graph):
        self.trace = trace
        self.G = G
        # 检查合法性
        if not self._is_trace_valid(trace):
            raise ValueError("Provided trace is not valid in the given graph.")
        self.source = trace[0]
        self.destination = trace[-1]
        self.progress = 0.0
        self.road_segment, self.road_segment_dist = self.get_road_segment()
        self.total_distance = self._get_path_distance()

    def _is_trace_valid(self, trace):
        """检查提供的trace是否在图G中是合法的路径"""
        for i in range(len(trace) - 1):
            if trace[i + 1] not in self.G[trace[i]]:
                return False
        return True

    def _get_path_distance(self):
        distance = 0
        for segment in self.road_segment:
            # 如果距离计算仅依赖于边的权重
            distance += nx.shortest_path_length(self.G, source=segment[0], target=segment[1], weight='length')

            # 如果需要根据更复杂的逻辑计算距离
            # edge_data = self.G[segment[0]][segment[1]]
            # distance += some_complex_function(edge_data)
        return distance

    def step(self, speed):
        self.progress += speed
        if self.progress >= self.total_distance:
            self.progress = self.total_distance
            return True
        else:
            return False

    def get_road_segment(self):
        # 构造路径，格式为：[(source, destination), (destination, destination), ...]
        road_segment = [(self.trace[i], self.trace[i + 1]) for i in range(len(self.trace) - 1)]
        # 计算每一段路径的长度
        road_segment_dist = []
        for i in range(len(road_segment)):
            road_segment_dist.append(get_shortest_distance(self.G, road_segment[i][0], road_segment[i][1]))
        return road_segment, road_segment_dist

    def get_current_road_segment(self):
        # 根据progress返回当前的路径段

        distance = 0
        for i in range(len(self.road_segment)):
            distance += get_shortest_distance(self.G, self.road_segment[i][0], self.road_segment[i][1])
            if distance >= self.progress:
                return self.road_segment[i]


class AgentState:
    def __init__(self, current_node: Node, network: nx.Graph, activity: Activity, schedule: [Activity], born_id,
                 nodes_dict: dict,
                 status: AgentStatus = AgentStatus.ENGAGED_IN_ACTIVITY):
        self.born_id = born_id
        self.nodes_dict = nodes_dict

        self.network = network
        self.activity = activity
        self.status = AgentStatus.ENGAGED_IN_ACTIVITY
        self.schedule = schedule
        self.current_activity_number = -1
        self.current_activity_duration = 0

        self.current_node: Node = current_node  # 智能体当前停留的节点

        self.destination_node: Node = None  # 智能体下一个要到达的节点

        self.node_record = []  # 记录智能体在每个节点
        self.node_stay_record = []  # 记录智能体在每个节点停留的时间
        self.travel_record = []  # 记录智能体的通勤记录

        self.current_travel = None

    def stay_at_current_node(self):
        """保持在当前节点。"""
        self.destination_node = None
        self.current_activity_duration += 1

    def stay_travelling(self, speed):
        """保持在当前节点。"""
        finsh = self.current_travel.step(speed)
        if finsh:
            self.travel_record.append(self.current_travel.total_distance)
            self.current_node = self.nodes_dict[self.current_travel.destination]
            self.start_activity()
            self.current_travel = None

    def start_travel(self, d: Node):
        """启动前往另一个节点的通勤。"""
        # 处理上一个活动的停留时间
        self.node_stay_record.append(self.current_activity_duration)
        self.current_activity_duration = 0  # 防止agent在旅行中时，仿真终止
        # 装填当前通勤信息
        path = get_shortest_path(self.network, self.current_node.id, d.id)
        self.current_travel = Journey(path, self.network)
        self.status = AgentStatus.TRAVELLING
        logging.debug(f"Agent is travelling from {self.current_node} to {d}.")
        logging.debug(f"tot_dist =  {self.current_travel.total_distance}")
        logging.debug(f"road_segment_dist = {self.current_travel.road_segment_dist}")

    def start_activity(self):
        """开始新的活动。"""
        self.destination_node = None
        self.current_activity_number += 1
        activity = self.schedule[self.current_activity_number]
        self.activity = activity
        logging.debug("[start_activity]self.current_activity_number = %s", self.current_activity_number)
        logging.debug("[start_activity]self.current_node = %s", self.current_node)
        self.status = AgentStatus.ENGAGED_IN_ACTIVITY
        self.current_activity_duration = 0
        self.node_record.append(self.current_node)

    def end_activity(self):
        """结束当前活动。"""
        self.status = AgentStatus.ENGAGED_IN_ACTIVITY
        self.current_activity_duration = 0
        self.node_record.append(self.current_node)
        self.node_stay_record.append(self.current_activity_duration)

    def reset(self):
        self.status = AgentStatus.ENGAGED_IN_ACTIVITY
        self.current_activity_number = -1
        self.current_activity_duration = 0
        # 回到出生点
        self.current_node = self.nodes_dict[self.born_id]
        self.destination_node = None
        self.node_record = []
        self.node_stay_record = []
        self.travel_record = []
        self.current_travel = None

    def __repr__(self):
        # 假设 current_node 是一个 Node 实例，我们可以直接访问它的 id
        location_str = f"Node(id={self.current_node.id}, type={self.current_node.node_type.name})"

        # 假设 activity 包含一个对应 Node 的引用或 ID
        # 这里我们需要根据你的 Activity 设计进行调整
        activity_str = self.activity.name if self.activity else "None"

        status_str = self.status.name if self.status else "None"

        # 修改这里来包含与每个活动相关的 Node id
        schedule_str = ", ".join([activity.name for activity in self.schedule])

        current_activity_number_str = str(self.current_activity_number)

        node_record_str = ", ".join([str(node.id) for node in self.node_record])

        node_stay_record_str = ", ".join([str(node) for node in self.node_stay_record])

        travel_record_str = ", ".join([str(node) for node in self.travel_record])

        return (f"Location: {location_str}, "
                f"Activity: {activity_str}, "
                f"Status: {status_str}, "
                f"Schedule: [{schedule_str}], "
                f"Node Record: [{node_record_str}], "
                f"Node Stay Record: [{node_stay_record_str}], "
                f"Travel Record: [{travel_record_str}], "
                f"Current Activity Number: {current_activity_number_str}")

    def end_of_schedule(self):
        self.status = AgentStatus.END_OF_SCHEDULE
        self.node_stay_record.append(self.current_activity_duration)

    def kill_travel(self):
        assert self.status == AgentStatus.TRAVELLING
        self.status = AgentStatus.ENGAGED_IN_ACTIVITY
        self.travel_record.append(self.current_travel.progress)

    def start_travel_to_id(self, action):
        # 处理上一个活动的停留时间
        self.node_stay_record.append(self.current_activity_duration)
        self.current_activity_duration = 0  # 防止agent在旅行中时，仿真终止
        # 装填当前通勤信息
        path = get_shortest_path(self.network, self.current_node.id, action)
        self.current_travel = Journey(path, self.network)
        self.status = AgentStatus.TRAVELLING
        logging.debug(f"Agent is travelling from {self.current_node} to {action}.")
        logging.debug(f"tot_dist =  {self.current_travel.total_distance}")
        logging.debug(f"road_segment_dist = {self.current_travel.road_segment_dist}")


class Agent:
    next_id = 0

    def __init__(self, schedule: [Activity], born_id, G: nx.Graph, nodes_dict: dict):
        self.G = G
        self.born_id = born_id
        first_activity = schedule[0]
        born_node = nodes_dict[born_id]
        self.state = AgentState(born_node, G, first_activity, schedule, self.born_id, nodes_dict)
        self.nodes_dict = nodes_dict
        self.available_actions = []
        self.state.start_activity()

        # 自增id
        self.id = Agent.next_id
        Agent.next_id += 1

    def choose_action_prob(self):
        # 定义概率分布
        actions = [Action.STAY, Action.MOVE_TO_NEXT]  # 可用行动
        probabilities = [0.8, 0.2]  # 对应行动的概率，80% STAY，20% MOVE_TO_NEXT

        # 根据概率分布选择行动
        chosen_action = random.choices(actions, weights=probabilities, k=1)[0]

        return chosen_action

    def choose_action(self, available_actions: [Action], G):
        # 检查是否在旅行中
        if self.state.status == AgentStatus.TRAVELLING:
            logging.debug("Agent is travelling. ignore the action.")
            return
        # 首先检查是否有行动可用
        if not available_actions:
            logging.debug("No available actions.")
            return
        # 检查schedule是否为空
        if not self.state.schedule:
            logging.debug("Agent's schedule is empty.")
            return

        # 检查schedule是否完成
        if self.state.current_activity_number >= len(self.state.schedule):
            logging.debug("Agent has completed the schedule.")
            return
        # 随机选择一个行动
        action = random.choice(available_actions)
        return action

    def step(self, action):
        # 更新状态
        if self.state.status == AgentStatus.TRAVELLING:
            self.agent_keep_travelling()
            logging.debug("Agent is travelling.")
        else:
            # action是Agent的下一个节点
            if self.state.current_node.id != action:
                # Action.MOVE_TO_NEXT
                if self.state.current_activity_number < len(self.state.schedule) - 1:
                    self.agent_start_travel_to_id(self.state.current_node, action)
                    # self.agent_start_travel(self.state.current_node,
                    #                         self.state.schedule[self.state.current_activity_number + 1])
                else:
                    logging.debug("Agent has completed the schedule. but you still want to move to next node.")
            else:
                self.agent_stay_at_current_node()
                logging.debug("Agent is staying at current node.")

    def agent_start_travel(self, o, d_activity: Activity):
        logging.debug(f"Agent is travelling from {o} to activity {d_activity}.")
        d = self.get_destination_node(d_activity)[0]
        self.state.start_travel(d)

    def agent_stay_at_current_node(self):
        logging.debug(f"Agent is staying at {self.state.current_node}.")
        self.state.stay_at_current_node()

    def agent_keep_travelling(self, traffic_state: Traffic_State):
        total_distance = self.state.current_travel.total_distance
        speed = traffic_state.get_speed(self.state.current_travel.get_current_road_segment()[0],
                                        self.state.current_travel.get_current_road_segment()[1])
        print(f"Agent is travelling from {self.state.current_node} to {self.state.current_travel.destination}. \
                      speed = {speed}")
        self.state.stay_travelling(speed)
        if self.state.status != AgentStatus.ENGAGED_IN_ACTIVITY:
            # 说明没结束
            progress = self.state.current_travel.progress
            logging.debug(f"Agent is keep travelling. Progress: {progress}/{total_distance}")

    def agent_forcefully_end_travel(self):
        self.state.kill_travel()

    def agent_end_activity(self):
        logging.debug(f"Agent has completed activity {self.state.activity}.")
        self.state.node_stay_record.append(self.state.current_activity_duration)

    def agent_end_of_schedule(self):
        logging.debug(f"Agent has completed the schedule.")

    def agent_arrive_at_node(self):
        logging.debug(f"Agent has arrived at {self.state.current_node}.")

    # def update_state(self, update):
    #     if update == "stay_at_node":
    #         self.agent_stay_at_current_node()
    #     elif update == "travelling":
    #         self.agent_keep_travelling()
    #     elif update == "arrive_at_node":
    #         self.agent_arrive_at_node()
    #     else:
    #         # start travel
    #         o = self.state.current_node
    #         d_act = self.state.schedule[self.state.current_activity_number + 1]
    #         print("下一个活动 d_act = ", d_act)
    #         d = self.get_destination_node(d_act)
    #         self.agent_start_travel(o, d)

    def add_to_schedule(self, activity: Activity):
        self.state.schedule.append(activity)

    def get_state(self):
        return self.state

    def check_for_end_of_schedule(self):
        if self.state.current_activity_number == len(self.state.schedule) - 1:
            logging.debug("Agent has completed the schedule.")
            return True
        else:
            return False

    def get_destination_node(self, d_act) -> [Node]:
        # 根据node_types字典，选择下一个节点
        logging.debug(d_act)
        node_ids = [n for n, d in self.G.nodes(data=True) if d['O/D'] == d_act.name.lower()]
        logging.debug("available node ids: %s", node_ids)
        return [self.nodes_dict[n] for n in node_ids]

    def reset(self):
        self.state.reset()
        self.state.start_activity()
        logging.debug("Agent reset to born node.")

    # rest is for reinforcement learning
    def get_rf_state(self):
        # 只能在活动中选择行动
        if self.state.status == AgentStatus.ENGAGED_IN_ACTIVITY:
            logging.debug("Agent is in activity.")
            # need cur_loc, cur_act_order, cur_act_start_time, cur_time
            cur_loc = self.state.current_node.id
            cur_act_order = self.state.current_activity_number
            current_activity_duration = self.state.current_activity_duration
            return cur_loc, cur_act_order, current_activity_duration
        else:
            logging.debug("Agent is not in activity.")
            return None

    def get_possible_actions(self, cur_loc, cur_act_order, cur_act_start_time, cur_time):
        # 下一个活动
        next_act_order = cur_act_order + 1
        if next_act_order >= len(self.state.schedule):
            return []
        # print("next_act_order = ", next_act_order)
        # print("self.state.schedule = ", self.state.schedule)
        next_act = self.state.schedule[next_act_order]
        # 下一个节点
        next_nodes = self.get_destination_node(next_act)
        # print("next_nodes = ", next_nodes)
        return [n.id for n in next_nodes]

    def agent_start_travel_to_id(self, current_node, action):
        """启动前往另一个节点的通勤。"""
        logging.debug(f"Agent is travelling from {current_node} to activity {action}.")
        self.state.start_travel_to_id(action)
