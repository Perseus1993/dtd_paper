import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from exp.Node import Node, NodeType


def get_graph(plot=False):
    file_path = r'suffix/data/'
    net = pd.read_csv(file_path + 'SiouxFalls_net.tntp', skiprows=8, sep='\t').drop(['~', ';'], axis=1)
    net['edge'] = net.index + 1
    flow = pd.read_csv(file_path + 'SiouxFalls_flow.tntp', sep='\t').drop(['From ', 'To '], axis=1)
    flow.rename(columns={"Volume ": "flow", "Cost ": "cost"}, inplace=True)
    node_coord = pd.read_csv(file_path + 'SiouxFalls_node.tntp', sep='\t').drop([';'],
                                                                                axis=1)  # Actual Sioux Falls coordinate
    node_xy = pd.read_csv(file_path + 'SiouxFalls_node_xy.tntp', sep='\t')  # X,Y position for good visualization
    sioux_falls_df = pd.concat([net, flow], axis=1)
    G = nx.from_pandas_edgelist(sioux_falls_df, 'init_node', 'term_node',
                                ['capacity', 'length', 'free_flow_time', 'b', 'power', 'speed', 'toll', 'link_type',
                                 'edge', 'flow', 'cost'], create_using=nx.MultiDiGraph())
    # 设置节点位置
    pos_xy = dict([(i, (a, b)) for i, a, b in zip(node_xy.Node, node_xy.X, node_xy.Y)])
    # 节点类型和颜色映射
    node_types = {
        'home': {'nodes': [1, 2, 7, 12, 13, 18, 20], 'color': 'green'},
        'edu': {'nodes': [3, 17, 23], 'color': 'orange'},
        'shopping': {'nodes': [11, 15, 16], 'color': 'red'},
        'work': {'nodes': [10], 'color': 'black'},
        'leisure': {'nodes': [4, 5, 6, 8, 9, 14, 19, 21, 22, 24], 'color': 'purple'}
    }

    nodes_dict = {}

    # 分配节点属性
    for node_type_str, info in node_types.items():  # 假设node_types的键是字符串
        for node in info['nodes']:
            G.nodes[node]['O/D'] = node_type_str
            G.nodes[node]['color'] = info['color']
            # print(f"Node {node} is a {node_type_str} node.")

            # 直接从字符串创建NodeType枚举实例，不改变node_type变量
            node_type_enum = NodeType[node_type_str.upper()]
            # print(f"Node type: {node_type_enum}")

            # 使用枚举实例创建Node实例
            node_instance = Node(node, node_type_enum)
            nodes_dict[node] = node_instance

    if plot:
        # 绘制网络
        plt.figure(figsize=(12, 12))
        # 使用节点的'O/D'属性直接生成颜色列表
        node_colors = [G.nodes[n]['color'] for n in G.nodes()]
        nx.draw_networkx(G, pos=pos_xy, with_labels=True, node_color=node_colors, arrows=True, arrowsize=20,
                         node_size=800,
                         font_color='white', font_size=14)

        # 创建图例
        color_node_type = {info['color']: node_type.capitalize() for node_type, info in node_types.items()}
        for color, node_type in color_node_type.items():
            plt.scatter([], [], c=color, label=node_type, s=200)
        plt.legend(loc='upper right', fontsize=12)

        # 绘制边的长度标签
        edge_labels = {(u, v): d['length'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos=pos_xy, edge_labels=edge_labels, font_color='red')

        plt.title('Sioux Falls Network', fontsize=20)
        plt.axis('off')
        plt.show()
    return G, nodes_dict


def utility_fun(umin, umax, alpha, beta, gamma, t_perf):
    return umin + (umax - umin) / np.power((1 + gamma * np.exp(beta * (alpha - t_perf))), 1 / gamma)


def get_utility(plot=False):
    param_dict = {}
    param_lines = open('ex/config/parameter.ini').readlines()
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))

    for line in param_lines:
        words = line.split(" ")
        activity_name, umin, umax, alpha, beta, gamma = words[0], int(words[1]), int(words[2]), int(words[3]), float(
            words[4]), int(words[5])

        if plot:
            t_hours = np.arange(0, 24, 0.1)
            utilities = [utility_fun(umin, umax, alpha, beta, gamma, i) for i in t_hours]
            plt.plot(t_hours, utilities, label=activity_name)

        parameters = {
            'umin': umin,
            'umax': umax,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma
        }
        param_dict[activity_name] = parameters

    if plot:
        plt.xlabel('Hours')
        plt.ylabel('Utility')
        plt.title('Utility Function over Time')
        plt.legend()
        plt.show()

    return param_dict


def get_shortest_distance(G, source, target):
    """
    返回图G中两点之间的最短距离。

    :param G: NetworkX图对象。
    :param source: 起点节点。
    :param target: 终点节点。
    :return: 两点之间的最短距离。
    """
    distance = nx.shortest_path_length(G, source=source, target=target, weight='length')
    return distance


def get_shortest_path(G, source, target):
    """
    返回图G中两点之间的最短路径。

    :param G: NetworkX图对象。
    :param source: 起点节点。
    :param target: 终点节点。
    :return: 两点之间的最短路径（节点序列）。
    """
    path = nx.shortest_path(G, source=source, target=target, weight='length')
    return path


def update_state(current_state, action, G):
    """
    根据当前状态和行动更新环境状态。

    :param current_state: 当前的环境状态。
    :param action: 执行的行动。
    :param G: NetworkX图对象。
    :return: 新的环境状态。
    """
    # 示例代码，需要根据您的状态和行动定义进行调整
    new_state = None
    # 更新逻辑
    return new_state


def calculate_reward(state, action, new_state):
    """
    根据行动前后的状态计算奖励。

    :param state: 行动前的状态。
    :param action: 执行的行动。
    :param new_state: 行动后的状态。
    :return: 奖励值。
    """
    reward = 0
    # 奖励计算逻辑
    return reward
