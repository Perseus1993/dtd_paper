import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def _data_path(filename: str) -> str:
    base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "backup", "suffix", "data"))
    return os.path.join(base_dir, filename)


def load_graph(draw: bool = True):
    net = pd.read_csv(_data_path('SiouxFalls_net.tntp'), skiprows=8, sep='\t').drop(['~', ';'], axis=1)
    net['edge'] = net.index + 1
    flow = pd.read_csv(_data_path('SiouxFalls_flow.tntp'), sep='\t').drop(['From ', 'To '], axis=1)
    flow.rename(columns={"Volume ": "flow", "Cost ": "cost"}, inplace=True)
    node_coord = pd.read_csv(_data_path('SiouxFalls_node.tntp'), sep='\t').drop([';'], axis=1)  # Actual Sioux Falls coordinate
    node_xy = pd.read_csv(_data_path('SiouxFalls_node_xy.tntp'), sep='\t')  # X,Y position for good visualization

    # dataframe containing all link attributes
    sioux_falls_df = pd.concat([net, flow], axis=1)

    G = nx.from_pandas_edgelist(sioux_falls_df, 'init_node', 'term_node',
                                ['capacity', 'length', 'free_flow_time', 'b', 'power', 'speed', 'toll', 'link_type',
                                 'edge', 'flow', 'cost'], create_using=nx.MultiDiGraph())

    pos_coord = dict([(i, (a, b)) for i, a, b in zip(node_coord.Node, node_coord.X, node_coord.Y)])

    # for better looking graph
    pos_xy = dict([(i, (a, b)) for i, a, b in zip(node_xy.Node, node_xy.X, node_xy.Y)])

    for n, p in pos_coord.items():
        G.nodes[n]['pos_coord'] = p

    for n, p in pos_xy.items():
        G.nodes[n]['pos_xy'] = p

    origin = [14, 15, 22, 23]
    destination = [4, 5, 6, 8, 9, 10, 11, 16, 17, 18]

    origin = [14, 15, 22, 23]

    destination = [4, 5, 6, 8, 9, 10, 11, 16, 17, 18]

    for n in G.nodes:
        if n in destination:
            G.nodes[n]['O/D'] = 'destination'
            G.nodes[n]['color'] = 'green'
        elif n in origin:
            G.nodes[n]['O/D'] = 'origin'
            G.nodes[n]['color'] = 'red'
        else:
            G.nodes[n]['O/D'] = 'transfer_node'
            G.nodes[n]['color'] = 'blue'

    demand = [2000, 9000, 7000, 2000]
    capacity = [5000, 4000, 6000, 5000, 4000, 4000, 4000, 4000, 1000, 5000]
    node_demand = dict([(i, a) for i, a in zip(origin, demand)])
    node_capacity = dict([(i, a) for i, a in zip(destination, capacity)])
    for n, p in node_demand.items():
        G.nodes[n]['demand'] = p
    for n, p in node_capacity.items():
        G.nodes[n]['capacity'] = p
    if draw:
        nx.draw(G, pos_coord, with_labels=True)
    return G, pos_coord, pos_xy, sioux_falls_df


def calculate_travel_time(G, edge_flows):
    path_times = {}
    for edge, flow in edge_flows.items():
        u, v, k = edge
        if G.has_edge(u, v, k):
            free_flow_time = G[u][v][k].get('free_flow_time', 0)  # 默认值为0
            capacity = G[u][v][k].get('capacity', 1)  # 避免除零错误
            travel_time = free_flow_time * (1 + 0.15 * (flow / capacity) ** 4)
            path_times[edge] = travel_time
        else:
            print(f"Warning: Edge {edge} not found in the graph.")
            # 如果需要，可以为这些边赋予默认旅行时间
            path_times[edge] = float('inf')  # 或其他默认值
    return path_times


if __name__ == '__main__':
    # 加载Sioux Falls网络
    G, pos_coord, pos_xy, _ = load_graph()

    # 假设你有一些edge_flows，下面是如何计算路径时间
    edge_flows = {
        (1, 2, 0): 500,  # 示例：边 (1, 2) 的流量为 500，0表示第一个边
    }

    # 检查 edge_flows 中的边是否在图中存在
    for edge in edge_flows.keys():
        if not G.has_edge(*edge):
            print(f"Error: Edge {edge} does not exist in the graph.")

    # 计算路径时间
    path_times = calculate_travel_time(G, edge_flows)

    # 查看计算结果
    print(path_times)