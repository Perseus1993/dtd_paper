import networkx as nx


# 假设G已经在这里被定义，例如：
# G = nx.DiGraph()
# G.add_edge(1, 2, road='101', length=100)

class Traffic_State:
    def __init__(self, G):
        self.G = G
        self.road_segments = self.build_road_segments()

    def build_road_segments(self):
        road_segments = {}
        for u, v, data in self.G.edges(data=True):
            # 使用(u, v)元组作为键，存储相关数据作为值
            road_segments[(u, v)] = {
                'length': data.get('length'),
                'cur_traffic': 0,
            }
        return road_segments

    def clear_traffic(self):
        for segment in self.road_segments.values():
            segment['cur_traffic'] = 0

    def get_speed(self, u, v):
        """返回道路段(u, v)的速度"""
        segment = self.road_segments[(u, v)]
        return max(6 - segment['cur_traffic'], 1)


