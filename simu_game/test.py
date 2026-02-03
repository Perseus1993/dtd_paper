import math
import os

import networkx as nx
import sys
from matplotlib import pyplot as plt

if __name__ == '__main__':
    from mesa import Agent, Model
    from mesa.time import RandomActivation
    from mesa.space import MultiGrid
    import pygame
    import random
    import sys

    # Pygame 显示相关设置
    WIDTH, HEIGHT = 800, 800
    SIDEBAR_WIDTH = 200
    TOTAL_WIDTH = WIDTH + SIDEBAR_WIDTH
    GRID_SIZE = 40
    CELL_SIZE = WIDTH // GRID_SIZE

    # 颜色定义
    WHITE = (255, 255, 255)
    GRAY = (200, 200, 200)
    GREEN = (0, 255, 0)  # 住宅区
    BLUE = (0, 0, 255)  # 工作区
    YELLOW = (255, 255, 0)  # 公园
    ORANGE = (255, 165, 0)  # 商场
    RED = (255, 0, 0)  # 居民

    # Pygame 初始化
    pygame.init()
    screen = pygame.display.set_mode((TOTAL_WIDTH, HEIGHT))
    pygame.display.set_caption("城市模拟")


    class ResidentAgent(Agent):
        def __init__(self, unique_id, model, home):
            super().__init__(unique_id, model)
            self.home = home
            self.position = home
            self.schedule = []
            self.current_task = 0
            self.path = []
            self.health = 100  # 初始健康值
            self.mood = 100  # 初始心情值

            self.color = (
                random.randint(50, 150),  # 红色通道
                random.randint(50, 150),  # 绿色通道
                random.randint(50, 150),  # 蓝色通道
            )
            # 当前停留时间
            self.stay_counter = 0
            self.friends = []


            # 随机选择一个图像文件
            img_folder = os.path.join("..", "img")  # 图像文件夹路径
            img_files = [f for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]  # 获取所有图片文件
            selected_img = random.choice(img_files)  # 随机选择一个文件
            img_path = os.path.join(img_folder, selected_img)
            print(img_path)
            # 加载图像并调整大小
            self.image = pygame.image.load(img_path)
            self.image = pygame.transform.scale(self.image, (CELL_SIZE , CELL_SIZE ))  # 调整为适合格子的大小


        def plan_path(self, destination_poi):
                """规划从当前位置到目标 POI 的路径"""
                current_node = self.model.find_nearest_road_node(self.position)
                dest_node = destination_poi["nearest_node"]
                if current_node and dest_node in self.model.road_network.nodes:
                    try:
                        self.path = nx.shortest_path(self.model.road_network, current_node, dest_node)
                    except nx.NetworkXNoPath:
                        self.path = []
                else:
                    self.path = []

        def step(self):
            """执行日程中的任务"""
            if not self.schedule or self.current_task >= len(self.schedule):
                # 如果所有任务完成，将居民从模型中移除
                print(f"Agent {self.unique_id} has completed all tasks and will be removed.")
                self.model.grid.remove_agent(self)  # 从网格中移除
                self.model.schedule.remove(self)  # 从调度器中移除
                return

            # 如果正在停留
            if self.stay_counter > 0:
                self.stay_counter -= 1
                self.health = min(self.health + 2, 100)  # 健康值最多为100
                self.mood = min(self.mood + 2, 100)  # 心情最多为100
                print(
                    f"Agent {self.unique_id} is staying at position {self.position}. Stay counter: {self.stay_counter}")

                return

            # 获取当前任务
            task = self.schedule[self.current_task]
            destination = task["destination"]["nearest_node"]

            # 如果没有路径，则规划路径
            if not self.path:
                self.plan_path(task["destination"])

            if self.path:
                self.position = self.path.pop(0)
                self.health = max(self.health - 1, 0)
                self.mood = max(self.mood - 1, 0)

                if not self.path:
                    self.stay_counter = task["stay_time"]
                    self.current_task += 1

        def add_friend(self, friend):
            """将朋友添加到社交网络中"""
            self.friends.append(friend)




    class CityModel(Model):
        def __init__(self, width, height, num_residents, screen):
            super().__init__()
            self.grid = MultiGrid(width, height, torus=False)
            self.schedule = RandomActivation(self)
            self.grid_size = width
            self.grid_matrix = [[0 for _ in range(width)] for _ in range(height)]
            self.road_network = nx.Graph()  # 初始化道路网络

            self.screen = screen

            # 路网和 POI 初始化
            self.pois = []
            self.generate_road_network()
            self.initialize_pois()

            # 创建全局社交网络
            self.social_network = nx.Graph()  # 创建全局社交网络图
            self.residents = []

            # 创建居民
            for i in range(num_residents):
                home = random.choice([p for p in self.pois if p["type"] == "home"])
                resident = ResidentAgent(i, self, home=home["position"])
                self.schedule.add(resident)
                self.grid.place_agent(resident, home["position"])
                print(f"Resident {resident.unique_id} initialized at {home['position']}")

                # 创建日程，确保目标是 POI 字典
                work = random.choice([p for p in self.pois if p["type"] == "work"])
                park = random.choice([p for p in self.pois if p["type"] == "park"])
                mall = random.choice([p for p in self.pois if p["type"] == "mall"])
                resident.schedule = [
                    {"time": "08:00", "destination": work, "stay_time": 5},
                    {"time": "12:00", "destination": park, "stay_time": 3},
                    {"time": "15:00", "destination": mall, "stay_time": 4},
                    {"time": "18:00", "destination": home, "stay_time": 0},
                ]
                self.social_network.add_node(resident.unique_id)  # 每个居民是一个节点

                # 随机为每个居民建立朋友关系（30% 的几率）
                for other_resident in self.schedule.agents:
                    if other_resident != resident and random.random() < 0.3:  # 30% 的几率成为朋友
                        self.social_network.add_edge(resident.unique_id, other_resident.unique_id)

                self.residents.append(resident)

            for resident in self.schedule.agents:
                print(f"Resident {resident.unique_id} added to schedule")

            self.update_friendships()

        def update_friendships(self):
            """更新所有居民的朋友关系"""
            for agent in self.schedule.agents:
                if isinstance(agent, ResidentAgent):
                    # 通过遍历所有居民检查是否在一定距离内并成为朋友
                    for other_agent in self.schedule.agents:
                        if other_agent != agent and self.are_nearby(agent, other_agent) and random.random() < 0.3:
                            self.add_friendship(agent, other_agent)

        def are_nearby(self, agent1, agent2):
            """判断两个居民是否在邻近的位置"""
            distance = abs(agent1.position[0] - agent2.position[0]) + abs(agent1.position[1] - agent2.position[1])
            return distance <= 2  # 定义为邻近的标准

        def add_friendship(self, agent1, agent2):
            """在社交网络中建立朋友关系"""
            self.social_network.add_edge(agent1.unique_id, agent2.unique_id)
            agent1.add_friend(agent2)  # 在居民的朋友列表中添加朋友
            agent2.add_friend(agent1)  # 双向添加朋友关系

        def find_nearest_road_node(self, position):
            """找到离指定位置最近的道路节点"""
            x, y = position
            nearest_node = None
            min_distance = float('inf')
            for node in self.road_network.nodes:
                distance = abs(node[0] - x) + abs(node[1] - y)  # 曼哈顿距离
                if distance < min_distance:
                    nearest_node = node
                    min_distance = distance
            return nearest_node

        def generate_road_network(self):
            """生成更复杂和多样化的道路网络"""

            # Step 1: 生成主干道
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    # 随机生成一些弯曲的主干道
                    if random.random() < 0.1 or x % 4 == 0 or y % 4 == 0:  # 增加随机性
                        self.grid_matrix[y][x] = 3  # 标记为道路
                        self.road_network.add_node((x, y))  # 添加节点到道路网络

                        # 添加主干道相邻的交叉路口
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            neighbor = (x + dx, y + dy)
                            if 0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size:
                                if self.grid_matrix[neighbor[1]][neighbor[0]] == 3:
                                    self.road_network.add_edge((x, y), neighbor)  # 添加边

            # Step 2: 创建交叉路口
            num_crossroads = random.randint(self.grid_size // 4, self.grid_size // 2)  # 随机生成交叉点
            for _ in range(num_crossroads):
                # 随机选择交叉点位置
                x, y = random.randint(1, self.grid_size - 2), random.randint(1, self.grid_size - 2)
                self.grid_matrix[y][x] = 3  # 将这些位置设置为主干道
                self.road_network.add_node((x, y))

                # 将交叉点与周围的节点连接
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor = (x + dx, y + dy)
                    if 0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size:
                        self.road_network.add_edge((x, y), neighbor)

            # Step 3: 生成随机的次干道
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    if self.grid_matrix[y][x] == 0 and random.random() < 0.05:  # 次干道的生成
                        self.grid_matrix[y][x] = 4  # 标记为次干道
                        self.road_network.add_node((x, y))  # 添加节点到道路网络

                        # 连接次干道与周围的道路
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            neighbor = (x + dx, y + dy)
                            if 0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size:
                                if self.grid_matrix[neighbor[1]][neighbor[0]] == 3:  # 连接到主干道
                                    self.road_network.add_edge((x, y), neighbor)  # 添加边

            # Step 4: 使用最小生成树优化道路网络
            # 通过Prim或Kruskal算法确保道路之间的连通性并优化结构
            mst = nx.minimum_spanning_tree(self.road_network)
            self.road_network = mst



        def check_and_draw_friend_connections(self):
            for agent in self.schedule.agents:
                if isinstance(agent, ResidentAgent):
                    for friend in agent.friends:
                        # 检查朋友是否在附近
                        distance = abs(agent.position[0] - friend.position[0]) + abs(
                            agent.position[1] - friend.position[1])
                        if distance <= 2:  # 如果朋友靠近（距离<=2）
                            self.draw_friend_connection(agent, friend)  # 绘制连接线

        def draw_friend_connection(self, agent, friend):
            agent_pos = agent.position
            friend_pos = friend.position

            print(f"Drawing connection between {agent.unique_id} and {friend.unique_id}")
            print(f"Agent position: {agent_pos}, Friend position: {friend_pos}")

            # 将坐标转换为屏幕上的位置
            agent_x, agent_y = agent_pos[0] * CELL_SIZE + CELL_SIZE // 2, agent_pos[1] * CELL_SIZE + CELL_SIZE // 2
            friend_x, friend_y = friend_pos[0] * CELL_SIZE + CELL_SIZE // 2, friend_pos[1] * CELL_SIZE + CELL_SIZE // 2

            # 绘制橙色线条连接朋友
            line_color = (255, 165, 0)  # 橙色
            pygame.draw.line(self.screen, line_color, (agent_x, agent_y), (friend_x, friend_y), 10)

            # 绘制朋友位置的框框
            # 偏移量为 CELL_SIZE // 4 让框适应格子大小
            agent_rect = pygame.Rect(agent_pos[0] * CELL_SIZE, agent_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            friend_rect = pygame.Rect(friend_pos[0] * CELL_SIZE, friend_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            # 绘制橙色框
            pygame.draw.rect(self.screen, line_color, agent_rect, 3)  # 边框宽度为3
            pygame.draw.rect(self.screen, line_color, friend_rect, 3)  # 边框宽度为3

            # 调试：显示两点的坐标
            print(f"Line drawn from {agent_x}, {agent_y} to {friend_x}, {friend_y}")

        def initialize_pois(self):
            num_homes = 10
            num_work = 5
            num_park = 3
            num_mall = 2

            def find_nearest_road_node(position):
                x, y = position
                nearest_node = None
                min_distance = float('inf')
                for node in self.road_network.nodes:
                    distance = abs(node[0] - x) + abs(node[1] - y)  # 曼哈顿距离
                    if distance < min_distance:
                        nearest_node = node
                        min_distance = distance
                return nearest_node

            def add_poi(poi_type, num):
                for _ in range(num):
                    x, y = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
                    while self.grid_matrix[y][x] != 0:  # 确保 POI 不覆盖道路或其他 POI
                        x, y = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
                    position = (x, y)
                    nearest_node = find_nearest_road_node(position)
                    self.pois.append({"type": poi_type, "position": position, "nearest_node": nearest_node})
                    if poi_type == "home":
                        self.grid_matrix[y][x] = 1
                    elif poi_type == "work":
                        self.grid_matrix[y][x] = 2
                    elif poi_type == "park":
                        self.grid_matrix[y][x] = 4
                    elif poi_type == "mall":
                        self.grid_matrix[y][x] = 5
                    print(f"POI {poi_type} initialized at {position}, nearest road node: {nearest_node}")

            add_poi("home", num_homes)
            add_poi("work", num_work)
            add_poi("park", num_park)
            add_poi("mall", num_mall)

        def update_social_network(self):
            """更新社交网络中居民的心情"""
            for resident in self.schedule.agents:
                # 检查朋友是否在附近，并增加心情
                for friend_id in self.social_network.neighbors(resident.unique_id):
                    friend = next(agent for agent in self.schedule.agents if agent.unique_id == friend_id)
                    # 判断朋友是否靠近，若靠近增加心情值
                    if abs(resident.position[0] - friend.position[0]) <= 2 and abs(
                            resident.position[1] - friend.position[1]) <= 2:
                        print(f"Agent {resident.unique_id} is near Agent {friend.unique_id}, increasing mood!")
                        resident.mood = min(resident.mood + 1, 100)  # 心情增加

        def step(self):
            print("Model stepping")
            self.schedule.step()
            self.update_friendships()  # 每个时间步更新朋友关系
            # self.check_and_draw_friend_connections()  # 绘制朋友之间的连线



    def draw_grid(model):
        screen.fill((50, 50, 50))
        for row in range(model.grid_size):
            for col in range(model.grid_size):
                x, y = col * CELL_SIZE, row * CELL_SIZE
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                if model.grid_matrix[row][col] == 0:  # 空地
                    pygame.draw.rect(screen, WHITE, rect)
                elif model.grid_matrix[row][col] == 1:  # 住宅
                    pygame.draw.rect(screen, GREEN, rect)
                elif model.grid_matrix[row][col] == 2:  # 工作区
                    pygame.draw.rect(screen, BLUE, rect)
                elif model.grid_matrix[row][col] == 4:  # 公园
                    pygame.draw.rect(screen, YELLOW, rect)
                elif model.grid_matrix[row][col] == 5:  # 商场
                    pygame.draw.rect(screen, ORANGE, rect)
                elif model.grid_matrix[row][col] == 3:  # 道路
                    pygame.draw.rect(screen, GRAY, rect)
                pygame.draw.rect(screen, GRAY, rect, 1)  # 网格线

        # 绘制箭头和居民
        for agent in model.schedule.agents:
            # 如果居民在执行停留任务
            if agent.stay_counter > 0:
                position = agent.schedule[agent.current_task]["destination"]["position"]
                destination = position  # 停留时箭头指向当前位置
            else:
                position = agent.position
                if agent.schedule and agent.current_task < len(agent.schedule):
                    destination = agent.schedule[agent.current_task]["destination"]["nearest_node"]
                else:
                    destination = agent.position

            # 绘制箭头：仅在居民有明确目标时绘制
            if position != destination:
                draw_arrow(screen, position, destination, agent.color, agent)

            # 绘制居民图像
            if agent.image:  # 仅当图像加载成功时才绘制
                # 计算居民在格子中的位置
                cell_x, cell_y = position[0] * CELL_SIZE, position[1] * CELL_SIZE

                # 绘制居民信息条（如健康、心情等）
                draw_agent_info(screen, agent, cell_x, cell_y)

                # 绘制居民图像：确保图像绘制在格子内
                image_x, image_y = cell_x + CELL_SIZE // 4, cell_y + CELL_SIZE // 4
                screen.blit(agent.image, (image_x, image_y))  # 绘制图像，位于格子内

            else:
                print(f"Agent {agent.unique_id} has no image!")

        model.check_and_draw_friend_connections()

        # pygame.display.flip()  # 更新屏幕


    def draw_agent_info(screen, agent, cell_x, cell_y):
        # 使用较小的字体来显示健康和心情
        font = pygame.font.SysFont(None, 16)  # 调整字体大小
        health_text = font.render(f"Health: {agent.health}", True, (255, 0, 0))
        mood_text = font.render(f"Mood: {agent.mood}", True, (0, 0, 255))

        # 计算文本和条形图的显示位置
        text_offset = 2  # 减少文本与条形图的间距

        # 绘制健康文本和健康条
        screen.blit(health_text, (cell_x + CELL_SIZE // 4, cell_y + 5))  # 放置健康文本
        draw_health_bar(screen, agent, cell_x + CELL_SIZE // 4, cell_y + 20)  # 绘制健康条

        # 绘制心情文本和心情条
        screen.blit(mood_text, (cell_x + CELL_SIZE // 4, cell_y + 30))  # 放置心情文本
        draw_mood_bar(screen, agent, cell_x + CELL_SIZE // 4, cell_y + 45)  # 绘制心情条

        # 绘制头像
        # screen.blit(agent.image, (cell_x + CELL_SIZE // 4, cell_y + 55))  # 头像放在下方


    def draw_health_bar(screen, agent, x, y):
        """绘制健康值条"""
        health_percentage = agent.health / 100
        bar_width = 30  # 缩小条形图的宽度
        bar_height = 4  # 缩小条形图的高度
        color = (255, 0, 0)  # 红色表示健康条
        pygame.draw.rect(screen, color, (x, y, bar_width * health_percentage, bar_height))


    def draw_mood_bar(screen, agent, x, y):
        """绘制心情值条"""
        mood_percentage = agent.mood / 100
        bar_width = 30  # 缩小条形图的宽度
        bar_height = 4  # 缩小条形图的高度
        color = (0, 0, 255)  # 蓝色表示心情条
        pygame.draw.rect(screen, color, (x, y, bar_width * mood_percentage, bar_height))


    def draw_arrow(screen, position, destination, color, agent):
        """绘制箭头（三角形），位置基于人物头像旁边"""

        # 如果目标位置与当前位置相同，返回
        if position == destination:
            return

        x, y = position
        dx, dy = destination

        # 计算箭头的角度
        angle = math.atan2(dy - y, dx - x)

        # 计算箭头的位置，位于头像旁边
        arrow_offset = CELL_SIZE // 3  # 偏移量，可以根据需要调整
        arrow_x = (x * CELL_SIZE) + CELL_SIZE // 2 + arrow_offset * math.cos(angle)
        arrow_y = (y * CELL_SIZE) + CELL_SIZE // 2 + arrow_offset * math.sin(angle)

        # 计算箭头三角形的大小
        size = CELL_SIZE // 4

        # 计算箭头三角形顶点
        point1 = (arrow_x + size * math.cos(angle), arrow_y + size * math.sin(angle))
        point2 = (arrow_x + size * math.cos(angle + 2.5), arrow_y + size * math.sin(angle + 2.5))
        point3 = (arrow_x + size * math.cos(angle - 2.5), arrow_y + size * math.sin(angle - 2.5))

        # 绘制箭头
        pygame.draw.polygon(screen, color, [point1, point2, point3])



    def draw_sidebar(model):
        """绘制侧边栏，显示社交网络图和居民状态"""
        sidebar_width = SIDEBAR_WIDTH
        pygame.draw.rect(screen, GRAY, (WIDTH, 0, sidebar_width, HEIGHT))

        font = pygame.font.SysFont(None, 24)
        title = font.render("Schedule & Status", True, WHITE)
        screen.blit(title, (WIDTH + 10, 10))

        # 绘制社交网络图
        G = model.social_network  # 获取全局社交网络

        # 创建社交网络图的圆形布局
        pos = nx.circular_layout(G)  # 使用圆形布局

        # 设置图形的尺寸，确保图形适配侧边栏的宽度
        fig = plt.figure(figsize=(5, 5))  # 固定图形大小
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_weight='bold', font_size=12)

        # 使图形显示为正方形
        plt.axis('equal')  # 强制保持比例，确保圆形布局不会被拉伸

        # 保存社交网络图为图片
        plt.savefig('social_network.png', format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 加载并显示社交网络图
        social_network_image = pygame.image.load("social_network.png")

        # 获取图像的原始尺寸
        img_width, img_height = social_network_image.get_size()

        # 设置图像的最大尺寸（侧边栏的宽度）
        max_width = sidebar_width - 20  # 留出一些空白
        max_height = HEIGHT - 60  # 留出顶部空间和底部空间

        # 计算缩放比例，保持图像比例
        scale_factor = min(max_width / img_width, max_height / img_height)

        # 缩放图像
        scaled_width = int(img_width * scale_factor)
        scaled_height = int(img_height * scale_factor)
        social_network_image = pygame.transform.scale(social_network_image, (scaled_width, scaled_height))

        # 绘制图像
        screen.blit(social_network_image, (WIDTH + 10, 40))  # 显示图像

        # 显示居民状态
        y_offset = 40 + 200
        for i, agent in enumerate(model.schedule.agents[:5]):  # 最多显示 5 个居民
            status = f"Agent {i}: Mood: {agent.mood}, Health: {agent.health}"
            header = font.render(status, True, WHITE)
            screen.blit(header, (WIDTH + 10, y_offset))
            y_offset += 20


    # 初始化模型
    model = CityModel(GRID_SIZE, GRID_SIZE, 3, screen)

    # 主循环
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        model.step()  # 更新智能体逻辑
        screen.fill((50, 50, 50))  # 清屏
        draw_grid(model)  # 绘制网格和居民
        draw_sidebar(model)  # 绘制侧边栏
        pygame.display.flip()
        clock.tick(3)  # 控制帧率

