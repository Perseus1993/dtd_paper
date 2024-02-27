from enum import Enum, auto
class NodeType(Enum):
    HOME = auto()
    EDU = auto()
    SHOPPING = auto()
    WORK = auto()
    LEISURE = auto()
    TRANSFER = auto()  # 表示是个连接节点，例如，只作为从一个节点到另一个节点的途径


class Node:
    def __init__(self, id, node_type: NodeType):
        self.id = id  # 唯一的节点ID
        self.node_type = node_type  # 节点的类型

    def __repr__(self):
        return f"Node(id={self.id},type={self.node_type})"