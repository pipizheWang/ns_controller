import numpy as np


class TargetTraj:
    """
    双机目标轨迹生成（直线对飞轨迹）：
    - FLAG=0: UAV0，从 (-2, 0, 5) 飞到 (2, 0, 5)
    - FLAG=1: UAV1，从 (2, 0, 5.5) 飞到 (-2, 0, 5.5)

    两架飞机沿直线匀速飞行，飞行时间 T 秒后到达终点并悬停。
    若 t < 0，视为尚未开始跟踪，保持在起始点。
    若 t >= T，到达终点后悬停。
    """

    def __init__(self, FLAG: int = 0):
        self.FLAG = int(FLAG)
        self.T = 10.0  # 飞行时间（秒）

        # 根据 FLAG 选择起点和终点
        if self.FLAG == 0:
            # UAV0: 从 (-2, 0, 5) 飞到 (2, 0, 5)
            self.start = np.array([[-2.0], [0.0], [5.0]])
            self.end = np.array([[2.0], [0.0], [5.0]])
        elif self.FLAG == 1:
            # UAV1: 从 (2, 0, 5.5) 飞到 (-2, 0, 5.5)
            self.start = np.array([[2.0], [0.0], [5.5]])
            self.end = np.array([[-2.0], [0.0], [5.5]])
        else:
            # 兜底：未知 FLAG 时按 UAV0 处理
            self.start = np.array([[-2.0], [0.0], [5.0]])
            self.end = np.array([[2.0], [0.0], [5.0]])

        # 计算匀速飞行的速度向量
        self.velocity = (self.end - self.start) / self.T

    # 目标位置
    def pose(self, t: float):
        if t < 0:
            # 尚未开始，停在起始点
            return self.start.copy()
        elif t <= self.T:
            # 匀速飞行中
            return self.start + self.velocity * t
        else:
            # 到达终点后悬停
            return self.end.copy()

    # 目标速度
    def velo(self, t: float):
        if t < 0:
            # 尚未开始
            return np.array([[0.0], [0.0], [0.0]])
        elif t <= self.T:
            # 匀速飞行中
            return self.velocity.copy()
        else:
            # 到达终点后悬停
            return np.array([[0.0], [0.0], [0.0]])

    # 目标加速度
    def acce(self, t: float):
        # 匀速直线运动，加速度始终为0
        return np.array([[0.0], [0.0], [0.0]])

    # 给定偏航角
    def yaw(self, t: float):
        return 0.0