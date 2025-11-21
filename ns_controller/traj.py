import numpy as np


class TargetTraj:
    """
    双机目标轨迹生成：
    - FLAG=0: UAV0，高度 z=5.0 m
    - FLAG=1: UAV1，高度 z=5.5 m

    两架飞机在水平面上的轨迹完全一致，均为圆轨迹：
      x = R*sin(w*t)
      y = R*(1 - cos(w*t))
      z = {5.0 | 5.5}
      yaw = 0.0

    若 t < 0，视为尚未开始跟踪，保持在对应高度的起始点 (0,0,z)。
    """

    def __init__(self, FLAG: int = 0):
        self.FLAG = int(FLAG)
        self.R = 6.0                 # 圆轨迹半径
        self.w = (2 * np.pi) / 30.0  # 角速度（对应 30s 一圈）

        # 根据 FLAG 选择高度
        if self.FLAG == 0:
            self.h_default = 5.0
        elif self.FLAG == 1:
            self.h_default = 5.5
        else:
            # 兜底：未知 FLAG 时按 UAV0 处理
            self.h_default = 5.0

    # 目标位置
    def pose(self, t: float):
        if t >= 0:
            x = self.R * np.sin(self.w * t)
            y = self.R * (1 - np.cos(self.w * t))
            z = self.h_default
            return np.array([[x], [y], [z]])
        else:
            # 尚未开始，停在原点上方对应高度
            return np.array([[0.0], [0.0], [self.h_default]])

    # 目标速度
    def velo(self, t: float):
        if t >= 0:
            vx = self.R * self.w * np.cos(self.w * t)
            vy = self.R * self.w * np.sin(self.w * t)
            vz = 0.0
            return np.array([[vx], [vy], [vz]])
        else:
            return np.array([[0.0], [0.0], [0.0]])

    # 目标加速度
    def acce(self, t: float):
        if t >= 0:
            ax = -self.R * (self.w ** 2) * np.sin(self.w * t)
            ay = self.R * (self.w ** 2) * np.cos(self.w * t)
            az = 0.0
            return np.array([[ax], [ay], [az]])
        else:
            return np.array([[0.0], [0.0], [0.0]])

    # 给定偏航角
    def yaw(self, t: float):
        return 0.0