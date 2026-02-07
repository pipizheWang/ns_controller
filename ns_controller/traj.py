import numpy as np


class TargetTraj:
    """
    双机目标轨迹生成：
    - FLAG=0: UAV0，从 (0, 0, 5) 飞到 (4, 0, 5)       [X方向对飞]
    - FLAG=1: UAV1，从 (0, 0, 5.15) 飞到 (-4, 0, 5.15) [X方向对飞]
    - FLAG=2: UAV0，从 (0, 0, 5) 飞到 (0, 4, 5)       [Y方向对飞]
    - FLAG=3: UAV1，从 (0, 0, 5.15) 飞到 (0, -4, 5.15) [Y方向对飞]
    - FLAG=4: UAV0，从 (0, 0, 5) 飞到 (4, 4, 5)       [对角线对飞]
    - FLAG=5: UAV1，从 (0, 0, 5.15) 飞到 (-4, -4, 5.15) [对角线对飞]
    - FLAG=6: UAV0，在 (0,0,5)-(4,4,5) 范围随机飞行
    - FLAG=7: UAV1，在 (0,0,5.15)-(-4,-4,5.15) 范围随机飞行
    """

    def __init__(self, FLAG: int = 0, seed: int = None):
        self.FLAG = int(FLAG)
        self.T = 16.0  # 飞行时间（秒）
        self.is_random = False  # 是否为随机轨迹
        self.waypoints = None   # 随机航点列表
        self.segment_time = None # 每段飞行时间

        # 根据 FLAG 选择起点和终点
        if self.FLAG == 0:
            # 任务0: X方向对飞 - UAV0
            self.start = np.array([[0.0], [0.0], [5.0]])
            self.end = np.array([[4.0], [0.0], [5.0]])
        elif self.FLAG == 1:
            # 任务0: X方向对飞 - UAV1
            self.start = np.array([[0.0], [0.0], [5.15]])
            self.end = np.array([[-4.0], [0.0], [5.15]])
        elif self.FLAG == 2:
            # 任务1: Y方向对飞 - UAV0
            self.start = np.array([[0.0], [0.0], [5.0]])
            self.end = np.array([[0.0], [4.0], [5.0]])
        elif self.FLAG == 3:
            # 任务1: Y方向对飞 - UAV1
            self.start = np.array([[0.0], [0.0], [5.15]])
            self.end = np.array([[0.0], [-4.0], [5.15]])
        elif self.FLAG == 4:
            # 任务2: 对角线对飞 - UAV0
            self.start = np.array([[0.0], [0.0], [5.0]])
            self.end = np.array([[4.0], [4.0], [5.0]])
        elif self.FLAG == 5:
            # 任务2: 对角线对飞 - UAV1
            self.start = np.array([[0.0], [0.0], [5.15]])
            self.end = np.array([[-4.0], [-4.0], [5.15]])
        elif self.FLAG == 6:
            # 任务3: 随机飞行 - UAV0
            self.is_random = True
            self._generate_random_waypoints(
                x_range=(0.0, 4.0),
                y_range=(0.0, 4.0),
                z=5.0,
                num_waypoints=8,
                seed=seed
            )
        elif self.FLAG == 7:
            # 任务3: 随机飞行 - UAV1
            self.is_random = True
            self._generate_random_waypoints(
                x_range=(0.0, -4.0),
                y_range=(0.0, -4.0),
                z=5.15,
                num_waypoints=8,
                seed=seed
            )
        else:
            # 默认轨迹
            self.start = np.array([[-4.0], [0.0], [5.0]])
            self.end = np.array([[4.0], [0.0], [5.0]])

        # 对于非随机轨迹，计算速度
        if not self.is_random:
            self.velocity = (self.end - self.start) / self.T

    def _generate_random_waypoints(self, x_range: tuple, y_range: tuple, z: float, 
                                   num_waypoints: int = 8, seed: int = None):
        """
        生成随机航点序列
        
        Args:
            x_range: X坐标范围 (min, max)
            y_range: Y坐标范围 (min, max)
            z: 固定高度
            num_waypoints: 航点数量（包括起点和终点）
            seed: 随机种子
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 确保起点和终点
        x_min, x_max = min(x_range), max(x_range)
        y_min, y_max = min(y_range), max(y_range)
        
        # 生成随机航点
        self.waypoints = [np.array([[x_range[0]], [y_range[0]], [z]])]  # 起点
        
        for _ in range(num_waypoints - 2):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            self.waypoints.append(np.array([[x], [y], [z]]))
        
        self.waypoints.append(np.array([[x_range[1]], [y_range[1]], [z]]))  # 终点
        
        # 计算每段飞行时间
        self.segment_time = self.T / (num_waypoints - 1)
        
    def _get_random_segment(self, t: float):
        """
        获取当前时间点所在的航点段
        
        Returns:
            (start_point, end_point, segment_progress)
        """
        if t < 0:
            return self.waypoints[0], self.waypoints[0], 0.0
        elif t >= self.T:
            return self.waypoints[-1], self.waypoints[-1], 1.0
        
        # 计算当前在哪一段
        segment_idx = int(t / self.segment_time)
        segment_idx = min(segment_idx, len(self.waypoints) - 2)
        
        # 计算段内进度
        segment_progress = (t - segment_idx * self.segment_time) / self.segment_time
        segment_progress = min(segment_progress, 1.0)
        
        return self.waypoints[segment_idx], self.waypoints[segment_idx + 1], segment_progress

    # 目标位置
    def pose(self, t: float):
        if self.is_random:
            # 随机轨迹
            start_point, end_point, progress = self._get_random_segment(t)
            return start_point + (end_point - start_point) * progress
        else:
            # 直线轨迹
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
        if self.is_random:
            # 随机轨迹
            if t < 0:
                return np.array([[0.0], [0.0], [0.0]])
            elif t >= self.T:
                return np.array([[0.0], [0.0], [0.0]])
            else:
                # 计算当前段的速度
                segment_idx = int(t / self.segment_time)
                segment_idx = min(segment_idx, len(self.waypoints) - 2)
                velocity = (self.waypoints[segment_idx + 1] - self.waypoints[segment_idx]) / self.segment_time
                return velocity
        else:
            # 直线轨迹
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
        # 注意：随机轨迹在航点处会有瞬时加速度，但这里简化为0
        return np.array([[0.0], [0.0], [0.0]])

    # 给定偏航角
    def yaw(self, t: float):
        return 0.0