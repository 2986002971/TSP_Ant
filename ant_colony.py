import numpy as np
from numba import float64, int64, jit


@jit(
    float64[:](int64, int64[:], float64[:, :], float64[:, :], float64, float64),
    nopython=True,
)
def calculate_probabilities(current, unvisited, pheromone, heuristic, alpha, beta):
    """计算转移概率"""
    n = len(unvisited)
    probs = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if unvisited[i]:
            probs[i] = (pheromone[current, i] ** alpha) * (
                heuristic[current, i] ** beta
            )

    # 归一化
    total = probs.sum()
    if total > 0:
        probs = probs / total

    return probs


@jit(int64(float64[:]), nopython=True)
def choose_next_city(probabilities):
    """根据概率选择下一个城市"""
    r = np.random.random()
    cum_prob = 0.0
    for i, prob in enumerate(probabilities):
        cum_prob += prob
        if r <= cum_prob:
            return i
    return len(probabilities) - 1


@jit(int64[:](float64[:, :], float64[:, :], float64, float64, int64), nopython=True)
def construct_solution(pheromone, heuristic, alpha, beta, n_cities):
    """单只蚂蚁构建解"""
    path = np.zeros(n_cities, dtype=np.int64)
    unvisited = np.ones(n_cities, dtype=np.int64)

    # 随机选择起始城市
    current = np.random.randint(0, n_cities)
    path[0] = current
    unvisited[current] = 0

    # 构建路径
    for i in range(1, n_cities):
        # 计算转移概率
        probs = calculate_probabilities(
            current, unvisited, pheromone, heuristic, alpha, beta
        )
        # 根据概率选择下一个城市
        current = choose_next_city(probs)
        path[i] = current
        unvisited[current] = 0

    return path


@jit(float64(int64[:], float64[:, :]), nopython=True)
def calculate_path_distance(path, distance_matrix):
    """计算路径总距离"""
    total_distance = 0.0
    for i in range(len(path) - 1):
        total_distance += distance_matrix[path[i], path[i + 1]]
    # 返回起点
    total_distance += distance_matrix[path[-1], path[0]]
    return total_distance


@jit(nopython=True)
def update_pheromone(pheromone, paths, distances, rho, q, min_pheromone, max_pheromone):
    """更新信息素"""
    n = len(pheromone)
    # 信息素挥发
    pheromone *= 1 - rho

    # 信息素增加
    for path, dist in zip(paths, distances):
        delta = q / dist
        for i in range(len(path) - 1):
            current = path[i]
            next_city = path[i + 1]
            pheromone[current, next_city] += delta
            pheromone[next_city, current] += delta
        # 处理回路
        pheromone[path[-1], path[0]] += delta
        pheromone[path[0], path[-1]] += delta

    # 限制信息素范围（由于numba不支持广播，使用显式循环）
    for i in range(n):
        for j in range(n):
            if pheromone[i, j] < min_pheromone:
                pheromone[i, j] = min_pheromone
            elif pheromone[i, j] > max_pheromone:
                pheromone[i, j] = max_pheromone


# 主类
class AntColonyTSP:
    def __init__(
        self,
        distance_matrix,
        n_ants=None,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        q=1.0,
        max_iterations=100,
        min_pheromone=0.1,
        max_pheromone=10.0,
    ):
        self.distance_matrix = np.array(distance_matrix, dtype=np.float64)
        self.n_cities = len(distance_matrix)
        self.n_ants = n_ants if n_ants else self.n_cities
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.max_iterations = max_iterations
        self.min_pheromone = min_pheromone
        self.max_pheromone = max_pheromone

        # 初始化信息素矩阵
        self.pheromone = (
            np.ones((self.n_cities, self.n_cities), dtype=np.float64)
            * self.max_pheromone
        )
        # 计算启发式信息
        self.heuristic = 1 / (self.distance_matrix + 1e-10)

        # 记录最优解
        self.best_path = None
        self.best_distance = float("inf")
        self.best_path_history = []

    def solve(self):
        """运行蚁群算法"""
        for iteration in range(self.max_iterations):
            # 构建所有蚂蚁的解
            paths = np.array(
                [
                    construct_solution(
                        self.pheromone,
                        self.heuristic,
                        self.alpha,
                        self.beta,
                        self.n_cities,
                    )
                    for _ in range(self.n_ants)
                ]
            )

            # 计算路径长度
            distances = np.array(
                [calculate_path_distance(path, self.distance_matrix) for path in paths]
            )

            # 更新信息素
            update_pheromone(
                self.pheromone,
                paths,
                distances,
                self.rho,
                self.q,
                self.min_pheromone,
                self.max_pheromone,
            )

            # 更新最优解
            min_distance = distances.min()
            if min_distance < self.best_distance:
                self.best_distance = min_distance
                self.best_path = paths[distances.argmin()].copy()

            self.best_path_history.append(self.best_distance)
