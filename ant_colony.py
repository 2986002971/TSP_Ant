import numpy as np
from numba import float32, int32, jit, prange, void


@jit(
    float32[:](int32, int32[:], float32[:, :], float32[:, :], float32, float32),
    nopython=True,
)
def calculate_probabilities(current, unvisited, pheromone, heuristic, alpha, beta):
    """计算转移概率"""
    n = len(unvisited)
    probs = np.zeros(n, dtype=np.float32)

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


@jit(int32(float32[:]), nopython=True)
def choose_next_city(probabilities):
    """根据概率选择下一个城市"""
    r = np.random.random()
    cum_prob = 0.0
    for i, prob in enumerate(probabilities):
        cum_prob += prob
        if r <= cum_prob:
            return i
    return len(probabilities) - 1


@jit(int32[:](float32[:, :], float32[:, :], float32, float32, int32), nopython=True)
def construct_solution(pheromone, heuristic, alpha, beta, n_cities):
    """单只蚂蚁构建解"""
    path = np.zeros(n_cities, dtype=np.int32)
    unvisited = np.ones(n_cities, dtype=np.int32)

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


@jit(float32(int32[:], float32[:, :]), nopython=True)
def calculate_path_distance(path, distance_matrix):
    """计算路径总距离"""
    total_distance = 0.0
    for i in range(len(path) - 1):
        total_distance += distance_matrix[path[i], path[i + 1]]
    # 返回起点
    total_distance += distance_matrix[path[-1], path[0]]
    return total_distance


@jit(
    void(
        float32[:, :],  # pheromone
        int32[:, :],  # paths
        float32[:],  # distances
        float32,  # rho
        float32,  # q
        float32,  # min_pheromone
        float32,  # max_pheromone
    ),
    nopython=True,
)
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

    # 限制信息素范围
    for i in range(n):
        for j in range(n):
            if pheromone[i, j] < min_pheromone:
                pheromone[i, j] = min_pheromone
            elif pheromone[i, j] > max_pheromone:
                pheromone[i, j] = max_pheromone


@jit(
    int32[:, :](
        int32,  # n_ants
        float32[:, :],  # pheromone
        float32[:, :],  # heuristic
        float32,  # alpha
        float32,  # beta
        int32,  # n_cities
    ),
    nopython=True,
    parallel=True,
)
def construct_solutions_parallel(n_ants, pheromone, heuristic, alpha, beta, n_cities):
    """并行构建多只蚂蚁的解"""
    paths = np.zeros((n_ants, n_cities), dtype=np.int32)
    for ant in prange(n_ants):
        paths[ant] = construct_solution(pheromone, heuristic, alpha, beta, n_cities)
    return paths


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
        self.distance_matrix = np.array(distance_matrix, dtype=np.float32)
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
            np.ones((self.n_cities, self.n_cities), dtype=np.float32)
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
            # 并行构建所有蚂蚁的解
            paths = construct_solutions_parallel(
                self.n_ants,
                self.pheromone,
                self.heuristic,
                self.alpha,
                self.beta,
                self.n_cities,
            )

            # 计算路径长度 - 确保使用float32
            distances = np.array(
                [calculate_path_distance(path, self.distance_matrix) for path in paths],
                dtype=np.float32,  # 明确指定类型为float32
            )

            # 更新信息素 - 确保所有参数都是float32
            update_pheromone(
                self.pheromone,
                paths,
                distances,
                np.float32(self.rho),
                np.float32(self.q),
                np.float32(self.min_pheromone),
                np.float32(self.max_pheromone),
            )

            # 更新最优解
            min_distance = distances.min()
            if min_distance < self.best_distance:
                self.best_distance = min_distance
                self.best_path = paths[distances.argmin()].copy()

            self.best_path_history.append(self.best_distance)
