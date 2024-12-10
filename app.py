import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ant_colony import AntColonyTSP
from map_data import distance_matrix, map_data


def run_aco(params):
    # 使用提供的距离矩阵
    aco = AntColonyTSP(
        distance_matrix=distance_matrix,
        n_ants=params["n_ants"],
        alpha=params["alpha"],
        beta=params["beta"],
        rho=params["rho"],
        q=params["q"],
        max_iterations=params["max_iterations"],
        min_pheromone=params["min_pheromone"],
        max_pheromone=params["max_pheromone"],
    )

    # 记录开始时间
    start_time = time.time()

    # 运行算法
    aco.solve()

    # 计算运行时间
    run_time = time.time() - start_time

    return aco, run_time


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("蚁群算法求解TSP问题")

    # 侧边栏参数设置
    st.sidebar.header("参数设置")

    params = {
        "n_ants": st.sidebar.slider("蚂蚁数量", 10, 100, 34),
        "alpha": st.sidebar.slider("信息素重要程度 (Alpha)", 0.1, 5.0, 1.0),
        "beta": st.sidebar.slider("启发式信息重要程度 (Beta)", 0.1, 5.0, 2.0),
        "rho": st.sidebar.slider("信息素挥发系数 (Rho)", 0.01, 0.5, 0.1),
        "q": st.sidebar.slider("信息素强度系数 (Q)", 0.1, 10.0, 1.0),
        "max_iterations": st.sidebar.slider("最大迭代次数", 10, 500, 100),
        "min_pheromone": st.sidebar.slider("最小信息素浓度", 0.01, 1.0, 0.1),
        "max_pheromone": st.sidebar.slider("最大信息素浓度", 1.0, 20.0, 10.0),
    }

    if st.sidebar.button("开始优化"):
        aco, run_time = run_aco(params)

        # 创建两列布局
        col1, col2 = st.columns(2)

        with col1:
            # 绘制收敛曲线
            fig1, ax1 = plt.subplots(figsize=(16, 16))
            ax1.plot(aco.best_path_history)
            ax1.set_title("Convergence Curve")
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Best Path Length")
            ax1.grid(True)
            st.pyplot(fig1)

            # 显示结果
            st.write(f"运行时间: {run_time:.2f} 秒")
            st.write(f"最优路径长度: {aco.best_distance:.2f}")

        with col2:
            # 绘制最优路径
            fig2, ax2 = plt.subplots(figsize=(16, 16))
            coords = np.array(map_data)

            # 绘制所有城市点
            ax2.scatter(coords[:, 0], coords[:, 1], c="red", s=50)

            # 绘制路径
            for i in range(len(aco.best_path)):
                j = (i + 1) % len(aco.best_path)
                city1 = aco.best_path[i]
                city2 = aco.best_path[j]
                ax2.plot(
                    [coords[city1, 0], coords[city2, 0]],
                    [coords[city1, 1], coords[city2, 1]],
                    "b-",
                    alpha=0.5,
                )

            ax2.set_title("Best Path Visualization")
            # 隐藏坐标轴
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.grid(True)
            st.pyplot(fig2)
