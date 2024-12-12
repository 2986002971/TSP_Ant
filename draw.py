import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 使用黑体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def plot_ant_count_results():
    """绘制蚂蚁数量实验结果"""
    data = pd.read_csv("experiment_ant_count.csv")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 绘制平均运行时间
    ax1.plot(
        data["蚂蚁数量"],
        data["平均运行时间"],
        marker="o",
        label="平均运行时间",
        color="b",
    )
    ax1.set_xlabel("蚂蚁数量")
    ax1.set_ylabel("平均运行时间", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    # 创建第二个纵轴
    ax2 = ax1.twinx()
    ax2.plot(
        data["蚂蚁数量"],
        data["平均最优路径长度"],
        marker="x",
        label="平均最优路径长度",
        color="r",
    )
    ax2.set_ylabel("平均最优路径长度", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    # 添加标题
    plt.title("蚂蚁数量与运行时间和路径长度的关系")
    fig.tight_layout()
    plt.show()


def plot_iteration_results():
    """绘制迭代次数实验结果"""
    data = pd.read_csv("experiment_iterations.csv")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 绘制平均运行时间
    ax1.plot(
        data["迭代次数"],
        data["平均运行时间"],
        marker="o",
        label="平均运行时间",
        color="b",
    )
    ax1.set_xlabel("迭代次数")
    ax1.set_ylabel("平均运行时间", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    # 创建第二个纵轴
    ax2 = ax1.twinx()
    ax2.plot(
        data["迭代次数"],
        data["平均最优路径长度"],
        marker="x",
        label="平均最优路径长度",
        color="r",
    )
    ax2.set_ylabel("平均最优路径长度", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    # 添加标题
    plt.title("迭代次数与运行时间和路径长度的关系")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_ant_count_results()
    plot_iteration_results()
