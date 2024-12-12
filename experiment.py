import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from ant_colony import AntColonyTSP
from map_data import distance_matrix


def run_experiment(params):
    """运行单次实验"""
    start_time = time.time()

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

    aco.solve()
    run_time = time.time() - start_time

    return {"运行时间": run_time, "最优路径长度": aco.best_distance}


def experiment_ant_count():
    """实验1：探究蚂蚁数量的影响"""
    results = []
    ant_counts = [10, 20, 50, 100, 200, 500, 1000]

    base_params = {
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.1,
        "q": 1.0,
        "max_iterations": 100,
        "min_pheromone": 0.1,
        "max_pheromone": 50.0,
    }

    print("实验1：探究蚂蚁数量的影响")
    for n_ants in tqdm(ant_counts):
        params = base_params.copy()
        params["n_ants"] = n_ants

        # 每组参数运行3次取平均
        experiment_results = []
        for _ in range(3):
            result = run_experiment(params)
            experiment_results.append(result)

        avg_result = {
            "蚂蚁数量": n_ants,
            "平均运行时间": np.mean([r["运行时间"] for r in experiment_results]),
            "平均最优路径长度": np.mean(
                [r["最优路径长度"] for r in experiment_results]
            ),
        }
        results.append(avg_result)

    return pd.DataFrame(results)


def experiment_iterations():
    """实验2：探究迭代次数的影响"""
    results = []
    iterations = [10, 20, 50, 100, 200, 500, 1000]

    base_params = {
        "n_ants": 50,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.1,
        "q": 1.0,
        "min_pheromone": 0.1,
        "max_pheromone": 50.0,
    }

    print("\n实验2：探究迭代次数的影响")
    for max_iter in tqdm(iterations):
        params = base_params.copy()
        params["max_iterations"] = max_iter

        # 每组参数运行3次取平均
        experiment_results = []
        for _ in range(3):
            result = run_experiment(params)
            experiment_results.append(result)

        avg_result = {
            "迭代次数": max_iter,
            "平均运行时间": np.mean([r["运行时间"] for r in experiment_results]),
            "平均最优路径长度": np.mean(
                [r["最优路径长度"] for r in experiment_results]
            ),
        }
        results.append(avg_result)

    return pd.DataFrame(results)


def experiment_parameters():
    """实验3：探究关键参数(alpha, beta, rho)的影响"""
    results = []

    base_params = {
        "n_ants": 50,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.1,
        "q": 1.0,
        "max_iterations": 100,
        "min_pheromone": 0.1,
        "max_pheromone": 50.0,
    }

    # 参数范围
    param_ranges = {
        "alpha": [0.5, 1.0, 2.0, 3.0],
        "beta": [1.0, 2.0, 3.0, 4.0],
        "rho": [0.05, 0.1, 0.2, 0.3],
    }

    print("\n实验3：探究关键参数的影响")
    for param_name, param_values in param_ranges.items():
        print(f"\n测试参数: {param_name}")
        for value in tqdm(param_values):
            params = base_params.copy()
            params[param_name] = value

            # 每组参数运行3次取平均
            experiment_results = []
            for _ in range(3):
                result = run_experiment(params)
                experiment_results.append(result)

            avg_result = {
                "参数名称": param_name,
                "参数值": value,
                "平均运行时间": np.mean([r["运行时间"] for r in experiment_results]),
                "平均最优路径长度": np.mean(
                    [r["最优路径长度"] for r in experiment_results]
                ),
            }
            results.append(avg_result)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # 运行实验1：蚂蚁数量的影响
    df_ants = experiment_ant_count()
    df_ants.to_csv("experiment_ant_count.csv", index=False)
    print("\n蚂蚁数量实验结果：")
    print(df_ants)

    # 运行实验2：迭代次数的影响
    df_iterations = experiment_iterations()
    df_iterations.to_csv("experiment_iterations.csv", index=False)
    print("\n迭代次数实验结果：")
    print(df_iterations)

    # 运行实验3：关键参数的影响
    df_params = experiment_parameters()
    df_params.to_csv("experiment_parameters.csv", index=False)
    print("\n参数影响实验结果：")
    print(df_params)
