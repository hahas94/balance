"""
File description:
-----------------
This file is used to merge results from subresult forlders into a unified folder.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--curr_dir_name', required=True, type=str,
                    help="Base name of directory where subresults are saved.")
parser.add_argument('--merged_dir_name', required=True, type=str,
                    help="Name of directory where merged results are saved.")
parser.add_argument('--start', required=True, type=int,
                    help="Subresults dir extension start.")
parser.add_argument('--stop', required=True, type=int,
                    help="Subresults dir extension stop.")
parser.add_argument('--step', required=True, type=int,
                    help="Subresults dir extension step for going from start to stop with that step.")


def merge_logs(curr_dir_name, merged_dir_name, start, stop, step):
    if not os.path.exists(f'./results/{merged_dir_name}'):
        os.makedirs(f'./results/{merged_dir_name}')
        os.makedirs(f'./results/{merged_dir_name}/models')
        with open(f"./results/{merged_dir_name}/logs.txt", 'w') as _:
            pass
    for file_ext in range(start, stop + 1, step):
        with open(f"./results/{curr_dir_name}_{file_ext}/logs.txt", 'r') as log:
            content = log.read()

        with open(f"./results/{merged_dir_name}/logs.txt", 'a') as merged_log:
            merged_log.write(content)
    print('Logging done.')


def merge_dfs(curr_dir_name, merged_dir_name, start, stop, step):
    all_dfs = [pd.read_pickle(f"./results/{curr_dir_name}_{file_ext}/data.pickle") 
            for file_ext in range(start, stop + 1, step)]

    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df.to_pickle(f"./results/{merged_dir_name}/data.pickle")
    print("Merging of dataframes done.")

    return merged_df


def plot_merged_results(dataframe, name):
    plt.style.use("seaborn")

    greedy_color, ip_color = 'mediumaquamarine', 'violet'
    data = [[dataframe.greedy_obj, dataframe.ip_obj],
            [dataframe.greedy_runtime, dataframe.ip_runtime],
            [dataframe.greedy_memory_usage, dataframe.ip_memory_usage]]
    x_label = 'Number of intents'
    y_labels = ['Cost', 'Runtime (min)', 'Memory usage (GB)']
    titles = ['Objectives of the methods', 'Runtimes of the methods', 'Memory usages of the methods']
    plot_names = ['objective', 'runtime', 'memory_usage']
    x_axis = dataframe.num_intents.values
    fontsize = 16
    linewidth = 3

    for indx, elem in enumerate(data):
        mask0, mask1 = np.isfinite(elem[0]), np.isfinite(elem[1])
        fig, ax = plt.subplots(1, 1)
        ax.plot(x_axis[mask0], elem[0][mask0], marker='o', ls='-',
                linewidth=linewidth, color=greedy_color, label='Greedy')

        ax.plot(x_axis[mask1], elem[1][mask1], marker='o', ls='-',
                linewidth=linewidth, color=ip_color, label='IP')

        ax.set_xlabel(x_label, family='serif', fontsize=fontsize)
        ax.set_ylabel(y_labels[indx], family='serif', fontsize=fontsize)
        ax.set_title(titles[indx], family='serif', fontsize=fontsize)
        ax.set_xticks(x_axis)

        ax.legend()

        fig.savefig(f'./results/{name}/{plot_names[indx]}.png')
        print("Plotting done.")

    return


if __name__ == "__main__":
    args = parser.parse_args()
    dir_name, merged_name, starting_from, stop_at, step_with = (
        args.curr_dir_name, args.merged_dir_name, int(args.start), int(args.stop), int(args.step))
    merge_logs(dir_name, merged_name, starting_from, stop_at, step_with)
    merged_dataframe = merge_dfs(dir_name, merged_name, starting_from, stop_at, step_with)
    plot_merged_results(merged_dataframe, merged_name)

# ================= END OF FILE =================

