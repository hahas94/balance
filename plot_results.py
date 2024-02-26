import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--curr_dir_name', required=True, type=str,
                    help="Base name of directory where results are saved. "
                         "in this case `curr_dir_name` is `stockholm_medium.`")

def plot_results(dataframe, name):
    """Creating plots of the data in the merged dataframes and saving them to disk."""
    # plt.style.use("seaborn")

    greedy_color, ip_color = 'mediumaquamarine', 'violet'
    data = [[dataframe.greedy_obj/60, dataframe.ip_obj/60],
            [dataframe.greedy_runtime, dataframe.ip_runtime],
            [dataframe.greedy_memory_usage, dataframe.ip_memory_usage]]
    x_label = 'Number of intents'
    y_labels = ['Total travel time extension (min)', 'Runtime (min)', 'Memory usage (GB)']
    titles = ['', 'Runtimes of the methods', 'Memory usages of the methods']
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

    dir_name = args.curr_dir_name

    df = pd.read_pickle(f"./results/{dir_name}/data.pickle")

    plot_results(df, dir_name)