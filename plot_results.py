"""
This file contains functionality for plotting different aspects of the results.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--curr_dir_name', required=True, type=str,
                    help="Base name of directory where subresults are saved. "
                         "A subresult folder may have the name `stockholm_medium_1`, "
                         "in this case `curr_dir_name` is `stockholm_medium.`")


def plot_results(dataframe, name):
    """Creating plots of the data in the merged dataframes and saving them to disk."""

    ideal_times = [[intent.ideal_time for intent in dataframe.intents_collection[i].intents_list]
                   for i in range(len(dataframe))]

    greedy_delays = [[intent.greedy_time_difference for intent in dataframe.intents_collection[i].intents_list]
                     for i in range(len(dataframe))]
    ip_delays = [[intent.ip_time_difference for intent in dataframe.intents_collection[i].intents_list]
                 for i in range(len(dataframe))]

    objectives = [dataframe.greedy_obj/60, dataframe.ip_obj/60]
    run_times = [dataframe.greedy_runtime, dataframe.ip_runtime]
    memory_usages = [dataframe.greedy_memory_usage, dataframe.ip_memory_usage]

    maximum_delayed_drones_delay = [[round(max(delays) / 60, 1) for delays in greedy_delays],
                                    [round(max(delays) / 60, 1) for delays in ip_delays]]

    average_travel_time_extension = [
        [(sum(delays) / sum(ideal_times[idx])) / 60 for idx, delays in enumerate(greedy_delays)],
        [(sum(delays) / sum(ideal_times[idx])) / 60 for idx, delays in enumerate(ip_delays)]
    ]

    max_travel_time_extension_percentage = 80
    greedy_travel_time_extension_percentages = [[round(100 * td / ideal_times[idx][indx], 1)
                                                 for indx, td in enumerate(delays)]
                                                for idx, delays in enumerate(greedy_delays)]
    greedy_percentage_of_drones_with_exceeded_time_extension = [
        sum([percentage > max_travel_time_extension_percentage for percentage in percentages])
        for percentages in greedy_travel_time_extension_percentages]

    ip_travel_time_extension_percentages = [[round(100 * td / ideal_times[idx][indx], 1)
                                             for indx, td in enumerate(delays)]
                                            for idx, delays in enumerate(ip_delays)]
    ip_percentage_of_drones_with_exceeded_time_extension = [
        sum([percentage > max_travel_time_extension_percentage for percentage in percentages])
        for percentages in ip_travel_time_extension_percentages]

    percentage_of_drones_with_exceeded_time_extension = [greedy_percentage_of_drones_with_exceeded_time_extension,
                                                         ip_percentage_of_drones_with_exceeded_time_extension]
    # print(percentage_of_drones_with_exceeded_time_extension)

    data = [objectives,
            run_times,
            memory_usages,
            maximum_delayed_drones_delay,
            average_travel_time_extension,
            percentage_of_drones_with_exceeded_time_extension
            ]

    greedy_color, ip_color = 'mediumaquamarine', 'violet'
    x_label = 'Number of intents'
    y_labels = ['Total travel time extension (min)', 'Runtime (min)', 'Memory usage (GB)',
                'Delay of most delayed drone (min)', 'Average travel time extension (min)',
                f'Percentage of drones with a travel time extension more than {max_travel_time_extension_percentage}%']
    titles = ['', 'Runtimes of the methods', 'Memory usages of the methods', '', '', '']
    plot_names = ['objective', 'runtime', 'memory_usage', 'delay', 'average_time_extension',
                  'percentage_time_extension']
    x_axis = dataframe.num_intents.values
    fontsize = 12
    linewidth = 3

    for indx, elem in enumerate(data):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(x_axis, elem[0], marker='o', ls='-', linewidth=linewidth, color=greedy_color, label='Greedy')
        ax.plot(x_axis, elem[1], marker='o', ls='-', linewidth=linewidth, color=ip_color, label='IP')

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

# ================= END OF FILE =================
