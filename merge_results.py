"""
File description:
-----------------
This file is used to merge results from subresult folders into a single folder.
"""

import argparse
import os

import pandas as pd
from plot_results import plot_results

parser = argparse.ArgumentParser()
parser.add_argument('--curr_dir_name', required=True, type=str,
                    help="Base name of directory where subresults are saved. "
                         "A subresult folder may have the name `stockholm_medium_1`, "
                         "in this case `curr_dir_name` is `stockholm_medium.`")
parser.add_argument('--merged_dir_name', required=True, type=str,
                    help="Name of directory where merged results are saved. "
                         "Example `stockholm_medium_merged.`")
parser.add_argument('--start', required=True, type=int,
                    help="Subresults directory extension start.")
parser.add_argument('--stop', required=True, type=int,
                    help="Subresults directory extension stop.")
parser.add_argument('--step', required=True, type=int,
                    help="Subresults directory extension step for going from start to stop with that step.")


def merge_logs(curr_dir_name, merged_dir_name, start, stop, step):
    """Merging the logs from multiple log files into one unified log file."""
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
    """Merging multiple pandas.DataFrames belonging to different runs into one dataframe and writing it to disk."""
    all_dfs = [pd.read_pickle(f"./results/{curr_dir_name}_{file_ext}/data.pickle") for
               file_ext in range(start, stop + 1, step)]

    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df.to_pickle(f"./results/{merged_dir_name}/data.pickle")
    print("Merging of dataframes done.")

    return merged_df


if __name__ == "__main__":
    args = parser.parse_args()

    dir_name, merged_name, starting_from, stop_at, step_with = (
        args.curr_dir_name, args.merged_dir_name, int(args.start), int(args.stop), int(args.step))

    merge_logs(dir_name, merged_name, starting_from, stop_at, step_with)

    merged_dataframe = merge_dfs(dir_name, merged_name, starting_from, stop_at, step_with)

    plot_results(merged_dataframe, merged_name)

# ================= END OF FILE =================

