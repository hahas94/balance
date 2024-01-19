"""
File description:
-----------------
This file is used to measure performance of the ip model across a suit of examples.
Examples are create at random but with a seed for reproducibility.
"""
import itertools

import numpy as np
import networkx as nx

import main


def create_intent():
    """
    Populates a random intent. If no path between source and
    destination exists, the function calls itself recursively.

    Returns
        intent: dict
            The generated intent
    -------

    """
    random_index = np.random.randint(low=0, high=len(global_possible_intents))
    source, destination = global_possible_intents[random_index]
    source, destination = source['name'], destination['name']
    U = np.random.randint(low=0, high=6)
    start = np.random.randint(low=0, high=min(20, global_time_horizon))

    intent = {"source": source, "destination": destination, "start": start, "uncertainty": U}

    return intent


if __name__ == "__main__":
    seed = 2718
    np.random.seed(seed=seed)

    example_path = "./examples/test_performance.json"
    num_examples = 5
    intents = []

    global_start, global_time_horizon, global_time_delta, global_nodes, global_edges, _ = \
        main.read_example(path=example_path)

    graph_edges = [(edge["source"], edge["destination"]) for edge in global_edges]
    global_graph = nx.DiGraph(graph_edges)
    global_possible_intents = [(s, t) for s, t in itertools.product(global_nodes, global_nodes)
                               if s['name'] != t['name'] and nx.has_path(global_graph, s['name'], t['name'])]

    for example in range(1, num_examples+1):
        n_intents = 2*example
        intents = [create_intent() for _ in range(n_intents)]

        ip_objective = None
        greedy_objective = None
        time_horizon_extender = 1

        while ip_objective is None or greedy_objective is None:
            global_intents = intents

            global_time_horizon *= time_horizon_extender

            global_nodes_dict, global_edges_dict, global_intents_dict = \
                main.create_dicts(global_nodes, global_edges, global_intents, global_time_horizon, global_time_delta)

            global_time_steps = range(global_start, global_time_horizon + 1, global_time_delta)

            greedy_objective, ip_objective = main.main(global_nodes_dict, global_edges_dict, global_intents_dict,
                                                       global_time_delta, global_time_steps, False)
            time_horizon_extender += 1

        print(f"\nExample: {example}\nIntents: {intents}")
        print(f"Greedy objective: {greedy_objective}\nInteger programming objective: {ip_objective}")
        print(f"{40*'-'}\n")


# =============================================== END OF FILE ===============================================
