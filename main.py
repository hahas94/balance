"""
File description:
-----------------
This file is used to read a problem, build a queue of operational intents,
use both the greedy approach and integer programming approach to solve the
routing problem and save the results.

To run this file, define the path to an example json file first, then run. Example:
    `example_path = "./examples/test1.json"`
"""

import json
from typing import List, Dict, Sequence, Tuple, Union

import graph
import intent
import optimization


def read_example(path: str) -> Tuple[int, int, int, List[Dict], List[Dict], List[Dict]]:
    """
    Opens an example file, reads and returns its content. Program exists if json data is invalid.
    Parameters
    ----------
    path: str
        Path to an example json file. Ex. 'ex1.json'.

    Returns
    -------
    return value: Tuple[int, int, int, List[Dict], List[Dict], List[Dict]]
        A tuple (start_time, time_horizon, time_delta, nodes_list, edges_list, intents_list)

    """
    with open(path, 'r') as f:
        content = f.read()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON example file at {path}:", e)
        exit()
    else:
        start = data["start"]
        assert start >= 0 and isinstance(start, int), "Start time is either negative or not an integer."

        time_horizon = data["time_horizon"]
        assert time_horizon > 0 and isinstance(time_horizon, int), "Time horizon must be a positive integer."

        time_delta = data["time_delta"]
        assert time_delta > 0 and isinstance(time_delta, int) and time_horizon % time_delta == 0, \
            ("Time delta must be a positive integer and a multiple of it equal the time horizon, "
             "i.e. T = k*delta for an integer k.")

        k, r = divmod(time_horizon, time_delta)
        assert isinstance(k, int) and r == 0, "Time horizon must be an integer multiple of time delta."

        nodes = data["nodes"]
        for node in nodes:
            assert isinstance(node['name'], str) and isinstance(node['capacity'], int) and node['capacity'] >= 0, (
                f"{node}: Either syntax error in name, capacity being non integer or negative.")

        edges = data["edges"]
        for edge in edges:
            assert isinstance(edge['source'], str) and isinstance(edge['destination'], str) and \
                   isinstance(edge['weight'], int) and edge['weight'] >= 0, (
                f"{edge}: Either syntax error in edge names, weight being non integer or negative.")

        intents = data["intents"]
        for op_intent in intents:
            assert isinstance(op_intent['source'], str) and isinstance(op_intent['destination'], str) \
                   and isinstance(op_intent['start'], int) and op_intent['start'] >= start \
                   and isinstance(op_intent['uncertainty'], int) and op_intent['uncertainty'] >= 0, (
                f"{op_intent}: Either syntax error in names, start time being non integer, it starts before "
                f"operations start time or time uncertainty is negative/non-integer.")

        return start, time_horizon, time_delta, nodes, edges, intents


def create_dicts(nodes: List[Dict], edges: List[Dict], intents: List[Dict], time_horizon: int, time_delta: int) \
        -> Tuple[Dict, Dict, Dict]:
    """
    Creates dictionaries of intents, nodes, edges given lists of these.
    Parameters
    ----------
    nodes: List[Dict]
        A list of nodes, each node being a dict.
    edges: List[Dict]
        A list of edges, each edge being a dict.
    intents: List[Dict]
        A list of intents, each intent being a dict.
    time_horizon: int
        Planning time horizon.
    time_delta: int
        Time delta.

    Returns
    -------
    return value: Tuple[Dict, Dict, Dict]
        Three dictionaries (nodes_dict, edges_dict, intents_dict).

    """
    num_layers = time_horizon // time_delta

    nodes_dict = {v["name"]: graph.Node(v["name"], v["capacity"], num_layers) for v in nodes}
    edges_dict = {
        (e["source"], e["destination"]):
            graph.Edge(nodes_dict[e["source"]], nodes_dict[e["destination"]], e["weight"]) for e in edges
    }
    intents_dict = {
        (i["source"], i["destination"], i["start"]):
            intent.Intent(nodes_dict[i["source"]], nodes_dict[i["destination"]], i["start"], i["uncertainty"])
        for i in intents
    }

    return nodes_dict, edges_dict, intents_dict


def solve_intent(name: str, operational_intent: intent.Intent, time_delta: int,
                 nodes: Sequence[graph.Node]) -> Tuple[Union[int, None], Union[graph.ExtendedNode, None]]:
    """
    Given data related to an operational intent, it runs the greedy algorithms on that intent.

    Parameters
    ----------
    name: str
        Operational intent name
    operational_intent: intent.Intent
        An intent object.
    time_delta: int
        Time delta
    nodes: Sequence[graph.Node]
        A list of nodes objects.

    Returns
    -------
    time_difference: Tuple[Union[int, None]
        The difference in time between actual and ideal planning times.
    goal_node: Union[graph.ExtendedNode, None]]
        An extended node representing the destination node (if reached).

    """
    goal_node = optimization.dijkstra_extended(operation_intent, time_delta, nodes)
    optimization.dijkstra_original(operation_intent, nodes)

    print(f"Intent {name}:")
    operation_intent.solution()

    return operational_intent.time_difference, goal_node


def adjust_capacities(goal_node: graph.ExtendedNode, nodes_dict: Dict[str, graph.Node]) -> None:
    """
    Given a dictionary of original graph nodes, it adjusts their capacities
    to reflect to the latest planned intent.

    Args:
        goal_node: graph.ExtendedNode
            The destination node in the extended graph.
        nodes_dict: Dict[str, graph.Node]
            Dictionary of the original graphs nodes.

    Returns:
        None
    """
    for node, layer_cap in goal_node.capacities.items():
        nodes_dict[node].layer_capacities = layer_cap


def increment_reservations(time_uncertainty, path, nodes_dict, delta):
    decrementor = time_uncertainty // -time_uncertainty  # either -1 or 1
    for index, el in enumerate(path[1:]):
        name, _, _ = el
        _, previous_layer, start_time = path[index]
        new_left_layer = (start_time - time_uncertainty) // delta
        l, r = max(0, new_left_layer), previous_layer+1
        nodes_dict[name].layer_capacities[l:r] = [cap+decrementor for cap in nodes_dict[name].layer_capacities[l:r]]

    return None


def decrement_reservations(time_uncertainties, prev_intent_path, curr_intent_path, nodes_dict, delta):
    # find common vertiports, based on name as well as time intersection
    common_nodes = []

    u_prev, u_curr = time_uncertainties
    # for finding tu right layer
    pk, pr = divmod(u_prev, delta)
    ck, cr = divmod(u_curr, delta)
    prev_right_layer = pk + (pr > 0)
    curr_right_layer = ck + (cr > 0)

    for ind_pel, pel in enumerate(prev_intent_path[1:]):
        for ind_cel, cel in enumerate(curr_intent_path[1:]):
            if pel[0] == cel[0]:
                prev_intent_layer_right = pel[1] + prev_right_layer
                curr_intent_layer_right = cel[1] + curr_right_layer

                if not ((prev_intent_layer_right < curr_intent_path[ind_cel][1]) or
                        (prev_intent_path[ind_cel][1] > curr_intent_layer_right)):
                    common_nodes.append(pel)

    # for each vertiport in prev_intent_path, if it is not a common vertiport, decrement its cap
    prev_intent_path = [el for el in prev_intent_path if el not in common_nodes]
    increment_reservations(-u_curr, prev_intent_path, nodes_dict, delta)

    return None


if __name__ == "__main__":
    example_path = "./examples/test12.json"

    global_start, global_time_horizon, global_time_delta, global_nodes, global_edges, global_intents = \
        read_example(path=example_path)

    global_nodes_dict, global_edges_dict, global_intents_dict = \
        create_dicts(global_nodes, global_edges, global_intents, global_time_horizon, global_time_delta)

    greedy_objective: int = 0

    for intent_name, operation_intent in global_intents_dict.items():
        uncertainty = operation_intent.time_uncertainty
        # for each previous drone, get path, update vertiport capacities
        for prev_intent_name, prev_intent in global_intents_dict.items():
            if prev_intent_name == intent_name:
                break
            intent_path = prev_intent.path
            increment_reservations(uncertainty, intent_path, global_nodes_dict, global_time_delta)

        global_time_difference, global_goal_node = \
            solve_intent(intent_name, operation_intent, global_time_delta, list(global_nodes_dict.values()))

        if global_goal_node and global_time_difference:
            adjust_capacities(global_goal_node, global_nodes_dict)
            greedy_objective += global_time_difference

        # for each previous drone, get path, update vertiport capacities
        # for prev_intent_name, prev_intent in global_intents_dict.items():
        #     if prev_intent_name == intent_name:
        #         break
        #     intent_path = prev_intent.path
        #     decrement_reservations([prev_intent.time_uncertainty, uncertainty], intent_path,
        #                            operation_intent.path, global_nodes_dict, global_time_delta)

    print(f"Greedy objective:{greedy_objective}")

# =============================================== END OF FILE ===============================================
