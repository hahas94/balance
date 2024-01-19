"""
File description:
-----------------
This file is used to read a problem, build a queue of operational intents,
use both the greedy approach and integer programming approach to solve the
routing problem and save the results.

To run this file, define the path to an example json file first, then run. Example:
    `example_path = "./examples/example1.json"`
"""

import json
import math
from typing import List, Dict, Sequence, Tuple, Union
import time
import datetime

import checks
import graph
import intent
import optimization
import utils


class InvalidSolutionError(Exception):
    """A custom exception class to be raised when either of the two methods produce invalid solutions."""
    def __init__(self, message=None):
        super().__init__(message)
        self.message = message


def read_example(path: str) -> Tuple[int, int, int, List[Dict], List[Dict], List[Dict]]:
    """
    Opens an example file, reads it and returns its content. Program exits if json data format is invalid.
    It will throw assertion errors if the actual data is of incorrect type or value.

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


def solve_greedy(operational_intent: intent.Intent, time_delta: int,
                 nodes: Sequence[graph.Node]) -> Tuple[Union[int, None], Union[graph.ExtendedNode, None]]:
    """
    Given data related to an operational intent, it runs the greedy algorithms on that intent,
    and prints the path found for it, if successfull.

    Parameters
    ----------
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
    goal_node = optimization.find_shortest_path_extended(operational_intent, time_delta, nodes)
    optimization.find_shortest_path(operational_intent, nodes)

    return operational_intent.greedy_time_difference, goal_node


def solve_ip(nodes: dict, edges: dict, intents: dict, time_steps: range, time_delta: int) -> Union[float, None]:
    """
    This function calls the ip optimization model to find a schedule for all the intents.

    Parameters
    ----------
    nodes: dict
        Dictionary of the nodes in a graph and their names.
    edges: dict
        Dictionary of the edges in a graph and their names.
    intents: dict
        Dictionary of the operational intents to be scheduled and their names.
    time_steps: range
        A range object for the discretized time steps.
    time_delta: int
        Time discretization step.

    Returns
    -------
    ip_obj: Union[float, None]
        The objective of the model or none if no solution was found.

    """
    ip_obj = optimization.ip_optimization(nodes, edges, intents, time_steps, time_delta)
    return ip_obj


def print_solutions(intents: dict) -> None:
    """
    Printing the solutions for each intent, in both greedy and ip parts.
    Also prints the operational time information.
    Parameters
    ----------
    intents: dict
        Dictionary of operational intents.

    Returns
    -------
        None
    """
    for name, operational_intent in intents.items():
        print(f"Intent {name}:")
        operational_intent.solution()


def adjust_capacities(goal_node: graph.ExtendedNode, nodes_dict: Dict[str, graph.Node]) -> None:
    """
    Given a dictionary of original graph nodes, it adjusts their capacities
    to include the capacities of the latest planned intent.

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


def increment_reservations(time_uncertainty: int, prev_intent, nodes_dict: dict,  delta: int,
                           indices: Sequence = None) -> None:
    """
    Increments reservations of the vertiports of a scheduled operational intent
    to mitigate the uncertainty of an operation after it in the queue.

    Args:
        time_uncertainty: int
            Time uncertainty of an operation down in the intents queue.
        prev_intent: intents.Intent
            A scheduled operational intent.
        nodes_dict: dict
            Dictionary of nodes objects and their names.
        delta: int
            Time discretization delta
        indices: Sequence (optinal)
            A sequence of indices used for looping over a path.

    Returns:

    """
    decrementor = int(math.copysign(1, time_uncertainty))  # sign of time_uncertainty
    path = prev_intent.path_greedy
    indices = indices if indices else range(1, len(path))

    for index in indices:
        name = path[index].name
        prev_name = path[index-1].name

        if name != prev_name:
            previous_layer, start_time = path[index-1].layer, path[index-1].layer*delta
            new_left_layer = (start_time - decrementor*time_uncertainty + 1) // delta
            l, r = max(1, new_left_layer), previous_layer+1
            # if previously set let layer, adjust it
            if prev_intent.path_greedy[index].probably_left_reserved_layer:
                prev_intent.path_greedy[index].probably_left_reserved_layer += decrementor*time_uncertainty
            else:
                prev_intent.path_greedy[index].probably_left_reserved_layer = new_left_layer
            nodes_dict[name].layer_capacities[l:r] = [cap-decrementor for cap in nodes_dict[name].layer_capacities[l:r]]

    return None


def decrement_reservations(prev_intent: intent.Intent, curr_intent: intent.Intent, nodes_dict: dict, delta: int) \
        -> None:
    """
    Undoing `increment_reservations(...)` for the vertiports not being common of
    the two intents or that their uncertainty buffers don't intersect.

    Args:
        prev_intent: intent.Intent
            Previously scheduled intent
        curr_intent: intent.Intent
            An intent being scheduled currently.
        nodes_dict: dict
            Dictionary of nodes objects and their names.
        delta: int
            Time discretization delta

    Returns:
    """
    # find common vertiports, based on name as well as time intersection
    common_nodes = []
    u_prev, u_curr = [prev_intent.time_uncertainty, curr_intent.time_uncertainty]
    prev_intent_path: List[utils.Link] = prev_intent.path_greedy
    curr_intent_path: List[utils.Link] = curr_intent.path_greedy

    # find if the two intents have a common vertiport and their uncertainty buffers intersect at that vertiport
    for ind_p, prev_node in enumerate(prev_intent_path[1:]):
        for ind_c, curr_node in enumerate(curr_intent_path[1:]):
            if prev_node.name == curr_node.name:
                prev_intent_reaches_before = prev_node.right_reserved_layer <= curr_intent_path[ind_c].layer
                prev_intent_starts_after = prev_intent_path[ind_p].layer >= curr_node.right_reserved_layer
                if not (prev_intent_reaches_before or prev_intent_starts_after):
                    common_nodes.append(prev_node)
                break

    # for each vertiport in prev_intent_path, if it is not a common vertiport, decrement its cap
    indices = [i for i in range(1, len(prev_intent_path)) if prev_intent_path[i] not in common_nodes]
    increment_reservations(-u_curr, prev_intent, nodes_dict, delta, indices)

    return None


def uncertainty_reservation_handling(res_type: str, curr_intent_name: str, curr_intent: intent.Intent,
                                     nodes_dict: dict, intents_dict: dict, time_delta: int) -> None:
    """
    When planning an intent, previously scheduled intents must be safe from this current intent being delayed,
    hence the vertiports along all the paths of the previous intents are reserved for longer time.

    When the current intent is scheduled, then all the vertiports that were reserved for longer time but
    that are not included in the path of this current intent are being freed from that extra reservation.

    Depending on those two opposing cases, this function calls other functions to do the job.

    Args:
        res_type: str
            Either 'increment' or 'decrement', indicating whether a vertiport is reserved or freed at a layer.
        curr_intent_name: str
            Name of operational intent
        curr_intent: intent.Intent
            The operational intent
        nodes_dict: dict
            Dictionary of nodes objects and their names.
        intents_dict: dict
            Dictionary of intent objects and their names.
        time_delta: int
            Time discretization delta

    Returns:

    """
    curr_u = curr_intent.time_uncertainty

    for prev_intent_name, prev_intent in intents_dict.items():
        if prev_intent_name == curr_intent_name:
            break
        # p_path = prev_intent.path_greedy
        if res_type == 'increment':
            increment_reservations(curr_u, prev_intent, nodes_dict, time_delta)
        elif res_type == 'decrement':
            decrement_reservations(prev_intent, curr_intent, nodes_dict, time_delta)
    return None


def main(nodes_dict: dict, edges_dict: dict, intents_dict: dict, time_delta: int, time_steps: range,
         verbose: bool = True) -> Tuple[Union[int, None], Union[float, None]]:
    """
    The main function that solves each operational intent in sequence,
    as well as solving them all at once.

    Args:
        nodes_dict: dict
            Dictionary of nodes objects and their names.
        edges_dict: dict
            Dictionary of edges objects and their names.
        intents_dict: dict
            Dictionary of intent objects and their names.
        time_delta: int
            Time discretization delta
        time_steps: range
            A range object for the discretized time steps.
        verbose: bool
            Whether to print solutions or not.

    Returns:
        greedy_obj: int
            The greedy objective
        ip_obj: int
            The ip optimization objective

    """
    greedy_obj: Union[int, None] = 0

    start_ip = time.perf_counter()
    ip_obj = solve_ip(nodes_dict, edges_dict, intents_dict, time_steps, time_delta)
    end_ip = time.perf_counter()
    ip_solution_time = str(datetime.timedelta(seconds=round(end_ip-start_ip, 2)))

    # in case the ip model is unsolvable for this instance due to short time horizon, quit and rerun with longer time.
    if ip_obj is None:
        return greedy_obj, ip_obj

    print(f"Integer Programming runtime: {ip_solution_time}")  # remove when finished with debugging solving times

    start_greedy = time.perf_counter()
    for intent_name, operation_intent in intents_dict.items():
        # for each previous drone, get path, update vertiport capacities
        uncertainty_reservation_handling('increment', intent_name, operation_intent, nodes_dict, intents_dict,
                                         time_delta)

        # solve intent
        time_difference, goal_node = solve_greedy(operation_intent, time_delta, list(nodes_dict.values()))

        if goal_node and time_difference is not None:
            adjust_capacities(goal_node, nodes_dict)
            greedy_obj += time_difference
        else:
            # no solution found, so exit
            return ip_obj, None

        # for each previous drone, get path, update vertiport capacities
        uncertainty_reservation_handling('decrement', intent_name, operation_intent, nodes_dict, intents_dict,
                                         time_delta)
    end_greedy = time.perf_counter()
    greedy_solution_time = str(datetime.timedelta(seconds=round(end_greedy-start_greedy, 2)))

    time_horizon = time_steps[-1]
    valid_solutions = checks.sanity_check(intents_dict, nodes_dict, edges_dict, time_delta, time_horizon)

    if not valid_solutions:
        raise InvalidSolutionError("Invalid solutions!")

    if verbose:
        print_solutions(intents_dict)
    print(f"Integer Programming runtime: {ip_solution_time}\nGreedy runtime: {greedy_solution_time}")

    sum_ideal_times = sum(op_intent.ideal_time for op_intent in intents_dict.values())
    ip_obj -= sum_ideal_times
    ip_obj = round(ip_obj, 1)

    return greedy_obj, ip_obj


if __name__ == "__main__":
    example_path = "./examples/test4.json"
    ip_objective = None
    greedy_objective = None
    time_horizon_extender = 1

    while ip_objective is None or greedy_objective is None:
        global_start, global_time_horizon, global_time_delta, global_nodes, global_edges, global_intents = \
            read_example(path=example_path)

        global_time_horizon *= time_horizon_extender

        global_nodes_dict, global_edges_dict, global_intents_dict = \
            create_dicts(global_nodes, global_edges, global_intents, global_time_horizon, global_time_delta)

        global_time_steps = range(global_start, global_time_horizon + 1, global_time_delta)

        greedy_objective, ip_objective = main(global_nodes_dict, global_edges_dict, global_intents_dict,
                                              global_time_delta, global_time_steps)
        time_horizon_extender += 1

    print(f"Greedy objective: {greedy_objective}\nInteger programming objective: {ip_objective}")

# =============================================== END OF FILE ===============================================
