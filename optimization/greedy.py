"""
File description:
-----------------
This file implements the original Dijkstra's algorithm as well as a specialized version of it.
It is used to find the shortest paths for operational intents.
"""

from typing import Dict, Union
import queue

import graph
import intent


def find_shortest_path(operation_intent: intent.Intent, nodes_dict: Dict[str, graph.Node]) -> None:
    """
    Runs the original Dijkstra's algorithm to find the shortest path.
    The graph is built successively.

    Args:
        operation_intent: intent.Intent
            An operational intent.
        nodes_dict: Dict[str, graph.Node]
            Dictionary of all node objects and their names as keys.

    Returns:

    """
    source_node = operation_intent.source
    destination_node = operation_intent.destination
    ideal_time: Union[float, int] = 0

    distances: Dict[str, Union[float, int]] = {node.name: float('inf') for node in nodes_dict.values()}
    distances[source_node.name] = 0

    priority_queue: queue.PriorityQueue = queue.PriorityQueue()
    priority_queue.put((0, source_node))

    while not priority_queue.empty():
        dist_u, u = priority_queue.get()
        if u == destination_node:
            ideal_time = distances[u.name]
            break
        if dist_u > distances[u.name]:
            continue
        for edge in u.outgoing_edges:
            v, weight = edge.destination, edge.weight
            path_distance = dist_u + weight
            if path_distance < distances[v.name]:
                distances[v.name] = path_distance
                priority_queue.put((path_distance, v))

    if isinstance(ideal_time, int):
        operation_intent.ideal_time = ideal_time

    return None


def find_shortest_path_extended(operation_intent: intent.Intent, delta: int, nodes_dict: Dict[str, graph.Node]) \
        -> Union[None, graph.ExtendedNode]:
    """
    Runs the specialized Dijkstra's algorithm to find the shortest
    path in an extended network. The extended graph is built successively.

    Args:
        operation_intent: intent.Intent
            An operational intent.
        delta: int
            Time delta
        nodes_dict: Dict[str, graph.Node]
            Dictionary of node objects and their names as keys.

    Returns:
        destination_extended: Union[None, graph.ExtendedNode]
            The goal node in the extended graph or None if no path found.

    """
    source_node = operation_intent.source
    destination_node = operation_intent.destination
    start_time = operation_intent.start
    time_uncertainty = operation_intent.time_uncertainty

    # determine start layer
    k, r = divmod(start_time, delta)
    layer = k + (r > 0)
    operation_intent.start = delta * layer

    # create an extended node for the source node and add it to the queue
    source_node_extended = graph.ExtendedNode(source_node.name, layer, None, 0, 0, 0, -1)
    destination_extended: Union[None, graph.ExtendedNode] = None

    unvisited_queue: queue.PriorityQueue = queue.PriorityQueue()
    distances: Dict[str, Union[float, int]] = {source_node_extended.name: 0}
    unvisited_queue.put((source_node_extended.travel_time, source_node_extended))

    # this constant can be used to make nodes representing ground delay to be of less priority than other nodes.
    # this is a temporary hack until it has to be changed into something more robust.
    GROUND_DELAY_PRIORITY_DECREMENTOR = 1e6

    while not unvisited_queue.empty():
        current_dist, current_node = unvisited_queue.get()
        current_node_original = nodes_dict[current_node.name_original]

        if current_node_original == source_node and current_dist >= GROUND_DELAY_PRIORITY_DECREMENTOR:
            current_dist -= GROUND_DELAY_PRIORITY_DECREMENTOR

        current_layer = current_node.layer

        # goal check
        if current_node_original == destination_node:
            destination_extended = current_node
            break

        # if node already exist with better path, then skip iteration
        if current_dist > distances[current_node.name]:
            continue

        index = -1  # used for insertion_order
        for index, edge in enumerate(current_node_original.outgoing_edges):
            v: graph.Node = edge.destination
            k, r = divmod(edge.weight, delta)
            k1, r1 = divmod(edge.weight + time_uncertainty, delta)
            n_deltas = k + (r > 0)
            n_deltas_uncertainty = k1 + (r1 > 0)
            new_weight = n_deltas * delta
            v_travel_time = current_dist + new_weight
            v_extended_name = f"{v.name}_{current_layer + n_deltas}"

            # checking whether this neighbor has been explored previously
            try:
                distances[v_extended_name]
            except KeyError as _:
                distances[v_extended_name] = float('inf')

            v_has_capacity = v.has_capacity(current_layer + 1, current_layer + n_deltas_uncertainty + 1)
            shorter_path_found = v_travel_time < distances[v_extended_name]

            if shorter_path_found and v_has_capacity:
                distances[v_extended_name] = v_travel_time
                v_extended = graph.ExtendedNode(name=v.name, layer=current_layer+n_deltas, previous=current_node,
                                                travel_time=v_travel_time, left_reserve=current_layer+1,
                                                right_reserve=current_layer+n_deltas_uncertainty,
                                                insertion_order=index)

                unvisited_queue.put((v_travel_time, v_extended))

        # add the node itself to the queue to indicate the drone
        # can stay where it is and wait, if it is the departure node
        ground_delay_possible = current_node_original == source_node and current_layer+1 < source_node.num_layers
        if ground_delay_possible:
            current_itself = graph.ExtendedNode(name=current_node.name_original, layer=current_layer+1,
                                                previous=current_node, travel_time=current_node.travel_time+delta,
                                                left_reserve=current_layer+1, right_reserve=current_layer+1,
                                                insertion_order=index+1)

            distances[current_itself.name] = current_itself.travel_time
            unvisited_queue.put((current_itself.travel_time + GROUND_DELAY_PRIORITY_DECREMENTOR, current_itself))

    if destination_extended:
        operation_intent.actual_greedy_time = destination_extended.travel_time
        operation_intent.build_greedy_path(destination_extended)

    return destination_extended

# =============================================== END OF FILE ===============================================
