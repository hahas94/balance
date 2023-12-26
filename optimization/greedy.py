"""
File description:
-----------------
This file implements the original Dijkstra's algorithm as well as a specialized version of it.
It is used to find the shortest paths for operational intents.
"""

from typing import Dict, Sequence, Union
import copy
import queue

import graph
import intent


def find_shortest_path(operation_intent: intent.Intent, nodes: Sequence[graph.Node]) -> None:
    """
    Runs the original Dijkstra's algorithm to find the shortest path.
    The graph is built successively.

    Args:
        operation_intent: intent.Intent
            An operational intent.
        nodes: Sequence[graph.Node]
            A list of nodes objects.

    Returns:

    """
    source_node = operation_intent.source
    destination_node = operation_intent.destination
    ideal_time: Union[float, int] = 0

    distances: Dict[str, Union[float, int]] = {node.name: float('inf') for node in nodes}
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


def create_extended_node(name: str, layer: int, previous: graph.ExtendedNode, original: graph.Node,
                         travel_time: int, insertion_order: int, start: int, stop: int, n_deltas_uncertainty) \
        -> graph.ExtendedNode:
    """
    Helper function that creates and returns an ExtendedNode and modifies some of its variables.

    Args:
        name: str
            Original node name.
        layer: int
            Layer in extended network.
        previous: graph.ExtendedNode
            Node that leads to the current node.
        original: graph.Node
            Original node being represented by this node.
        travel_time: int
            Time it takes to get to this node from start.
        insertion_order: int
            The order this node is put into the queue.
        start: int
            Layer from which the drone departs.
        stop: int
            Layer (exclusive) at which the drone reaches this node.
        n_deltas_uncertainty: int
            Travel time plus time uncertainty converted to time deltas.

    Returns:
        extended: graph.ExtendedNode
            The newly created node.
    """
    extended = graph.ExtendedNode(name, layer, previous, original, travel_time)
    extended.insertion_order = insertion_order
    extended.uncertainty_layer = previous.layer + n_deltas_uncertainty
    extended.capacities = copy.deepcopy(previous.capacities)
    extended.decrement_capacity(name, start, stop)

    return extended


def find_shortest_path_extended(operation_intent: intent.Intent, delta: int, nodes: Sequence[graph.Node]) \
        -> Union[None, graph.ExtendedNode]:
    """
    Runs the specialized Dijkstra's algorithm to find the shortest
    path in an extended network. The extended graph is built successively.

    Args:
        operation_intent: intent.Intent
            An operational intent.
        delta: int
            Time delta
        nodes: Sequence[graph.Node]
            A list of nodes objects.

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

    unvisited_queue: queue.PriorityQueue = queue.PriorityQueue()

    # create an extended node for the source node and add it to the queue
    start_node_extended = graph.ExtendedNode(source_node.name, layer, None, source_node, 0)
    start_node_extended.capacities = {node.name: node.layer_capacities for node in nodes}
    unvisited_queue.put((start_node_extended.travel_time, start_node_extended))

    destination_extended: Union[None, graph.ExtendedNode] = None
    distances: Dict[str, Union[float, int]] = {start_node_extended.name: 0}

    while not unvisited_queue.empty():
        current_dist, current_node = unvisited_queue.get()
        curr_layer = current_node.layer

        # goal check
        if current_node.original == destination_node:
            destination_extended = current_node
            break

        # if node already exist with better path, then skip iteration
        if current_dist > distances[current_node.name]:
            continue

        index = -1  # used for insertion_order
        for index, edge in enumerate(current_node.original.outgoing_edges):
            v: graph.Node = edge.destination
            k, r = divmod(edge.weight, delta)
            k1, r1 = divmod(edge.weight + time_uncertainty, delta)
            n_deltas = k + (r > 0)
            n_deltas_uncertainty = k1 + (r1 > 0)
            new_weight = n_deltas * delta
            v_travel_time = current_dist + new_weight
            v_extended_name = v.name + str(current_node.layer + n_deltas)

            # checking whether this neighbor has been explored previously
            try:
                distances[v_extended_name]
            except KeyError as _:
                distances[v_extended_name] = float('inf')

            v_has_capacity = current_node.has_capacity(v.name, curr_layer + 1, curr_layer + n_deltas_uncertainty + 1)
            shorter_path_found = v_travel_time < distances[v_extended_name]

            if shorter_path_found and v_has_capacity:
                distances[v_extended_name] = v_travel_time
                v_extended = create_extended_node(v.name, curr_layer + n_deltas, current_node, v, v_travel_time, index,
                                                  curr_layer + 1, curr_layer + n_deltas_uncertainty + 1,
                                                  n_deltas_uncertainty)
                unvisited_queue.put((v_travel_time, v_extended))

        # add the node itself to the queue to indicate the drone
        # can stay where it is and wait, if it is the departure node
        ground_delay_possible = (current_node.original == source_node) and \
                                (current_node.has_capacity(current_node.original.name, curr_layer + 1, curr_layer + 2))
        if ground_delay_possible:
            current_itself = create_extended_node(current_node.original.name, curr_layer + 1, current_node,
                                                  current_node.original, current_node.travel_time + delta,
                                                  index + 1, curr_layer + 1, curr_layer + 2, 1)
            distances[current_itself.name] = current_itself.travel_time
            unvisited_queue.put((current_itself.travel_time, current_itself))

    if destination_extended:
        operation_intent.actual_greedy_time = destination_extended.travel_time
        operation_intent.build_greedy_path(destination_extended)

    return destination_extended

# =============================================== END OF FILE ===============================================
