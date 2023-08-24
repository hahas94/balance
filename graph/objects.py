"""
File description:
-----------------
This file implements objects found in a graph, such as nodes and edges.
Additionally, it implements an object called ExtendedNode, representing
a node object in a time-extended network.
"""

from __future__ import annotations  # Needed for forward declarations

from typing import List, Dict, Union


class Node:
    """Implements a regular node object."""
    def __init__(self, name: str, capacity: int, num_layers: int) -> None:
        """
        Creates an instance of a Node object.

        Args:
            name: str
                Node name.
            capacity: int:
                Capacity of node.
            num_layers: int
                Number of layers during time horizon.
        """
        self._name = name
        self._capacity = capacity
        self._num_layers = num_layers + 1  # element at index 0 is never used.
        self._outgoing_edges: List[Edge] = []
        self._layer_capacities: List[int] = [capacity for _ in range(self._num_layers)]

    def __repr__(self):
        return f"Node(name={self._name}, capacity={self._capacity})"

    def __str__(self):
        return f"Node {self._name} has capacity {self._capacity}"

    def __eq__(self, other):
        """Two nodes are identical if their names are the same."""
        return self._name == other.name

    def __lt__(self, other):
        """
        Always returns true. It exists so that two nodes can
        be compared in a `queue.PriorityQueue` object.
        """
        return self._name <= other.name

    @property
    def name(self) -> str:
        return self._name

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def outgoing_edges(self) -> List[Edge]:
        return self._outgoing_edges

    @property
    def layer_capacities(self) -> List[int]:
        return self._layer_capacities

    @layer_capacities.setter
    def layer_capacities(self, capacities: List[int]) -> None:
        self._layer_capacities = capacities

    def add_outgoing_edge(self, o_edge: Edge) -> None:
        """
        Appends an outgoing edge of this node to a list of such edges.

        Args:
            o_edge: Edge
                An outgoing edge from this node.

        Returns:

        """
        if isinstance(o_edge, Edge):
            self._outgoing_edges.append(o_edge)
        else:
            raise ValueError("Instance must be of type Edge.")
        return


class ExtendedNode:
    """Implements an ExtendedNode object."""
    def __init__(self, name: str, layer: int, previous: Union[ExtendedNode, None], original: Node, travel_time: int) \
            -> None:
        """
        Creates an instance of ExtendedNode object.
        Args:
            name: str
                Name of original node this extended node represents.
            layer: int
                The layer of the extended network this node exists at.
            previous: Union[ExtendedNode, None]
                The previous ExtendedNode in the graph from which a drone travelled towards this node.
                If this is the start node, then no previous node can exist hence `None`.
            original: Node
                The original node this extended node represents.
            travel_time: int
                Time it took to reach this node from start.
        """
        self._name = name + f"{layer}"  # node name is `original_name`+`Layer`
        self._layer = layer
        self._previous = previous
        self._original = original
        self._capacities_dict: Dict[str, List[int]] = {}  # a dictionary of all original node's capacities at `layer`.
        self._travel_time = travel_time
        self._insertion_order: int = 0  # node will be added to a `queue.PriorityQueue` in an order.

    def __repr__(self):
        return (f"ExtendedNode(name={self._name}, layer={self._layer}, previous={self._previous}, " 
                f"original={self._original}, travel_time={self._travel_time})")

    def __str__(self):
        return f"Extended Node {self._name} at layer {self._layer} has travel time {self._travel_time}"

    def __lt__(self, other):
        """Comparing two nodes based on their travel time firstly, and then based on their insertion order."""
        if self._travel_time == other.travel_time:
            return self._insertion_order < other.insertion_order
        return self._travel_time < other.travel_time

    @property
    def name(self) -> str:
        return self._name

    @property
    def previous(self) -> Union[ExtendedNode, None]:
        return self._previous

    @property
    def original(self) -> Node:
        return self._original

    @property
    def travel_time(self) -> int:
        return self._travel_time

    @property
    def layer(self) -> int:
        return self._layer

    @property
    def insertion_order(self) -> int:
        return self._insertion_order

    @insertion_order.setter
    def insertion_order(self, x: int) -> None:
        self._insertion_order = x

    @property
    def capacities(self) -> Dict[str, List[int]]:
        return self._capacities_dict

    @capacities.setter
    def capacities(self, capacities: Dict[str, List[int]]) -> None:
        self._capacities_dict = capacities

    def has_capacity(self, name: str, start: int, stop: int) -> bool:
        """
        Checks whether a node has capacity at all layers between `start` and `stop` layers.
        If a layer is outside the possible layers, then it is False.

        Args:
            name: str
                Original node name.
            start: int
                Start layer.
            stop: int
                Stop layer (exclusive)

        Returns:
            return value: bool
                Node has necessary capacity (True) or not (False).
        """
        bools_list = []
        for layer in range(start, stop):
            try:
                bools_list.append(self._capacities_dict[name][layer] > 0)
            except IndexError as _:
                bools_list.append(False)
        return False not in bools_list

    def decrement_capacity(self, name: str, start: int, stop: int) -> None:
        """
        Decrements the capacity of several layers of the original node by 1.

        Args:
            name: str
                Original node name.
            start: int
                Start layer.
            stop: int
                Stop layer (exclusive)

        Returns:

        """
        for layer in range(start, stop):
            self._capacities_dict[name][layer] -= 1


class Edge:
    """Implements an Edge object."""
    def __init__(self, source: Node, destination: Node, weight: int) -> None:
        """
        Creates an instance of an Edge object.

        Args:
            source: Node
                Source node.
            destination: Node
                Destination node
            weight: int
                Time it takes to go from `source` to `destination`.
        """
        self._source = source
        self._destination = destination
        self._weight = weight

        source.add_outgoing_edge(self)

    def __repr__(self):
        return f"Edge(source={self._source.name}, destination={self._destination.name}, weight={self._weight})"

    def __str__(self):
        return f"Edge {(self._source.name, self._destination.name)} with weight {self._weight}"

    @property
    def source(self) -> Node:
        return self._source

    @property
    def destination(self) -> Node:
        return self._destination

    @property
    def weight(self) -> int:
        return self._weight

# =============================================== END OF FILE ===============================================
