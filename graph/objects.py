"""
File description:
-----------------
This file implements objects found in a graph, such as nodes and edges.
Additionally, it implements an object called ExtendedNode, representing
a node object in a time-extended network.
"""

from __future__ import annotations  # Needed for forward declarations

from typing import List, Union
import numpy as np


class Node:
    """Implements a regular node object, which represents a vertiport."""
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
        self._layer_capacities: np.ndarray[np.intc] = capacity * np.ones(self._num_layers, dtype=np.intc)

    def __repr__(self):
        return f"Node(name={self._name}, capacity={self._capacity}, num_layers={self._num_layers})"

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
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def outgoing_edges(self) -> List[Edge]:
        return self._outgoing_edges

    @property
    def layer_capacities(self) -> np.ndarray[np.intc]:
        return self._layer_capacities

    @layer_capacities.setter
    def layer_capacities(self, capacities: np.ndarray[np.intc]) -> None:
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
            raise TypeError("Instance must be of type Edge.")
        return

    def has_capacity(self, start: int, stop: int) -> bool:
        """
        Checks whether the node has capacity at all layers between `start` and `stop` layers.
        If a layer is outside the possible layers, then it is False.

        Args:
            start: int
                Start layer.
            stop: int
                Stop layer (exclusive)

        Returns:
            return value: bool
                Node has necessary capacity (True) or not (False).
        """
        if start < 0 or stop > len(self._layer_capacities):
            return False
        return np.all(self._layer_capacities[start:stop] > 0)

    def decrement_capacity(self, start: int, stop: int) -> None:
        """
        Decrements the capacity of several layers of the node by 1.

        Args:
            start: int
                Start layer.
            stop: int
                Stop layer

        Returns:

        """
        self._layer_capacities[start:stop+1] -= 1


class ExtendedNode:
    """Implements an object which represents a node in a specific layer in the time extended graph."""
    def __init__(self, name: str, layer: int, previous: Union[ExtendedNode, None], travel_time: int, left_reserve: int,
                 right_reserve: int, insertion_order: int) -> None:
        """
        Creates an instance of ExtendedNode object.
        Args:
            name: str
                Name of original node this extended node represents.
            layer: int
                The layer of the extended network this node exists at.
            previous: Union[ExtendedNode, None]
                The previous ExtendedNode in the graph leading to this node.
                If this is the start node, then no previous node can exist hence `None`.
            travel_time: int
                Time it took to reach this node from source node.
            left_reserve: int
                The left most layer to reserve in the original node.
            right_reserve: int
                The right most layer to reserve in the original node.
            insertion_order: int
                The order at which the node is added to a priority queue.

        """
        self._name = name + f"_{layer}"  # node name is `original_name`+`Layer`
        self._name_original = name  # name of the original node this extended node represents.
        self._layer = layer
        self._previous = previous
        self._travel_time = travel_time
        self._left_reserve = left_reserve
        self._right_reserve = right_reserve
        self._insertion_order = insertion_order

    def __repr__(self):
        return (f"ExtendedNode(name={self._name}, layer={self._layer}, previous={self._previous}, " 
                f"travel_time={self._travel_time}, left_reserve={self._left_reserve}, "
                f"right_reserve={self._right_reserve}, insertion_order={self._insertion_order})")

    def __str__(self):
        return f"Extended Node {self._name} has travel time {self._travel_time}"

    def __lt__(self, other):
        """Comparing two nodes based on their travel time firstly, and then based on their insertion order."""
        if self._travel_time == other.travel_time:
            return self._insertion_order < other.insertion_order
        return self._travel_time < other.travel_time

    @property
    def name(self) -> str:
        return self._name

    @property
    def name_original(self) -> str:
        return self._name_original

    @property
    def previous(self) -> Union[ExtendedNode, None]:
        return self._previous

    @property
    def travel_time(self) -> int:
        return self._travel_time

    @property
    def layer(self) -> int:
        return self._layer

    @property
    def left_reserve(self) -> int:
        return self._left_reserve

    @property
    def right_reserve(self) -> int:
        return self._right_reserve

    @property
    def insertion_order(self) -> int:
        return self._insertion_order


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
