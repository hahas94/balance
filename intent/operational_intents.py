"""
File description:
-----------------
This file implements operational intents class. An intent is a drone operation
with source and destination nodes and departure time.
"""

from typing import List, Tuple, Union

import graph


class Intent:
    """Implements an Operational intent object, which represents a drone operation."""
    def __init__(self, source: graph.Node, destination: graph.Node, start: int) -> None:
        """
        Creates an instance of an Intent object.

        Args:
            source: graph.Node
                Source node.
            destination: graph.Node
                Destination node.
            start: int
                Departure time for drone.
        """
        self._source = source
        self._destination = destination
        self._start = start
        self._path: List[Tuple[str, int, int]] = []  # the path between `source` and `destination`.
        self._actual_time = 0
        self._ideal_time = 0
        self._solution_found = False

    def __repr__(self):
        return f"Intent(source={self._source.name}, destination={self._destination.name}, start={self._start})"

    def __str__(self):
        return f"Intent {(self._source.name, self._destination.name)} starts at {self._start}"

    @property
    def source(self) -> graph.Node:
        return self._source

    @property
    def destination(self) -> graph.Node:
        return self._destination

    @property
    def start(self) -> int:
        return self._start

    @start.setter
    def start(self, s) -> None:
        self._start = s

    @property
    def path(self) -> List[Tuple[str, int, int]]:
        return self._path

    @property
    def time_difference(self) -> Union[int, None]:
        return self._actual_time - self._ideal_time if (self._ideal_time > 0 and self._actual_time > 0) else None

    @property
    def actual_time(self) -> int:
        return self._actual_time

    @actual_time.setter
    def actual_time(self, time: int) -> None:
        self._actual_time = time

    @property
    def ideal_time(self) -> int:
        return self._ideal_time

    @ideal_time.setter
    def ideal_time(self, time: int) -> None:
        self._ideal_time = time

    @property
    def solution_found(self) -> bool:
        return self._solution_found

    def build_path(self, goal_node: Union[None, graph.ExtendedNode]) -> None:
        """Given a goal node, it backtracks on it to create a path from start to destination."""
        while goal_node is not None:
            self._path.insert(0, (goal_node.original.name, goal_node.layer, goal_node.travel_time))
            goal_node = goal_node.previous
        self._solution_found = True

    def solution(self) -> None:
        """
        Prints a string representation of the operation path and times.
        A path may not exist for a number of reasons, such as planning tie
        beyond time horizon, vertiports being all reserved, or no real path exist.
        """
        if self._solution_found:
            solution = "".join([f"[node:{el[0]}, layer:{el[1]}, time:{self._start+el[2]}]" +
                                (" --> " if index < len(self._path)-1 else "") for index, el in enumerate(self._path)])
            print(f"\t{solution}")
            print(f"\tideal time:{self.ideal_time}, actual time:{self.actual_time}, "
                  f"time difference:{self.time_difference}\n\n")
        else:
            print(f"\tNo solution is possible for this operational intent.\n\n")


# =============================================== END OF FILE ===============================================
