"""
File description:
-----------------
This file implements operational intents class. An intent is a drone operation
with source and destination nodes and departure time.
"""

from typing import List, Tuple, Union

import graph

import utils


class Intent:
    """Implements an Operational intent object, which represents a drone operation."""
    def __init__(self, source: graph.Node, destination: graph.Node, start: int, uncertainty: int) -> None:
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
        self._time_uncertainty = uncertainty
        # A link in the path between `source` and `destination`. Each link has the following instance variables:
        # name, layer, travel_time, left_most_reserved_layer, right_most_reserved_layer
        self._path_greedy: List[utils.Link] = []
        self._path_ip: List[utils.Link] = []
        self._actual_greedy_time = 0
        self._actual_ip_time = 0
        self._ideal_time = 0
        self._greedy_solution_found = False
        self._ip_solution_found = False

    def __repr__(self):
        return (f"Intent(source={self._source.name}, destination={self._destination.name}, "
                f"start={self._start}, uncertainty={self.time_uncertainty})")

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
    def time_uncertainty(self) -> int:
        return self._time_uncertainty

    @property
    def path_greedy(self) -> List[Tuple[str, int, int, int]]:
        return self._path_greedy

    @property
    def path_ip(self) -> List[Tuple[str, int, int, int]]:
        return self._path_ip

    @property
    def greedy_time_difference(self) -> Union[int, None]:
        td = self._actual_greedy_time - self._ideal_time \
            if (self._ideal_time > 0 and self._actual_greedy_time > 0) else None
        return td

    @property
    def ip_time_difference(self) -> Union[int, None]:
        td = self._actual_ip_time - self._ideal_time if (self._ideal_time > 0 and self._actual_ip_time > 0) else None
        return td

    @property
    def actual_greedy_time(self) -> int:
        return self._actual_greedy_time

    @actual_greedy_time.setter
    def actual_greedy_time(self, time: int) -> None:
        self._actual_greedy_time = time

    @property
    def actual_ip_time(self) -> int:
        return self._actual_ip_time

    @actual_ip_time.setter
    def actual_ip_time(self, time: int) -> None:
        self._actual_ip_time = time

    @property
    def ideal_time(self) -> int:
        return self._ideal_time

    @ideal_time.setter
    def ideal_time(self, time: int) -> None:
        self._ideal_time = time

    @property
    def greedy_solution_found(self) -> bool:
        return self._greedy_solution_found

    @property
    def ip_solution_found(self) -> bool:
        return self._ip_solution_found

    def build_greedy_path(self, goal_node: Union[None, graph.ExtendedNode]) -> None:
        """Given a goal node, it backtracks on it to create a path from start to destination."""
        while goal_node is not None:
            link = utils.Link(name=goal_node.name_original, layer=goal_node.layer, travel_time=goal_node.travel_time,
                              left_reserved_layer=goal_node.left_reserve, right_reserved_layer=goal_node.right_reserve)
            self._path_greedy.insert(0, link)
            goal_node = goal_node.previous
        self._greedy_solution_found = True

    def build_ip_path(self, path: list) -> None:
        """
        Assigns the ip-found path for the intent to the relevant instance variable.

        Parameters
        ----------
        path: list
            A list of tuples for each link in the path.

        Returns
        -------
            None

        """
        self._path_ip = path
        self._ip_solution_found = True

    def solution(self) -> None:
        """
        Prints a string representation of the operation paths and times.
        A path may not exist for a number of reasons, such as planning tie
        beyond time horizon, vertiports being all reserved, or no real path exist.

        The greedy and ip solutions are printed one fter another.
        """
        if self._greedy_solution_found:
            solution_greedy = "".join([f"[node:{link.name}, layer:{link.layer}, "
                                       f"travel_time:{self._start+link.travel_time}]" +
                                       (" --> " if index < len(self._path_greedy) - 1 else "")
                                       for index, link in enumerate(self._path_greedy)])
            print(f"\t{solution_greedy}")
        else:
            print(f"\tNo solution is possible for this operational intent.")

        if self._ip_solution_found:
            solution_ip = "".join([f"[node:{link.name}, layer:{link.layer}, "
                                   f"travel_time:{self._start + link.travel_time}]" +
                                   (" --> " if index < len(self._path_ip) - 1 else "")
                                   for index, link in enumerate(self._path_ip)])
            print(f"\t{solution_ip}")
        else:
            print(f"\tNo solution is possible for this operational intent.")

        print(f"\tideal time:{self.ideal_time}, "
              f"actual greedy time:{self.actual_greedy_time}, "
              f"actual ip time: {self._actual_ip_time}, "
              f"greedy time difference:{self.greedy_time_difference}, "
              f"ip time difference:{self.ip_time_difference}\n\n")


class IntentsCollection:
    """A class representing a collection of intents. It is useful for storing a list of intents in pandas dataframe."""
    __slots__ = ['intents_list']

    def __init__(self, intents_list: List[Intent]):
        """
        Initialize an instance.
        Parameters
        ----------
        intents_list: List[Intent]
            A list of intent instances.
        """
        self.intents_list = intents_list

    def __repr__(self):
        return f"IntentsCollection(intents_list={self.intents_list})"

    def __str__(self):
        return f"IntentsCollection(length={len(self.intents_list)})"
# =============================================== END OF FILE ===============================================
