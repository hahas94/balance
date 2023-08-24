"""
File description:
-----------------
Testing functionality of the two implemented versions of the Dijkstra's algorithm.
To run all tests, execute this command from project root:
    `python3 -m unittest discover -s tests`
"""

import unittest

import main
import optimization


class Example1(unittest.TestCase):
    """A small graph of two edges and one intent."""
    def setUp(self) -> None:
        self.time_horizon = 30
        self.time_delta = 10
        nodes = [
            {"name": "v0", "capacity": 2},
            {"name": "v1", "capacity": 2},
            {"name": "v2", "capacity": 2}
        ]

        edges = [
            {"source": "v0", "destination": "v1", "weight": 5},
            {"source": "v1", "destination": "v2", "weight": 6}
        ]

        intents = [
            {"source": "v0", "destination": "v2", "start": 0}
        ]

        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents, self.time_horizon,
                                                                 self.time_delta)

        self.ideal_times = [11]
        self.actual_times = [20]
        self.destination_layers = [2]

    def test_dijkstra(self):
        index = 0

        intent_name, operation_intent = list(self.intents.items())[index]
        nodes_list = list(self.nodes.values())
        goal_node = optimization.dijkstra_extended(operation_intent, self.time_delta, nodes_list)
        optimization.dijkstra_original(operation_intent, nodes_list)

        # assert goal node is reached
        self.assertEqual(goal_node.original.name, operation_intent.destination.name)

        # assert goal node is reached at correct layer
        self.assertEqual(goal_node.layer, self.destination_layers[index])

        # assert actual time is correct
        self.assertEqual(operation_intent.actual_time, self.actual_times[index])

        # assert ideal time is correct
        self.assertEqual(operation_intent.ideal_time, self.ideal_times[index])


class Example2(unittest.TestCase):
    """A small graph of two edges and one intent, where layer jump is needed."""
    def setUp(self) -> None:
        self.time_horizon = 20
        self.time_delta = 5
        nodes = [
            {"name": "v0", "capacity": 2},
            {"name": "v1", "capacity": 2},
            {"name": "v2", "capacity": 2}
        ]

        edges = [
            {"source":  "v0", "destination":  "v1", "weight": 4},
            {"source":  "v1", "destination":  "v2", "weight": 6}
        ]

        intents = [
            {"source":  "v0", "destination":  "v2", "start": 0}
        ]

        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents, self.time_horizon,
                                                                 self.time_delta)

        self.ideal_times = [10]
        self.actual_times = [15]
        self.destination_layers = [3]

    def test_dijkstra(self):
        index = 0

        intent_name, operation_intent = list(self.intents.items())[index]
        nodes_list = list(self.nodes.values())
        goal_node = optimization.dijkstra_extended(operation_intent, self.time_delta, nodes_list)
        optimization.dijkstra_original(operation_intent, nodes_list)

        # assert goal node is reached
        self.assertEqual(goal_node.original.name, operation_intent.destination.name)

        # assert goal node is reached at correct layer
        self.assertEqual(goal_node.layer, self.destination_layers[index])

        # assert actual time is correct
        self.assertEqual(operation_intent.actual_time, self.actual_times[index])

        # assert ideal time is correct
        self.assertEqual(operation_intent.ideal_time, self.ideal_times[index])


class Example3(unittest.TestCase):
    """A small graph where there is no path between `src` and `dest`, hence no solution."""
    def setUp(self) -> None:
        self.time_horizon = 20
        self.time_delta = 5
        nodes = [
            {"name": "v0", "capacity": 2},
            {"name": "v1", "capacity": 2},
            {"name": "v2", "capacity": 2}
        ]

        edges = [
            {"source":  "v0", "destination":  "v1", "weight": 4}
        ]

        intents = [
            {"source":  "v0", "destination":  "v2", "start": 0}
        ]

        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents, self.time_horizon,
                                                                 self.time_delta)

    def test_dijkstra(self):
        index = 0

        intent_name, operation_intent = list(self.intents.items())[index]
        nodes_list = list(self.nodes.values())
        goal_node = optimization.dijkstra_extended(operation_intent, self.time_delta, nodes_list)
        optimization.dijkstra_original(operation_intent, nodes_list)

        # assert goal node is reached
        self.assertIs(goal_node, None)
        self.assertEqual(operation_intent.actual_time, 0)
        self.assertEqual(operation_intent.ideal_time, 0)


class Example4(unittest.TestCase):
    """A small graph where intent travel time is beyond time horizon, hence no solution."""
    def setUp(self) -> None:
        self.time_horizon = 10
        self.time_delta = 5
        nodes = [
            {"name": "v0", "capacity": 2},
            {"name": "v1", "capacity": 2},
            {"name": "v2", "capacity": 2}
        ]

        edges = [
            {"source":  "v0", "destination":  "v1", "weight": 4},
            {"source": "v1", "destination": "v2", "weight": 6}
        ]

        intents = [
            {"source":  "v0", "destination":  "v2", "start": 0}
        ]

        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents, self.time_horizon,
                                                                 self.time_delta)

    def test_dijkstra(self):
        index = 0

        intent_name, operation_intent = list(self.intents.items())[index]
        nodes_list = list(self.nodes.values())
        goal_node = optimization.dijkstra_extended(operation_intent, self.time_delta, nodes_list)
        optimization.dijkstra_original(operation_intent, nodes_list)

        # assert goal node is reached
        self.assertIs(goal_node, None)
        self.assertEqual(operation_intent.actual_time, 0)
        self.assertEqual(operation_intent.ideal_time, 10)


class Example5(unittest.TestCase):
    """A graph with two identical paths between `src` and `dest`."""
    def setUp(self) -> None:
        self.time_horizon = 10
        self.time_delta = 5
        nodes = [
            {"name": "v0", "capacity": 2},
            {"name": "v1", "capacity": 2},
            {"name": "v2", "capacity": 2},
            {"name": "v3", "capacity": 2}
        ]

        edges = [
            {"source": "v0", "destination": "v1", "weight": 4},
            {"source": "v0", "destination": "v2", "weight": 4},
            {"source": "v1", "destination": "v3", "weight": 5},
            {"source": "v2", "destination": "v3", "weight": 5}
        ]

        intents = [
            {"source":  "v0", "destination":  "v3", "start": 0}
        ]

        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents, self.time_horizon,
                                                                 self.time_delta)

        self.ideal_times = [9]
        self.actual_times = [10]
        self.destination_layers = [2]

    def test_dijkstra(self):
        index = 0

        intent_name, operation_intent = list(self.intents.items())[index]
        nodes_list = list(self.nodes.values())
        goal_node = optimization.dijkstra_extended(operation_intent, self.time_delta, nodes_list)
        optimization.dijkstra_original(operation_intent, nodes_list)

        # assert goal node is reached
        self.assertEqual(goal_node.original.name, operation_intent.destination.name)

        # assert goal node is reached at correct layer
        self.assertEqual(goal_node.layer, self.destination_layers[index])

        # assert actual time is correct
        self.assertEqual(operation_intent.actual_time, self.actual_times[index])

        # assert ideal time is correct
        self.assertEqual(operation_intent.ideal_time, self.ideal_times[index])


class Example6(unittest.TestCase):
    """start time for the intent is at first layer instead of time 0."""
    def setUp(self) -> None:
        self.time_horizon = 15
        self.time_delta = 5
        nodes = [
            {"name": "v0", "capacity": 2},
            {"name": "v1", "capacity": 2},
            {"name": "v2", "capacity": 2}
        ]

        edges = [
            {"source": "v0", "destination": "v1", "weight": 4},
            {"source": "v1", "destination": "v2", "weight": 4}
        ]

        intents = [
            {"source": "v0", "destination": "v2", "start": 3}
        ]

        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents, self.time_horizon,
                                                                 self.time_delta)

        self.ideal_times = [8]
        self.actual_times = [10]
        self.destination_layers = [3]

    def test_dijkstra(self):
        index = 0

        intent_name, operation_intent = list(self.intents.items())[index]
        nodes_list = list(self.nodes.values())
        goal_node = optimization.dijkstra_extended(operation_intent, self.time_delta, nodes_list)
        optimization.dijkstra_original(operation_intent, nodes_list)

        # assert goal node is reached
        self.assertEqual(goal_node.original.name, operation_intent.destination.name)

        # assert goal node is reached at correct layer
        self.assertEqual(goal_node.layer, self.destination_layers[index])

        # assert actual time is correct
        self.assertEqual(operation_intent.actual_time, self.actual_times[index])

        # assert ideal time is correct
        self.assertEqual(operation_intent.ideal_time, self.ideal_times[index])


class Example7(unittest.TestCase):
    """Test contains two intents starting at different nodes."""
    def setUp(self) -> None:
        self.time_horizon = 20
        self.time_delta = 5
        nodes = [
            {"name": "v0", "capacity": 2},
            {"name": "v1", "capacity": 2},
            {"name": "v2", "capacity": 2},
            {"name": "v3", "capacity": 2},
            {"name": "v4", "capacity": 2}
        ]

        edges = [
            {"source": "v0", "destination": "v2", "weight": 4},
            {"source": "v1", "destination": "v3", "weight": 3},
            {"source": "v2", "destination": "v4", "weight": 3},
            {"source": "v3", "destination": "v4", "weight": 4}
        ]

        intents = [
            {"source": "v0", "destination": "v4", "start": 0},
            {"source": "v1", "destination": "v4", "start": 0}
        ]

        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents, self.time_horizon,
                                                                 self.time_delta)

        self.ideal_times = [7, 7]
        self.actual_times = [10, 10]
        self.destination_layers = [2, 2]

    def test_dijkstra(self):
        index = 0

        for intent_name, operation_intent in self.intents.items():
            nodes_list = list(self.nodes.values())
            goal_node = optimization.dijkstra_extended(operation_intent, self.time_delta, nodes_list)
            optimization.dijkstra_original(operation_intent, nodes_list)

            # assert goal node is reached
            self.assertEqual(goal_node.original.name, operation_intent.destination.name)

            # assert goal node is reached at correct layer
            self.assertEqual(goal_node.layer, self.destination_layers[index])

            # assert actual time is correct
            self.assertEqual(operation_intent.actual_time, self.actual_times[index])

            # assert ideal time is correct
            self.assertEqual(operation_intent.ideal_time, self.ideal_times[index])

            index += 1


class Example8(unittest.TestCase):
    """Test contains two intents where the second one has to wait at `src` due to limited capacity at `dest`."""
    def setUp(self) -> None:
        self.time_horizon = 20
        self.time_delta = 5
        nodes = [
            {"name": "v0", "capacity": 2},
            {"name": "v1", "capacity": 2},
            {"name": "v2", "capacity": 2},
            {"name": "v3", "capacity": 2},
            {"name": "v4", "capacity": 1}
        ]

        edges = [
            {"source": "v0", "destination": "v2", "weight": 4},
            {"source": "v1", "destination": "v3", "weight": 3},
            {"source": "v2", "destination": "v4", "weight": 3},
            {"source": "v3", "destination": "v4", "weight": 4}
        ]

        intents = [
            {"source": "v0", "destination": "v4", "start": 0},
            {"source": "v1", "destination": "v4", "start": 0}
        ]

        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents, self.time_horizon,
                                                                 self.time_delta)

        self.ideal_times = [7, 7]
        self.actual_times = [10, 15]
        self.destination_layers = [2, 3]

    def test_dijkstra(self):
        index = 0

        for intent_name, operation_intent in self.intents.items():
            nodes_list = list(self.nodes.values())
            goal_node = optimization.dijkstra_extended(operation_intent, self.time_delta, nodes_list)
            optimization.dijkstra_original(operation_intent, nodes_list)

            # assert goal node is reached
            self.assertEqual(goal_node.original.name, operation_intent.destination.name)

            # assert goal node is reached at correct layer
            self.assertEqual(goal_node.layer, self.destination_layers[index])

            # assert actual time is correct
            self.assertEqual(operation_intent.actual_time, self.actual_times[index])

            # assert ideal time is correct
            self.assertEqual(operation_intent.ideal_time, self.ideal_times[index])

            main.adjust_capacities(goal_node, self.nodes)

            index += 1


class Example9(unittest.TestCase):
    """Test contains two intents where the second one has to follow a longer path due to limited capacity at `dest`."""
    def setUp(self) -> None:
        self.time_horizon = 20
        self.time_delta = 5
        nodes = [
            {"name": "v0", "capacity": 2},
            {"name": "v1", "capacity": 2},
            {"name": "v2", "capacity": 2},
            {"name": "v3", "capacity": 2},
            {"name": "v4", "capacity": 1}
        ]

        edges = [
            {"source": "v0", "destination": "v2", "weight": 4},
            {"source": "v1", "destination": "v3", "weight": 3},
            {"source": "v2", "destination": "v4", "weight": 3},
            {"source": "v3", "destination": "v2", "weight": 3},
            {"source": "v3", "destination": "v4", "weight": 4}
        ]

        intents = [
            {"source": "v0", "destination": "v4", "start": 0},
            {"source": "v1", "destination": "v4", "start": 0}
        ]

        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents, self.time_horizon,
                                                                 self.time_delta)

        self.ideal_times = [7, 7]
        self.actual_times = [10, 15]
        self.destination_layers = [2, 3]

    def test_dijkstra(self):
        index = 0

        for intent_name, operation_intent in self.intents.items():
            nodes_list = list(self.nodes.values())
            goal_node = optimization.dijkstra_extended(operation_intent, self.time_delta, nodes_list)
            optimization.dijkstra_original(operation_intent, nodes_list)

            # assert goal node is reached
            self.assertEqual(goal_node.original.name, operation_intent.destination.name)

            # assert goal node is reached at correct layer
            self.assertEqual(goal_node.layer, self.destination_layers[index])

            # assert actual time is correct
            self.assertEqual(operation_intent.actual_time, self.actual_times[index])

            # assert ideal time is correct
            self.assertEqual(operation_intent.ideal_time, self.ideal_times[index])

            main.adjust_capacities(goal_node, self.nodes)

            index += 1


class Example10(unittest.TestCase):
    """A large graph with several intents."""
    def setUp(self) -> None:
        self.time_horizon = 30
        self.time_delta = 5
        nodes = [
            {"name": "v0", "capacity": 2},
            {"name": "v1", "capacity": 2},
            {"name": "v2", "capacity": 1},
            {"name": "v3", "capacity": 3},
            {"name": "v4", "capacity": 1},
            {"name": "v5", "capacity": 2}
        ]

        edges = [
            {"source": "v0", "destination": "v1", "weight": 2},
            {"source": "v0", "destination": "v2", "weight": 8},
            {"source": "v1", "destination": "v2", "weight": 5},
            {"source": "v2", "destination": "v1", "weight": 5},
            {"source": "v1", "destination": "v3", "weight": 6},
            {"source": "v3", "destination": "v1", "weight": 6},
            {"source": "v2", "destination": "v3", "weight": 3},
            {"source": "v3", "destination": "v2", "weight": 3},
            {"source": "v2", "destination": "v4", "weight": 2},
            {"source": "v4", "destination": "v2", "weight": 2},
            {"source": "v3", "destination": "v5", "weight": 9},
            {"source": "v5", "destination": "v3", "weight": 9},
            {"source": "v4", "destination": "v5", "weight": 3},
            {"source": "v5", "destination": "v4", "weight": 3}
        ]

        intents = [
            {"source": "v0", "destination": "v5", "start": 0},
            {"source": "v2", "destination": "v3", "start": 1},
            {"source": "v5", "destination": "v1", "start": 0},
            {"source": "v1", "destination": "v0", "start": 0}
        ]

        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents, self.time_horizon,
                                                                 self.time_delta)

        self.ideal_times = [12, 3, 10, 0]
        self.actual_times = [20, 5, 20, 0]
        self.destination_layers = [4, 2, 4, None]

    def test_dijkstra(self):
        index = 0

        for intent_name, operation_intent in list(self.intents.items())[:-1]:
            nodes_list = list(self.nodes.values())
            goal_node = optimization.dijkstra_extended(operation_intent, self.time_delta, nodes_list)
            optimization.dijkstra_original(operation_intent, nodes_list)

            # assert goal node is reached
            self.assertEqual(goal_node.original.name, operation_intent.destination.name)

            # assert goal node is reached at correct layer
            self.assertEqual(goal_node.layer, self.destination_layers[index])

            # assert actual time is correct
            self.assertEqual(operation_intent.actual_time, self.actual_times[index])

            # assert ideal time is correct
            self.assertEqual(operation_intent.ideal_time, self.ideal_times[index])

            main.adjust_capacities(goal_node, self.nodes)

            index += 1

        intent_name, operation_intent = list(self.intents.items())[-1]
        goal_node = optimization.dijkstra_extended(operation_intent, self.time_delta, list(self.nodes.values()))
        self.assertIs(goal_node, None)


if __name__ == "__main__":
    unittest.main()

# =============================================== END OF FILE ===============================================
