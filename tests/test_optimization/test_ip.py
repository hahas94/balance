"""
File description:
-----------------
Testing functionality of the proposed integer programming model.
To run all tests, execute this command from project root:
    `python3 -m unittest discover -s tests`
To run a test locally, the path to the example data must be updated. Example:
    path = '../test_examples/test1.json'
"""

import unittest

import main
import optimization


class Example1(unittest.TestCase):
    """A small graph of two edges and one intent."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, nodes, edges, intents = \
            main.read_example('tests/test_examples/test1.json')
        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents,
                                                                 self.time_horizon, self.time_delta)

        self.time_steps = range(self.start, self.time_horizon + 1, self.time_delta)

        self.actual_time = 20

    def test_ip(self):
        ip_obj = optimization.ip_optimization(self.nodes, self.edges, self.intents, self.time_steps, self.time_delta)
        ip_obj = round(ip_obj, 1)
        self.assertEqual(ip_obj, self.actual_time)


class Example2(unittest.TestCase):
    """A small graph of two edges and one intent, where layer jump is needed."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, nodes, edges, intents = \
            main.read_example('tests/test_examples/test2.json')
        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents,
                                                                 self.time_horizon, self.time_delta)

        self.time_steps = range(self.start, self.time_horizon + 1, self.time_delta)

        self.actual_time = 15

    def test_ip(self):
        ip_obj = optimization.ip_optimization(self.nodes, self.edges, self.intents, self.time_steps, self.time_delta)
        ip_obj = round(ip_obj, 1)
        self.assertEqual(ip_obj, self.actual_time)


class Example4(unittest.TestCase):
    """A small graph where intent travel time is beyond time horizon, hence no solution."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, nodes, edges, intents = \
            main.read_example('tests/test_examples/test4.json')
        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents,
                                                                 self.time_horizon, self.time_delta)

        self.time_steps = range(self.start, self.time_horizon + 1, self.time_delta)

        self.actual_time = None

    def test_ip(self):
        ip_obj = optimization.ip_optimization(self.nodes, self.edges, self.intents, self.time_steps, self.time_delta)
        ip_obj = round(ip_obj, 1) if ip_obj else ip_obj
        self.assertEqual(ip_obj, self.actual_time)


class Example5(unittest.TestCase):
    """A graph with two identical paths between `src` and `dest`."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, nodes, edges, intents = \
            main.read_example('tests/test_examples/test5.json')
        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents,
                                                                 self.time_horizon, self.time_delta)

        self.time_steps = range(self.start, self.time_horizon + 1, self.time_delta)

        self.actual_time = 10

    def test_ip(self):
        ip_obj = optimization.ip_optimization(self.nodes, self.edges, self.intents, self.time_steps, self.time_delta)
        ip_obj = round(ip_obj, 1) if ip_obj else ip_obj
        self.assertEqual(ip_obj, self.actual_time)


class Example6(unittest.TestCase):
    """Start time for the intent is at first layer instead of time 0."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, nodes, edges, intents = \
            main.read_example('tests/test_examples/test6.json')
        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents,
                                                                 self.time_horizon, self.time_delta)

        self.time_steps = range(self.start, self.time_horizon + 1, self.time_delta)

        self.actual_time = 10

    def test_ip(self):
        ip_obj = optimization.ip_optimization(self.nodes, self.edges, self.intents, self.time_steps, self.time_delta)
        ip_obj = round(ip_obj, 1) if ip_obj else ip_obj
        self.assertEqual(ip_obj, self.actual_time)


class Example7(unittest.TestCase):
    """Test contains two intents starting at different nodes."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, nodes, edges, intents = \
            main.read_example('tests/test_examples/test7.json')
        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents,
                                                                 self.time_horizon, self.time_delta)

        self.time_steps = range(self.start, self.time_horizon + 1, self.time_delta)

        self.actual_time = 20

    def test_ip(self):
        ip_obj = optimization.ip_optimization(self.nodes, self.edges, self.intents, self.time_steps, self.time_delta)
        ip_obj = round(ip_obj, 1) if ip_obj else ip_obj
        self.assertEqual(ip_obj, self.actual_time)


class Example8(unittest.TestCase):
    """Test contains two intents where the second one has to wait at `src` due to limited capacity at `dest`."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, nodes, edges, intents = \
            main.read_example('tests/test_examples/test8.json')
        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents,
                                                                 self.time_horizon, self.time_delta)

        self.time_steps = range(self.start, self.time_horizon + 1, self.time_delta)

        self.actual_time = 25

    def test_ip(self):
        ip_obj = optimization.ip_optimization(self.nodes, self.edges, self.intents, self.time_steps, self.time_delta)
        ip_obj = round(ip_obj, 1) if ip_obj else ip_obj
        self.assertEqual(ip_obj, self.actual_time)


class Example9(unittest.TestCase):
    """Test contains two intents where the second one has to follow a longer path due to limited capacity at `dest`."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, nodes, edges, intents = \
            main.read_example('tests/test_examples/test9.json')
        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents,
                                                                 self.time_horizon, self.time_delta)

        self.time_steps = range(self.start, self.time_horizon + 1, self.time_delta)

        self.actual_time = 25

    def test_ip(self):
        ip_obj = optimization.ip_optimization(self.nodes, self.edges, self.intents, self.time_steps, self.time_delta)
        ip_obj = round(ip_obj, 1) if ip_obj else ip_obj
        self.assertEqual(ip_obj, self.actual_time)


class Example10(unittest.TestCase):
    """A large graph with several intents."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, nodes, edges, intents = \
            main.read_example('tests/test_examples/test10.json')
        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents,
                                                                 self.time_horizon, self.time_delta)

        self.time_steps = range(self.start, self.time_horizon + 1, self.time_delta)

        self.actual_time = 45

    def test_ip(self):
        ip_obj = optimization.ip_optimization(self.nodes, self.edges, self.intents, self.time_steps, self.time_delta)
        ip_obj = round(ip_obj, 1) if ip_obj else ip_obj
        self.assertEqual(ip_obj, self.actual_time)


class Example11(unittest.TestCase):
    """Test contains two intents with time uncertainty where the second intent cannot be operated, hence infeasible."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, nodes, edges, intents = \
            main.read_example('tests/test_examples/test11.json')
        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents,
                                                                 self.time_horizon, self.time_delta)

        self.time_steps = range(self.start, self.time_horizon + 1, self.time_delta)

        self.actual_time = 32

    def test_ip(self):
        ip_obj = optimization.ip_optimization(self.nodes, self.edges, self.intents, self.time_steps, self.time_delta)
        ip_obj = round(ip_obj, 1) if ip_obj else ip_obj
        self.assertEqual(ip_obj, self.actual_time)


class Example12(unittest.TestCase):
    """Test contains two intents with time uncertainty where the second intent can be operated as capacity exists."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, nodes, edges, intents = \
            main.read_example('tests/test_examples/test12.json')
        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents,
                                                                 self.time_horizon, self.time_delta)

        self.time_steps = range(self.start, self.time_horizon + 1, self.time_delta)

        self.actual_time = 26

    def test_ip(self):
        ip_obj = optimization.ip_optimization(self.nodes, self.edges, self.intents, self.time_steps, self.time_delta)
        ip_obj = round(ip_obj, 1) if ip_obj else ip_obj
        self.assertEqual(ip_obj, self.actual_time)


class Example13(unittest.TestCase):
    """Test contains two intents where no vertiports are shared."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, nodes, edges, intents = \
            main.read_example('tests/test_examples/test13.json')
        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents,
                                                                 self.time_horizon, self.time_delta)

        self.time_steps = range(self.start, self.time_horizon + 1, self.time_delta)

        self.actual_time = 12

    def test_ip(self):
        ip_obj = optimization.ip_optimization(self.nodes, self.edges, self.intents, self.time_steps, self.time_delta)
        ip_obj = round(ip_obj, 1) if ip_obj else ip_obj
        self.assertEqual(ip_obj, self.actual_time)


class Example14(unittest.TestCase):
    """Same as test 11 but where the second intent starts later hence it can be operated."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, nodes, edges, intents = \
            main.read_example('tests/test_examples/test14.json')
        self.nodes, self.edges, self.intents = main.create_dicts(nodes, edges, intents,
                                                                 self.time_horizon, self.time_delta)

        self.time_steps = range(self.start, self.time_horizon + 1, self.time_delta)

        self.actual_time = 28

    def test_ip(self):
        ip_obj = optimization.ip_optimization(self.nodes, self.edges, self.intents, self.time_steps, self.time_delta)
        ip_obj = round(ip_obj, 1) if ip_obj else ip_obj
        self.assertEqual(ip_obj, self.actual_time)


if __name__ == "__main__":
    unittest.main()

# =============================================== END OF FILE ===============================================
