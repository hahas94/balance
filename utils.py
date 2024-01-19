"""
File description:
-----------------
This file contains functions and/or classes that can be used across the project files.
"""


class Link:
    """Class Link that represents a link (point) in a path. """
    def __init__(self, name: str, layer: int, travel_time: int, left_reserved_layer: int = None,
                 right_reserved_layer: int = None):
        """
        Initialization of a Link instance.
        Parameters
        ----------
        name: str
            Name of vertiport
        layer: int
            Layer at which drone arrives at vertiport.
        travel_time: int
            Travel time so far into the journey.
        left_reserved_layer: int
            The left most reserved layer of this vertiport.
        right_reserved_layer: int
            The right most reserved layer of this vertiport.
        """
        self._name = name
        self._layer = layer
        self._travel_time = travel_time
        self._left_reserved_layer = left_reserved_layer
        self._right_reserved_layer = right_reserved_layer
        self._probably_left_reserved_layer = None  # left reserved layer during increment/decrement

    def __str__(self):
        return f"{self._name}_{self._layer}"

    def __eq__(self, other):
        """Two links are equal if they are identical."""
        identical_names = self._name == other.name
        identical_layers = self._layer == other.layer
        identical_tt = self._travel_time == other.travel_time
        identical_lul = self._left_reserved_layer == other.left_reserved_layer
        identical_rul = self._right_reserved_layer == other.right_reserved_layer

        return identical_names and identical_layers and identical_tt and identical_lul and identical_rul

    @property
    def name(self):
        return self._name

    @property
    def layer(self):
        return self._layer

    @property
    def travel_time(self):
        return self._travel_time

    @property
    def left_reserved_layer(self):
        if self._left_reserved_layer:
            return self._left_reserved_layer
        elif self._probably_left_reserved_layer is None:
            return None
        return max(1, self._probably_left_reserved_layer)

    @left_reserved_layer.setter
    def left_reserved_layer(self, x):
        self._left_reserved_layer = x

    @property
    def right_reserved_layer(self):
        return self._right_reserved_layer

    @property
    def probably_left_reserved_layer(self):
        return self._probably_left_reserved_layer

    @probably_left_reserved_layer.setter
    def probably_left_reserved_layer(self, x):
        self._probably_left_reserved_layer = x

# =============================================== END OF FILE ===============================================
