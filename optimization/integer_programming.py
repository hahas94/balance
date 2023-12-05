"""
File description:
-----------------
This file implements an integer programming model for solving the drone routing problem
using the Gurobi solver.

Given a list of operational intents, this model creates a list for their earliest departure times as a constant,
as well as a list of all edges, and a list of time steps.

The model has a binary decision variable for each edge, drone and time step, where the variable has value 1
if drone `d` traverses that edge at time `t`.
Likewise, a binary decision variable exists for each vertiport, intent and time step, where it has value 1 if
vertiport `i` is reserved for drone d at time `t`.

There are five constraints to be satisfied:
1. Flow conservation: a drone entering a vertiport must also leave it, except the source and destination
   vertiports.
2. Reservation: a vertiport must be reserved for the entire duration of a drone flying towards it, inclusive the
   time uncertainties.
3. Capacity: a vertiport cannot be overcapacitated at any point in time.
4. Departure time: an intent cannot start earlier than time of departure, but it can start later.
5. Arrival: an intent must end up at its destination vertiport.

The objective of the model is to find the shortest total time of operation for all intents.
"""

import mip


def calculate_edge_weight(weight: int, time_delta: int) -> int:
    """
    This function calculates the corresponding time discretized weight of an edge given its original weight.

    Parameters
    ----------
    weight: int
        Original weight of an edge in minutes.
    time_delta: int
        Time discretization step.

    Returns
    -------
    weight: int
        New weight

    """
    k, r = divmod(weight, time_delta)
    weight = (k + int(r > 0)) * time_delta

    return weight


def ip_optimization(nodes: dict, edges: dict, intents: dict, time_steps: range, time_delta: int) -> float:
    """
    This function runs an optimization model to find routing solution for each operational intent.

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
    ip_obj: float
        The objective of the model
    """
    ip_obj = 0

    # ---- Constants ----
    vertiports_list = list(nodes.keys())
    vertiports_ids = range(len(vertiports_list))
    edges_list = list(edges.keys())
    edges_ids = range(len(edges_list))
    drones_list = list(intents.keys())
    drones_ids = range(len(drones_list))
    time_steps_ids = range(len(time_steps))
    destination_vertiports_names = [intent.destination.name for intent in intents.values()]
    earliest_departure_times = [intent.start for intent in intents.values()]
    edge_weights = [calculate_edge_weight(edge.weight, time_delta) for edge in edges.values()]

    # ---- Model definition ----
    model = mip.Model(name="balance", sense=mip.MINIMIZE, solver_name=mip.GUROBI)

    # ---- Decision variables ----
    # whether edge e is traversed by drone d starting at time step t
    drones_path = [[[model.add_var(name=f"e{e}d{d}t{t*time_delta}", var_type=mip.BINARY)
                     for t in time_steps_ids]
                    for d in drones_ids]
                   for e in edges_ids]

    # whether vertiport v is reserved for drone d starting at time step t
    vertiport_reserved = [[[model.add_var(name=f"v{v}d{d}t{t*time_delta}", var_type=mip.BINARY)
                            for t in time_steps_ids]
                           for d in drones_ids]
                          for v in vertiports_ids]

    # ---- Constraints ----
    # Arrival constraint: a drone must end up in its detination vertiport
    for d in drones_ids:
        dest = destination_vertiports_names[d]
        valid_edges = [idx for idx in edges_ids if edges_list[idx][1] == dest]
        model.add_constr(mip.xsum(drones_path[e][d][t] for t in time_steps_ids for e in valid_edges) == 1,
                         "Arrival constraint")

    # ---- Objective ----
    # minimizing the total sum of times of all drone schedules.
    model.objective = mip.xsum(drones_path[e][d][t] * edge_weights[e]
                               for e in edges_ids for d in drones_ids for t in time_steps_ids
                               )
    model.optimize()

    # ---- Output ----
    # checking if a solution was found
    if model.num_solutions:
        ip_obj = model.objective_value
        print(f"ip_obj={ip_obj}")
        for el in model.vars:
            print(f"{el}: {el.x}")

    return ip_obj

# =============================================== END OF FILE ===============================================
