"""
File description:
-----------------
This file implements an integer programming model for solving the drone routing problem
using the Gurobi solver.

Given a list of operational intents, this model creates a list for their earliest departure times as a constant,
as well as a list of all edges, and a list of time steps.

The model has a binary decision variable for each edge, drone and time step, where the variable has value 1
if drone `d` starts traversing that edge at time `t`.
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
import math

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
    # edges_list = list(edges.keys())
    # list of all valid edges plus extra edges indicating ground delay.
    edges_list = list(edges.keys()) + [(intent.source.name, intent.source.name) for intent in intents.values()]
    edges_ids = range(len(edges_list))
    drones_list = list(intents.keys())
    drones_ids = range(len(drones_list))
    time_steps_ids = range(len(time_steps))
    source_vertiports_names = [intent.source.name for intent in intents.values()]
    destination_vertiports_names = [intent.destination.name for intent in intents.values()]
    earliest_departure_times = [intent.start for intent in intents.values()]
    edge_weights = [edge.weight for edge in edges.values()] + [time_delta for _ in drones_ids]
    edge_weights_td = [calculate_edge_weight(edge.weight, time_delta) for edge in edges.values()] + \
                      [time_delta for _ in drones_ids]  # edge weights in terms of time delta

    # ---- Model definition ----
    model = mip.Model(name="balance", sense=mip.MINIMIZE, solver_name=mip.GUROBI)

    # ---- Decision variables ----
    # whether drone d traverses edge e starting at time step t
    drones_path = [[[model.add_var(name=f"Edge_{edges_list[e]}__Drone_{d}__Time_{t * time_delta}", var_type=mip.BINARY)
                     for t in time_steps_ids]
                    for d in drones_ids]
                   for e in edges_ids]

    # whether vertiport v is reserved for drone d starting at time step t
    vertiport_reserved = [[[model.add_var(name=f"Vertiport_{vertiports_list[v]}__Drone_{d}__Time_{t * time_delta}",
                                          var_type=mip.BINARY)
                            for t in time_steps_ids]
                           for d in drones_ids]
                          for v in vertiports_ids]

    # ---- Constraints ----
    # 1. Flow conservation, meaning for each drone and for each vertiport other than
    # departure and arrival vertiports, whatever comes in must also leave out. For
    # the departure vertiport nothing comes into it and for the arrival vertiport
    # nothing gets out of it.
    # for d in drones_ids:
    #     src, dest = source_vertiports_names[d], destination_vertiports_names[d]
    #     departure = earliest_departure_times[d]
    #     for k in vertiports_ids:
    #         for t in time_steps_ids:
    #             # case 1: where `k` is the origin vertiport and `t` is departure time:
    #             if vertiports_list[k] == src and (t*time_delta) == departure:
    #                 model.add_constr(mip.xsum(drones_path[e][d][t]
    #                                           for e in edges_ids if edges_list[e][1] == vertiports_list[k])
    #                                  - mip.xsum(drones_path[e][d][t]
    #                                             for e in edges_ids if edges_list[e][0] == vertiports_list[k])
    #                                  == -1, f"Flow conservation constraint, drone {d} origin vertiport and dept time.")
    #             # case 2: where `k` is the destination vertiport
    #             elif vertiports_list[k] == dest:
    #                 model.add_constr(mip.xsum(drones_path[e][d][t]
    #                                           for e in edges_ids if edges_list[e][1] == vertiports_list[k])
    #                                  - mip.xsum(drones_path[e][d][t]
    #                                             for e in edges_ids if edges_list[e][0] == vertiports_list[k])
    #                                  <= 1, f"Flow conservation constraint, drone {d} dest vertiport.")
    #             else:
    #                 # case 3: other vertiport
    #                 model.add_constr(mip.xsum(drones_path[e][d][t]
    #                                           for e in edges_ids if edges_list[e][1] == vertiports_list[k])
    #                                  - mip.xsum(drones_path[e][d][t]
    #                                             for e in edges_ids if edges_list[e][0] == vertiports_list[k])
    #                                  <= 0, f"Flow conservation constraint, drone {d} other vertiports.")

    # for d in drones_ids:
    #     src, dest = source_vertiports_names[d], destination_vertiports_names[d]
    #     for k in vertiports_ids:
    #         k_name = vertiports_list[k]
    #         valid_edges_in = [idx for idx in edges_ids if edges_list[idx][1] == k_name]
    #         valid_edges_out = [idx for idx in edges_ids if edges_list[idx][0] == k_name]
    #
    #         if k_name == dest:
    #             model.add_constr(mip.xsum(drones_path[e][d][t] for t in time_steps_ids for e in valid_edges_in)
    #                              - mip.xsum(drones_path[e][d][t] for t in time_steps_ids for e in valid_edges_out)
    #                              <= 1,
    #                              f"Flow conservation constraint, drone {d}")
    #
    #         else:
    #             rhs = -1 if k_name == src else 0
    #             model.add_constr(mip.xsum(drones_path[e][d][t] for t in time_steps_ids for e in valid_edges_in)
    #                              - mip.xsum(drones_path[e][d][t] for t in time_steps_ids for e in valid_edges_out)
    #                              == rhs,
    #                              f"Flow conservation constraint, drone {d}")

    model.add_constr(drones_path[0][0][0] == 1, "first edge")
    model.add_constr(drones_path[1][0][1] == 1, "second edge")

    # 2. A vertiport must be reserved for the entire duration of a drone flying towards it, inclusive
    #    the time uncertainties, that is being reserved from start time plus duration and uncertainty.
    # for v in vertiports_ids:
    #     M = len(edges_list)
    #     valid_edges = [idx for idx in edges_ids if edges_list[idx][1] == vertiports_list[v]]
    #     for d in drones_ids:
    #         U = intents[drones_list[d]].time_uncertainty
    #         # edge weights with drone uncertainty added to them, except for edges indicating ground delay.
    #         valid_edges_weight_uncertainty = [
    #             calculate_edge_weight(edge_weights[idx] + U*int(idx < len(edges)), time_delta) // time_delta
    #             for idx in valid_edges]
    #         for t in time_steps_ids:
    #             time_ranges = [
    #                 range(max(0, (t - U) // time_delta), min(t + w, time_steps_ids[-1]))
    #                 for w in valid_edges_weight_uncertainty
    #             ]
    #             model.add_constr(
    #                 (1 / M) * mip.xsum(drones_path[e][d][tau] for indx, e in enumerate(valid_edges)
    #                                    for tau in time_ranges[indx])
    #                 <= vertiport_reserved[v][d][t],
    #                 f"Reservation constraint Vertiport_{v}__Drone_{d}__Time_{t*time_delta}")

    for v in vertiports_ids:
        M = len(edges_list)
        valid_edges = [idx for idx in edges_ids if edges_list[idx][1] == vertiports_list[v]]
        for d in drones_ids:
            U = intents[drones_list[d]].time_uncertainty
            # edge weights with drone uncertainty added to them, except for edges indicating ground delay.
            valid_edges_weight_uncertainty = [
                calculate_edge_weight(edge_weights[idx] + U*int(idx < len(edges)), time_delta) // time_delta
                for idx in valid_edges]
            for t in time_steps_ids:
                time_ranges = [
                    range(max(0, t - w), min((t + U)//time_delta, time_steps_ids[-1]))
                    for w in valid_edges_weight_uncertainty
                ]
                model.add_constr(
                    (1 / M) * mip.xsum(drones_path[e][d][tau] for indx, e in enumerate(valid_edges)
                                       for tau in time_ranges[indx])
                    <= vertiport_reserved[v][d][t],
                    f"Reservation constraint Vertiport_{v}__Drone_{d}__Time_{t*time_delta}")

    # 3. A vertiport cannot be overcapacitated at any point in time, meaning sum
    #    of all reservations of a vertiport cannot exceed the maximum capacity.
    for v in vertiports_ids:
        v_max_capacity = nodes[vertiports_list[v]].capacity
        for t in time_steps_ids:
            model.add_constr(mip.xsum(vertiport_reserved[v][d][t] for d in drones_ids)
                             <= v_max_capacity, "Capacity constraint")

    # 4. An operation cannot start earlier than its time of departure, that is all edges with source
    #    being start vertiport and at all times before departure, these edges must have value 0.
    for d in drones_ids:
        src = source_vertiports_names[d]
        departure_time = earliest_departure_times[d]
        earlier_times = range(departure_time)
        valid_edges = [idx for idx in edges_ids if edges_list[idx][0] == src]
        model.add_constr(mip.xsum(drones_path[e][d][t] for t in earlier_times for e in valid_edges) == 0,
                         "No early departure constraint")

    # 5. Arrival constraint: a drone must end up in its detination vertiport.
    for d in drones_ids:
        dest = destination_vertiports_names[d]
        valid_edges = [idx for idx in edges_ids if edges_list[idx][1] == dest]
        model.add_constr(mip.xsum(drones_path[e][d][t] for t in time_steps_ids[:-1] for e in valid_edges) == 1,
                         "Arrival constraint")

    # Extra: A drone can traverse one edge at a time.
    for d in drones_ids:
        for t in time_steps_ids:
            model.add_constr(mip.xsum(drones_path[e][d][t] for e in edges_ids) <= 1, "One edge at a time is traversed.")

    # ---- Objective ----
    # minimizing the total sum of times of all drone schedules.
    model.objective = mip.xsum(drones_path[e][d][t] * edge_weights_td[e]
                               for e in edges_ids for d in drones_ids for t in time_steps_ids
                               )
    status = model.optimize()

    # ---- Output ----
    # checking if a solution was found
    if model.num_solutions:
        ip_obj = model.objective_value
        print(f"ip_obj={ip_obj}")
        for el in model.vars:
            print(f"{el}: {el.x}")

    return ip_obj

# =============================================== END OF FILE ===============================================
