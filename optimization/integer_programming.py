"""
File description:
-----------------
This file implements an integer programming model for solving the drone routing problem
using the Gurobi solver.

Given a list of operational intents, this model creates a list for their earliest departure times as a constant,
as well as a list of all edges, and a list of time steps.

The model has a binary decision variable for each edge, drone and time step, where the variable has value 1
if drone `d` starts traversing that edge at time `t`. Likewise, another binary variable exist for when the
drone finished taking that edge, i.e. arriving.
Additionally, a binary decision variable exists for each vertiport, drone and time step, where it has value 1 if
vertiport `i` is reserved for drone d at time `t`.

There are a number of constraints to be satisfied:
1. Flow conservation: a drone entering a vertiport must also leave it, except the source and destination
   vertiports.
2. Reservation: a vertiport must be reserved for the entire duration of a drone flying towards it, inclusive the
   time uncertainties.
3. Capacity: a vertiport cannot be overcapacitated at any point in time.
4. Departure time: an intent cannot start earlier than time of departure, but it can start later.
5. Arrival: an intent must end up at its destination vertiport.
6. Arrival and departure times: A drone arriving at a vertiport must have departed earlier as per the edge weight.

The objective of the model is to find the shortest total time of operation for all intents.
"""
import math
from typing import Union

import mip


def ip_optimization(nodes: dict, edges: dict, intents: dict, time_steps: range, time_delta: int) -> Union[float, None]:
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
    # list of all valid edges plus extra edges indicating ground delay.
    edges_list = list(edges.keys()) + [(intent.source.name, intent.source.name) for intent in intents.values()]
    edges_ids = range(len(edges_list))
    drones_list = list(intents.keys())
    drones_ids = range(len(drones_list))
    time_steps_ids = range(len(time_steps))
    source_vertiports_names = [intent.source.name for intent in intents.values()]
    destination_vertiports_names = [intent.destination.name for intent in intents.values()]
    earliest_departure_times = [math.ceil(intent.start/time_delta)*time_delta for intent in intents.values()]
    edge_weights = [edge.weight for edge in edges.values()] + [time_delta for _ in drones_ids]
    edge_weights_td = [math.ceil(edge.weight/time_delta)*time_delta for edge in edges.values()] + \
                      [time_delta for _ in drones_ids]  # edge weights in terms of time delta

    # ---- Model definition ----
    model = mip.Model(name="balance", sense=mip.MINIMIZE, solver_name=mip.GUROBI)

    # ---- Decision variables ----
    # whether drone `d` traverses edge `e` starting at time step `t`
    drones_departure = [[[model.add_var(name=f"Departure_{edges_list[e]}__D{d}__T{t * time_delta}",
                                        var_type=mip.BINARY)
                          for t in time_steps_ids] for d in drones_ids] for e in edges_ids]

    # whether drone `d` finishes taking edge `e` at time step `t`
    drones_arrival = [[[model.add_var(name=f"Arrival_{edges_list[e]}__D{d}__T_{t * time_delta}",
                                      var_type=mip.BINARY)
                        for t in time_steps_ids] for d in drones_ids] for e in edges_ids]

    # whether vertiport `v` is reserved for drone `d` starting at time step `t`
    vertiport_reserved = [[[model.add_var(name=f"Vertiport_{vertiports_list[v]}__D{d}__T_{t * time_delta}",
                                          var_type=mip.BINARY)
                            for t in time_steps_ids] for d in drones_ids] for v in vertiports_ids]

    # ---- Constraints ----
    # 1. Flow conservation, meaning for each drone and for each vertiport other than departure
    # and arrival vertiports, whatever comes in must also leave out. For the departure
    # vertiport nothing comes into it and for the arrival vertiport nothing gets out of it.
    for d in drones_ids:
        src, dest = source_vertiports_names[d], destination_vertiports_names[d]
        dept_time = earliest_departure_times[d]
        for k in vertiports_ids:
            k_name = vertiports_list[k]
            incoming_edges = [e for e in edges_ids if edges_list[e][1] == k_name]
            outgoing_edges = [e for e in edges_ids if edges_list[e][0] == k_name]
            msg = f"Flow conservation constraint__D{d}__{k_name} "
            for t in time_steps_ids:
                # case 1: where `k` is the origin vertiport and `t` is departure time:
                if k_name == src and (t*time_delta) == dept_time:
                    model.add_constr(mip.xsum(drones_arrival[e][d][t] for e in incoming_edges)
                                     - mip.xsum(drones_departure[e][d][t] for e in outgoing_edges)
                                     == -1, msg)
                # case 2: where `k` is the destination vertiport
                elif k_name == dest:
                    model.add_constr(mip.xsum(drones_arrival[e][d][t] for e in incoming_edges)
                                     - mip.xsum(drones_departure[e][d][t] for e in outgoing_edges)
                                     <= 1, msg)
                else:
                    # case 3: other vertiport
                    model.add_constr(mip.xsum(drones_arrival[e][d][t] for e in incoming_edges)
                                     - mip.xsum(drones_departure[e][d][t] for e in outgoing_edges)
                                     == 0, msg)

    # 2. A vertiport must be reserved for the entire duration of a drone flying towards it, inclusive
    #    the time uncertainties, that is being reserved from start time plus duration and uncertainty.
    for v in vertiports_ids:
        M = len(edges_list)
        valid_edges = [idx for idx in edges_ids if edges_list[idx][1] == vertiports_list[v]]
        for d in drones_ids:
            U = intents[drones_list[d]].time_uncertainty
            # edge weights with drone uncertainty added to them, except for edges indicating ground delay.
            weight_uncertainty = [
                math.ceil((edge_weights[idx]+U*int(idx < len(edges))) / time_delta) for idx in valid_edges
            ]
            for t in time_steps_ids[:-1]:
                T = time_steps_ids[-1]
                ub = math.ceil((t*time_delta + U) / time_delta)
                time_ranges = [range(max(0, t - weight_uncertainty[idx] + 1), min(ub if idx < len(edges) else t, T) + 1)
                               for idx, _ in enumerate(valid_edges)]
                model.add_constr(
                    (1 / M) * mip.xsum(drones_departure[e][d][tau] for indx, e in enumerate(valid_edges)
                                       for tau in time_ranges[indx])
                    <= vertiport_reserved[v][d][t+1],
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
        departure_time = earliest_departure_times[d] // time_delta
        earlier_times = range(departure_time)
        valid_edges = [idx for idx in edges_ids if edges_list[idx][0] == src]
        model.add_constr(mip.xsum(drones_departure[e][d][t] for t in earlier_times for e in valid_edges) == 0,
                         "No early departure constraint")

    # 5. Arrival constraint: a drone must end up in its detination vertiport.
    for d in drones_ids:
        dest = destination_vertiports_names[d]
        valid_edges = [idx for idx in edges_ids if edges_list[idx][1] == dest]
        model.add_constr(mip.xsum(drones_departure[e][d][t] for t in time_steps_ids[:-1] for e in valid_edges) == 1,
                         "Arrival constraint")

    # 6. The value of the decision variable for arrival of an edge must equal
    #    the value of the decision variable of departure time for that edge.
    for e in edges_ids:
        edge_weight = edge_weights_td[e]
        for d in drones_ids:
            for t in time_steps_ids:
                w = t - (edge_weight//time_delta)
                if w >= 0:
                    model.add_constr(drones_arrival[e][d][t] == drones_departure[e][d][w],
                                     "Drone arrival and departure times differs by edge weight.")

    # Extra: A drone's total travel time cannot pass beyond time horizon.
    T = time_steps[-1]
    for d in drones_ids:
        model.add_constr(mip.xsum(drones_departure[e][d][t]*edge_weights_td[e]
                                  for t in time_steps_ids for e in edges_ids)
                         <= T, "")

    # Extra: A drone can traverse one edge at a time.
    for d in drones_ids:
        for t in time_steps_ids:
            model.add_constr(mip.xsum(drones_departure[e][d][t] for e in edges_ids) <= 1,
                             "One edge at a time is traversed.")

    # ---- Objective ----
    # minimizing the total sum of times of all drone schedules.
    model.objective = (mip.xsum(drones_departure[e][d][t] * edge_weights_td[e]
                                for e in edges_ids for d in drones_ids for t in time_steps_ids)
                       + mip.xsum(vertiport_reserved[v][d][t]
                                  for t in time_steps_ids for d in drones_ids for v in vertiports_ids)/9999999)
    model.optimize()

    # ---- Output ----
    # checking if a solution was found
    if model.num_solutions:
        ip_obj = model.objective_value

        # Creating the layers reservations dict
        # reservations = {v: (time_steps_ids[-1]+1)*[0] for v in vertiports_list}
        # for var in model.vars:
        #     if var.name[:3] == "Ver" and var.x == 1:
        #         word_lst = var.name.split('_')
        #         name = word_lst[1]
        #         time = int(word_lst[-1])
        #         reservations[name][time] = var.x
        # for k, v in reservations.items():
        #     print(f"{k}: {v}")

        for d in drones_ids:
            path = []
            intent = intents[drones_list[d]]
            travel_time = 0
            arr_var = None
            for t in time_steps_ids:
                for e in edges_ids:
                    w = edge_weights_td[e]
                    dep_var = drones_departure[e][d][t]
                    arr_var = drones_arrival[e][d][t + (w//time_delta)] if dep_var.x > 0 else arr_var
                    if dep_var.x > 0:
                        # (node_name, layer, travel_time, right_most_reserved_layer)
                        U = intent.time_uncertainty
                        right_most_reserved_layer = math.ceil(edge_weights[e] + U * int(e < len(edges)))
                        link = (edges_list[e][0], t, travel_time, right_most_reserved_layer)
                        travel_time += w
                        path.append(link)

            # adding arrival link for completeness
            if arr_var:
                layer = int(arr_var.name.split('_')[-1]) // time_delta
                rmsl = math.ceil(layer + intent.time_uncertainty/time_delta)
                arrival_link = (intent.destination.name, layer, travel_time, rmsl)
                path.append(arrival_link)

                intent.actual_ip_time = travel_time
                intent.build_ip_path(path)

    return ip_obj if model.num_solutions else None

# =============================================== END OF FILE ===============================================
