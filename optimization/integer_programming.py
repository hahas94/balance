"""
File description:
-----------------
This file implements an integer programming model for solving the drone routing problem
using the Gurobi solver.

Given a list of operational intents, this model creates a list for their earliest departure times as a constant,
as well as a list of all edges, and a list of time steps.

The model has a binary decision variable for each edge, drone and time step, where the variable has value 1
if drone d traverses that edge at time t.
Likewise, a binary decision variable exists for each vertiport, intent and time step, where it has value 1 if
vertiport i is reserved for drone d at time t.

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

model = mip.Model(solver_name="gurobi")
print(model)

# =============================================== END OF FILE ===============================================
