"""
Integrated ATM Cash Replenishment + Vehicle Routing Model
Kolkata case study | Undergraduate OR project

Synthetic data:
    kolkata_atm_synthetic_data.csv

Model:
    - one cash depot
    - one truck
    - at most 10 ATM visits
    - truck cash capacity = 25 lakh INR
    - three demand scenarios: low, normal, high
    - scenario-weighted shortage penalties
    - ATM priority
    - 9:00-17:00 service window
    - 8-hour route duration
    - TSP/VRP-style routing with MTZ subtour elimination

Requires:
    pip install pandas gurobipy
    A valid Gurobi license
"""

import math
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

DATA_FILE = "kolkata_atm_synthetic_data.csv"

TRUCK_CAPACITY_LAKH = 25.0
MAX_VISITS = 10
AVERAGE_SPEED_KMPH = 25.0
MAX_ROUTE_HOURS = 8.0

# Objective weights
TRAVEL_COST_PER_KM = 1.0
VISIT_FIXED_COST = 1.5
CASH_HOLDING_COST = 0.15
SHORTAGE_PENALTY = 30.0

# Demand scenario probabilities
SCENARIOS = {
    "low_demand_lakh": 0.20,
    "normal_demand_lakh": 0.50,
    "high_demand_lakh": 0.30,
}

# -----------------------------
# Distance calculation
# -----------------------------
def road_distance_km(lat1, lon1, lat2, lon2, road_factor=1.25):
    """Approximate road distance from coordinates."""
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    h = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    )
    straight = 2 * R * math.asin(math.sqrt(h))
    return road_factor * straight


# -----------------------------
# Data
# -----------------------------
df = pd.read_csv(DATA_FILE)

# Synthetic central Kolkata depot
depot = {
    "atm_id": "D0",
    "location": "Cash Depot",
    "latitude": 22.5726,
    "longitude": 88.3639,
}

locations = [depot] + df[
    ["atm_id", "location", "latitude", "longitude"]
].to_dict("records")

N = len(locations)
nodes = range(N)
customers = range(1, N)

# -----------------------------
# Distance matrix
# -----------------------------
dist = {}
for i in nodes:
    for j in nodes:
        if i != j:
            a, b = locations[i], locations[j]
            dist[i, j] = road_distance_km(
                a["latitude"], a["longitude"],
                b["latitude"], b["longitude"]
            )

# -----------------------------
# Model
# -----------------------------
model = gp.Model("Integrated_ATM_Cash_Routing")

# x[i,j] = 1 if truck travels i -> j
x = model.addVars(
    [(i, j) for i in nodes for j in nodes if i != j],
    vtype=GRB.BINARY,
    name="x"
)

# y[i] = 1 if ATM i is visited
y = model.addVars(customers, vtype=GRB.BINARY, name="y")

# q[i] = cash delivered to ATM i (lakh INR)
q = model.addVars(
    customers,
    lb=0,
    ub=GRB.INFINITY,
    vtype=GRB.CONTINUOUS,
    name="q"
)

# shortage[i,s] = shortage in scenario s
shortage = model.addVars(
    customers,
    list(SCENARIOS.keys()),
    lb=0,
    vtype=GRB.CONTINUOUS,
    name="shortage"
)

# t[i] = arrival time at ATM i
t = model.addVars(
    customers,
    lb=9.0,
    ub=17.0,
    vtype=GRB.CONTINUOUS,
    name="arrival"
)

# u[i] = MTZ ordering variable
u = model.addVars(
    customers,
    lb=0,
    ub=N-1,
    vtype=GRB.CONTINUOUS,
    name="order"
)

# -----------------------------
# Routing constraints
# -----------------------------

# If ATM i is visited, exactly one incoming arc.
for i in customers:
    model.addConstr(
        gp.quicksum(x[j, i] for j in nodes if j != i) == y[i],
        name=f"one_in_{i}"
    )

# If ATM i is visited, exactly one outgoing arc.
for i in customers:
    model.addConstr(
        gp.quicksum(x[i, j] for j in nodes if j != i) == y[i],
        name=f"one_out_{i}"
    )

# Truck leaves and returns to depot once.
model.addConstr(
    gp.quicksum(x[0, j] for j in customers) == 1,
    name="depot_departure"
)
model.addConstr(
    gp.quicksum(x[i, 0] for i in customers) == 1,
    name="depot_return"
)

# No arc between two unvisited ATMs.
for i in customers:
    for j in customers:
        if i != j:
            model.addConstr(x[i, j] <= y[i], name=f"arc_from_{i}_{j}")
            model.addConstr(x[i, j] <= y[j], name=f"arc_to_{i}_{j}")

# At most MAX_VISITS ATMs.
model.addConstr(
    gp.quicksum(y[i] for i in customers) <= MAX_VISITS,
    name="maximum_visits"
)

# Priority-3 ATMs must be visited.
for i in customers:
    if int(df.iloc[i-1]["priority"]) == 3:
        model.addConstr(y[i] == 1, name=f"priority_{i}")

# MTZ subtour elimination.
for i in customers:
    model.addConstr(u[i] <= (N-1) * y[i], name=f"order_upper_{i}")
    model.addConstr(u[i] >= y[i], name=f"order_lower_{i}")

for i in customers:
    for j in customers:
        if i != j:
            model.addConstr(
                u[i] - u[j] + (N-1) * x[i, j] <= N-2,
                name=f"mtz_{i}_{j}"
            )

# -----------------------------
# Cash/inventory constraints
# -----------------------------

# Truck capacity.
model.addConstr(
    gp.quicksum(q[i] for i in customers) <= TRUCK_CAPACITY_LAKH,
    name="truck_cash_capacity"
)

# Cash can only be delivered if ATM is visited, and cannot exceed ATM capacity.
for i in customers:
    max_delivery = float(df.iloc[i-1]["max_delivery_lakh"])
    model.addConstr(
        q[i] <= max_delivery * y[i],
        name=f"delivery_visit_{i}"
    )

# Scenario shortage:
# current cash + delivery + shortage >= scenario demand.
for i in customers:
    current = float(df.iloc[i-1]["current_cash_lakh"])

    for scenario in SCENARIOS:
        demand = float(df.iloc[i-1][scenario])
        model.addConstr(
            current + q[i] + shortage[i, scenario] >= demand,
            name=f"shortage_{i}_{scenario}"
        )

# -----------------------------
# Route duration
# -----------------------------
travel_time = gp.quicksum(
    (dist[i, j] / AVERAGE_SPEED_KMPH) * x[i, j]
    for i, j in dist
)

service_time = gp.quicksum(
    float(df.iloc[i-1]["service_hours"]) * y[i]
    for i in customers
)

model.addConstr(
    travel_time + service_time <= MAX_ROUTE_HOURS,
    name="route_duration"
)

# -----------------------------
# Time windows
# -----------------------------
BIG_M = 50.0

# If the truck travels i -> j, arrival at j must occur after:
# arrival at i + service time at i + travel time.
for i in customers:
    for j in customers:
        if i != j:
            model.addConstr(
                t[j] >=
                t[i]
                + float(df.iloc[i-1]["service_hours"])
                + dist[i, j] / AVERAGE_SPEED_KMPH
                - BIG_M * (1 - x[i, j]),
                name=f"time_{i}_{j}"
            )

# Depot departure occurs at 9:00.
for j in customers:
    model.addConstr(
        t[j] >=
        9.0
        + dist[0, j] / AVERAGE_SPEED_KMPH
        - BIG_M * (1 - x[0, j]),
        name=f"depot_time_{j}"
    )

# -----------------------------
# Objective
# -----------------------------
travel_cost = TRAVEL_COST_PER_KM * gp.quicksum(
    dist[i, j] * x[i, j] for i, j in dist
)

visit_cost = VISIT_FIXED_COST * gp.quicksum(
    y[i] for i in customers
)

holding_cost = CASH_HOLDING_COST * gp.quicksum(
    q[i] for i in customers
)

expected_shortage_cost = gp.quicksum(
    SHORTAGE_PENALTY * probability * shortage[i, scenario]
    for scenario, probability in SCENARIOS.items()
    for i in customers
)

model.setObjective(
    travel_cost + visit_cost + holding_cost + expected_shortage_cost,
    GRB.MINIMIZE
)

# -----------------------------
# Solve
# -----------------------------
model.Params.MIPGap = 0.01
model.Params.TimeLimit = 300

model.optimize()

if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
    raise RuntimeError(f"Gurobi ended with status {model.Status}")

# -----------------------------
# Report results
# -----------------------------
print("\n========== MODEL RESULTS ==========")
print(f"Objective value: {model.ObjVal:.2f}")

visited = [i for i in customers if y[i].X > 0.5]
print(f"ATMs visited: {len(visited)}")

# Recover route
route = [0]
current = 0
while True:
    next_nodes = [
        j for j in nodes
        if j != current and x[current, j].X > 0.5
    ]
    if not next_nodes:
        break
    nxt = next_nodes[0]
    route.append(nxt)
    if nxt == 0:
        break
    current = nxt

print("\nRoute:")
for k, node in enumerate(route):
    print(f"{k}. {locations[node]['location']}")

total_distance = sum(
    dist[route[k], route[k+1]]
    for k in range(len(route)-1)
)

cash_loaded = sum(q[i].X for i in customers)
route_hours = (
    total_distance / AVERAGE_SPEED_KMPH
    + sum(df.iloc[i-1]["service_hours"] * y[i].X for i in customers)
)

print(f"\nTravel distance: {total_distance:.2f} km")
print(f"Cash loaded: {cash_loaded:.2f} lakh")
print(f"Route duration: {route_hours:.2f} hours")

print("\nCash allocation:")
for i in customers:
    if y[i].X > 0.5:
        print(
            f"{df.iloc[i-1]['atm_id']} "
            f"{df.iloc[i-1]['location']}: "
            f"{q[i].X:.2f} lakh"
        )

print("\nScenario shortages:")
for scenario in SCENARIOS:
    total = sum(shortage[i, scenario].X for i in customers)
    print(f"{scenario}: {total:.2f} lakh")
