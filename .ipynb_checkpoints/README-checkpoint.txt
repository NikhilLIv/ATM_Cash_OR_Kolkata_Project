ATM CASH REPLENISHMENT + ROUTING — KOLKATA
==============================================

Files
-----
ATM_Cash_Replenishment_OR_Report_Kolkata.docx
    Full undergraduate-level report.

atm_cash_routing_gurobi.py
    Integrated MILP implementation using Gurobi.

kolkata_atm_synthetic_data.csv
    15 synthetic ATM records.

results_template.csv
    Suggested sensitivity-analysis experiments.

Model
-----
One depot, one truck, 15 candidate ATMs.

The model decides:
1. Which ATMs to visit.
2. How much cash to deliver.
3. The order of visits.
4. How to trade off travel, cash carried, and shortage risk.

It includes:
- truck capacity = 25 lakh
- maximum 10 visits
- priority-3 mandatory ATMs
- low/normal/high demand scenarios
- scenario probabilities
- shortage variables
- 09:00–17:00 time windows
- 8-hour route limit
- MTZ subtour elimination

Run
---
pip install pandas gurobipy

Then place the .py and .csv in the same folder and run:

python atm_cash_routing_gurobi.py

A valid Gurobi license is required.

Important
---------
All ATM locations, cash balances, demands, capacities, priorities, and costs are synthetic.
They must be described as synthetic data in an academic submission.

The report deliberately does not invent a Gurobi optimum because Gurobi is not installed
in the current execution environment. Run the supplied code to obtain the numerical result.
