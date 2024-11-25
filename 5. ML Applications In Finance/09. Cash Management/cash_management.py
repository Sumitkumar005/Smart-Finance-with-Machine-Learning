"""
Python script for calculating optimal cash reserves and investment allocation.
We'll use Linear Programming (via PuLP library) to optimize the allocation 
based on constraints such as emergency cash needs, expected returns, and risk tolerance.
"""

# Importing required librariess
import pandas as pd
import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

# Step 1: Define inputs and parameters
"""
These are hypothetical values and can be replaced with real data.

- Total cash available
- Minimum cash reserve required
- Expected returns for different investment options
- Risk tolerance for each investment type
"""
total_cash = 100000  # Total cash available ($)
min_cash_reserve = 20000  # Minimum amount to keep as cash reserves ($)

# Investment options with expected returns (in % per year) and risk scores (1 = low, 5 = high)
investment_options = {
    'stocks': {'expected_return': 8, 'risk': 4},
    'bonds': {'expected_return': 5, 'risk': 2},
    'real_estate': {'expected_return': 7, 'risk': 3},
    'gold': {'expected_return': 6, 'risk': 2},
}

# Risk tolerance (maximum average risk allowed)
max_risk_tolerance = 3

# Step 2: Initialize the optimization problem
"""
The goal is to maximize expected returns while meeting the constraints:
- Keep a minimum cash reserve.
- Maintain risk within the specified tolerance.
"""
problem = LpProblem("Optimal_Cash_And_Investment", LpMaximize)

# Step 3: Define decision variables
"""
Each investment type will have a decision variable representing the amount allocated to it.
"""
decision_variables = {name: LpVariable(name, lowBound=0, cat='Continuous') 
                      for name in investment_options}

cash_reserve = LpVariable("cash_reserve", lowBound=min_cash_reserve, cat='Continuous')

# Step 4: Define the objective function
"""
Maximize the total expected returns from investments.
"""
expected_returns = lpSum([
    decision_variables[name] * (investment['expected_return'] / 100)
    for name, investment in investment_options.items()
])
problem += expected_returns, "Total_Expected_Returns"

# Step 5: Add constraints
"""
1. The sum of cash reserves and investments must equal total cash available.
2. The average risk of the portfolio must be within the risk tolerance.
"""
# Total cash constraint
problem += cash_reserve + lpSum(decision_variables.values()) == total_cash, "Total_Cash_Constraint"

# Risk tolerance constraint
weighted_risk = lpSum([
    decision_variables[name] * investment['risk']
    for name, investment in investment_options.items()
])
total_investment = lpSum(decision_variables.values())
problem += weighted_risk / (total_investment + 1e-5) <= max_risk_tolerance, "Risk_Tolerance_Constraint"

# Step 6: Solve the problem
problem.solve()

# Step 7: Display the results
print("\nOptimization Results:")
if problem.status == 1:  # Check if the solution is optimal
    print("Optimal allocation found!")
    print(f"Cash Reserves: ${cash_reserve.varValue:.2f}")
    for name, var in decision_variables.items():
        print(f"Investment in {name.capitalize()}: ${var.varValue:.2f}")
    print(f"Total Expected Returns: ${problem.objective.value():.2f}")
else:
    print("No optimal solution found. Check the constraints or input data.")

# Optional: Visualizing the allocation
import matplotlib.pyplot as plt

allocation = [cash_reserve.varValue] + [var.varValue for var in decision_variables.values()]
labels = ["Cash Reserve"] + [name.capitalize() for name in investment_options.keys()]

plt.figure(figsize=(8, 6))
plt.pie(allocation, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title("Optimal Cash and Investment Allocation")
plt.show()

"""
Summary of the Code:

1. Inputs: The script starts by defining the available cash, investment options, and constraints (e.g., minimum cash reserve, risk tolerance).
2. Optimization Problem: The problem is modeled using Linear Programming to maximize returns while adhering to the constraints.
3. Outputs: The optimal allocation is displayed and optionally visualized using a pie chart.
4. Customization: Replace the inputs with real data to make the script specific to your project.

Real-World Use Case: This script can be used by financial planners to balance cash reserves and investments based on client preferences for safety and growth.
"""
