# Code developed by Nousi Marina for the p-median problem, implementing a Gurobi-based optimization model

# Import necessary libraries
import numpy as np  # Library for handling arrays, matrices and numerical computations
import gurobipy as gp  # Main Gurobi module used to create and solve optimization models
from gurobipy import GRB  # Imports Gurobi constants and enums
import os  # For operating system operations (files, directories)
import sys  # Provides access to system-specific parameters and functions
from model import read_data  # Function to read input data from files

# Define constants
TIME_LIMIT = 3600  # Maximum execution time (in seconds)
F = [2, 3]  # Types of chargers (Level 2 and Level 3)
c_level = [22, 100]  # Power of chargers in kW (Level 2 -> 22 kW, Level 3 -> 100 kW)
D = 5  # Maximum distance a customer can travel to reach a facility


# Creates a model for the pmedian problem for gurobipy
def pmedian(filename):

    # Read data
    (
        P,
        N,
        CL,
        place,
        capacity,
        binaryConstraint,
        fpEucDistances,
        clEucDistances,
        clSpDistances,
    ) = read_data(filename)

    num_vars = N * len(F) + CL * N + N  # total number of variables
    model = gp.Model()  # create Gurobi model

    integer_count = N * len(F)  # number of integer variables xij
    ub = [
        5 if i < integer_count else 1 for i in range(num_vars)
    ]  # the first N * len(F) variables are integer, rest of them are binary
    lb = [0 for i in range(num_vars)]  # all variables have lowewr bound = 0

    # Create variables
    x = model.addVars(num_vars, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="x")

    # Constraint 1: Max number of chargers per location
    expr = 0
    for i in range(N):
        for j in range(len(F)):
            expr += x[i * len(F) + j]
        model.addLConstr(expr <= place[i])
        expr = 0

    # Constraint 2: Power limit per installation
    expr = 0
    for i in range(N):
        for j in range(len(F)):
            expr += c_level[j] * x[i * len(F) + j]
        model.addLConstr(expr <= capacity[i])
        expr = 0

    # Constraint 3: Open installation must have at least one charger
    expr = 0
    for i in range(N):
        for j in range(len(F)):
            expr += x[i * len(F) + j]
        model.addLConstr(expr >= x[N * len(F) + CL * N + i])
        expr = 0

    # Constraint 4: No chargers unless installation is open
    expr = 0
    for i in range(N):
        for j in range(len(F)):
            expr += x[i * len(F) + j]
        model.addLConstr(expr <= place[i] * x[N * len(F) + CL * N + i])
        expr = 0

    # Constraint 5: Exactly P facilities must be opened
    expr = 0
    for i in range(N):
        expr += x[N * len(F) + CL * N + i]
    model.addLConstr(expr == P)
    expr = 0

    # Constraint 6: Each customer must be served by exactly one installation
    expr = 0
    for k in range(CL):
        for i in range(N):
            expr += x[N * len(F) + k * N + i]
        model.addLConstr(expr == 1)
        expr = 0

    # Constraint 7: A customer can only be served by an open installation
    for k in range(CL):
        for i in range(N):
            model.addLConstr(x[N * len(F) + k * N + i] <= x[N * len(F) + CL * N + i])

    # Constraint 8: Close proximity installations cannot be opened simultaneously
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            for p in range(P - 1):
                for l in range(p + 1, P):
                    if fpEucDistances[i][j] <= binaryConstraint[p][l]:
                        model.addLConstr(
                            x[N * len(F) + CL * N + i] + x[N * len(F) + CL * N + j] <= 1
                        )

    # Constraint 9: Customer-installation distance must be <= D
    for k in range(CL):
        for i in range(N):
            model.addLConstr(clEucDistances[k][i] * x[N * len(F) + k * N + i] <= D)

    # Constraint 10: Charger availability must meet customer demand
    for i in range(N):
        expr = 12 * x[i * len(F)] + 48 * x[i * len(F) + 1]
        rhs = 0
        for k in range(CL):
            rhs += x[N * len(F) + k * N + i]
        model.addLConstr(expr >= rhs)

    # Constraint 11: Total cost must not exceed €60,000
    expr = 0
    for i in range(N):
        expr += (
            100 * x[N * len(F) + CL * N + i]
            + 2000 * x[i * len(F)]
            + 6000 * x[len(F) * i + 1]
        )
    model.addLConstr(expr <= 60000)

    # Create vector c by flattening the 2D list clSpDistances (dimensions: CL x N)
    c = [clSpDistances[i][j] for i in range(CL) for j in range(N)]

    # Define the objective function in Gurobi
    model.setObjective(
        gp.quicksum(
            c[k * N + i] * x[N * len(F) + k * N + i]
            for k in range(CL)
            for i in range(N)
        ),
        GRB.MINIMIZE,
    )

    model.Params.method = 1  # 1 indicates the dual Simplex algorithm in Gurobi
    model.update()  # Update the model to incorporate recent changes

    # Define which variables are integer
    integer_var = [True for i in range(num_vars)]

    return model, ub, lb, integer_var, num_vars, c


# Create a function to process all files in a folder
def process_all_files_in_folder(folder_path):
    # Loop through all files in the specified folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if it's a .txt file
        if os.path.isfile(file_path) and file_path.endswith(".txt"):
            print(f"Processing file: {file_path}")
            pmedian(file_path)


if __name__ == "__main__":
    # Check if exactly one argument (folder path) was provided
    if len(sys.argv) != 2:
        print("Please provide the folder path as an argument.")
        sys.exit(1)

    # Get the folder path from command-line arguments
    folder_path = sys.argv[1]
    # Check if the provided path is a valid directory
    if not os.path.isdir(folder_path):
        print(f"The path {folder_path} is not a valid directory.")
        sys.exit(1)

    # Process all .txt files in the folder
    process_all_files_in_folder(folder_path)
