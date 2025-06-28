# Code developed by Nousi Marina for the modeling and solution of the problem using Pyomo

# Import necessary libraries
import pyomo.environ as pyomo  # Modeling library for optimization problems
import numpy as np  # Library for arrays and numerical computations
import argparse  # Helps in parsing command-line arguments
import os  # For operating system operations (files, directories)
import csv  # For reading/writing CSV files

# Define constants
TIME_LIMIT = 3600  # Maximum execution time (in seconds)
F = [2, 3]  # Types of chargers (Level 2 and Level 3)
c_level = [22, 100]  # Power of chargers in kW (Level 2 -> 22 kW, Level 3 -> 100 kW)
D = 5  # Maximum distance a customer can travel to reach a facility


# Function to read data from a file
def read_data(filename):
    f = open(filename, "rt")
    lines = f.readlines()
    lines = [f.replace("\n", "") for f in lines if f != "\n"]

    # Read basic parameters P, N, CL
    line = lines[0].split(" ")
    P = int(line[3])  # Number of facilities that must be opened
    N = int(line[2])  # Number of potential installation sites
    CL = int(line[1])  # Number of clients
    line_idx = 2

    # Customer demand at each potential location
    clients = np.zeros(CL)
    for i in range(CL):
        clients[i] = int(lines[line_idx])
        line_idx += 1

    # Points considered for installation
    line_idx += 1
    line = lines[line_idx]
    points = np.zeros(N)
    for i in range(N):
        points[i] = int(lines[line_idx])
        line_idx += 1

    # Number of charger positions at each location
    line_idx += 1
    place = np.zeros(N, dtype=int)
    for i in range(N):
        place[i] = int(lines[line_idx])
        line_idx += 1

    # Available power per location
    line_idx += 1
    capacity = np.zeros(N, dtype=int)
    for i in range(N):
        capacity[i] = int(lines[line_idx])
        line_idx += 1

    # Unary constraints per facility
    line_idx += 1
    unaryConstraint = np.zeros(P)
    for i in range(P):
        unaryConstraint[i] = lines[line_idx].split(" ")[1]
        line_idx += 1

    # Binary constraints between facilities
    line_idx += 1
    line = lines[line_idx]
    binaryConstraint = np.zeros(shape=(P, P))
    for i in range(P - 1):
        for j in range(i + 1, P):
            line = lines[line_idx].split(" ")
            binaryConstraint[i][j] = float(line[2])
            binaryConstraint[j][i] = float(line[2])
            line_idx += 1

    # Distance between installation points
    line_idx += 1
    line = lines[line_idx]
    fp_euc_distances = np.zeros(shape=(N, N))
    fp_sp_distances = np.zeros(shape=(N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            line = lines[line_idx].split(" ")
            fp_sp_distances[i][j] = float(line[2])
            fp_euc_distances[i][j] = float(line[3])
            line_idx += 1

    # Customer-to-installation point distances
    line_idx += 1
    line = lines[line_idx]
    cl_euc_distances = np.zeros(shape=(CL, N))
    cl_sp_distances = np.zeros(shape=(CL, N))
    for i in range(CL):
        for j in range(N):
            line = lines[line_idx].split(" ")
            cl_sp_distances[i][j] = float(line[2])
            cl_euc_distances[i][j] = float(line[3])
            line_idx += 1

    return (
        P,
        N,
        CL,
        clients,
        points,
        place,
        capacity,
        unaryConstraint,
        binaryConstraint,
        fp_sp_distances,
        fp_euc_distances,
        cl_euc_distances,
        cl_sp_distances,
    )


# Function to solve the optimization model
def run_model(filename):
    # Read data
    (
        P,
        N,
        CL,
        clients,
        points,
        place,
        capacity,
        unaryConstraint,
        binaryConstraint,
        fp_sp_distances,
        fpEucDistances,
        clEucDistances,
        clSpDistances,
    ) = read_data(filename)

    # Create Pyomo model
    model = pyomo.ConcreteModel()

    # Create range sets for variables
    x_i_var_range = pyomo.RangeSet(0, N - 1)  # Potential installation sites
    x_j_var_range = pyomo.RangeSet(0, len(F) - 1)  # Charger types
    y_k_var_range = pyomo.RangeSet(0, CL - 1)  # Customers
    y_i_var_range = pyomo.RangeSet(0, N - 1)  # Potential installation sites
    z_i_var_range = pyomo.RangeSet(0, N - 1)  # Potential installation sites

    # Create variables
    model.x = pyomo.Var(
        x_i_var_range, x_j_var_range, domain=pyomo.Integers, bounds=(0, 5)
    )  # Integer variable, number of chargers per type and location
    model.y = pyomo.Var(
        y_k_var_range, y_i_var_range, domain=pyomo.Binary
    )  # Binary variable, indicates if customer k is served by facility i
    model.z = pyomo.Var(
        z_i_var_range, domain=pyomo.Binary
    )  # Binary variable, indicates if a facility is open at site i

    # Add constraints list
    model.c = pyomo.ConstraintList()

    # Constraint 1: Max number of chargers per location
    expr = 0
    for i in range(N):
        for j in range(len(F)):
            expr += model.x[i, j]
        model.c.add(expr <= place[i])
        expr = 0

    # Constraint 2: Power limit per installation
    expr = 0
    for i in range(N):
        for j in range(len(F)):
            expr += c_level[j] * model.x[i, j]
        model.c.add(expr <= capacity[i])
        expr = 0

    # Constraint 3: Open installation must have at least one charger
    expr = 0
    for i in range(N):
        for j in range(len(F)):
            expr += model.x[i, j]
        model.c.add(expr >= model.z[i])
        expr = 0

    # Constraint 4: No chargers unless installation is open
    expr = 0
    for i in range(N):
        for j in range(len(F)):
            expr += model.x[i, j]
        model.c.add(expr <= place[i] * model.z[i])
        expr = 0

    # Constraint 5: Exactly P facilities must be opened
    expr = 0
    for i in range(N):
        expr += model.z[i]
    model.c.add(expr == P)

    # Constraint 6: Each customer must be served by exactly one installation
    expr = 0
    for k in range(CL):
        for i in range(N):
            expr += model.y[k, i]
        model.c.add(expr == 1)
        expr = 0

    # Constraint 7: A customer can only be served by an open installation
    for k in range(CL):
        for i in range(N):
            expr = model.y[k, i]
            model.c.add(expr <= model.z[i])
            expr = 0

    # Constraint 8: Close proximity installations cannot be opened simultaneously
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            for p in range(P - 1):
                for l in range(p + 1, P):
                    if fpEucDistances[i][j] <= binaryConstraint[p][l]:
                        model.c.add(model.z[i] + model.z[j] <= 1)

    # Constraint 9: Customer-installation distance must be <= D
    for k in range(CL):
        for i in range(N):
            model.c.add(clEucDistances[k][i] * model.y[k, i] <= D)

    # Constraint 10: Charger availability must meet customer demand
    expr = 0
    for i in range(N):
        expr = 12 * model.x[i, 0] + 48 * model.x[i, 1]
        rhs = sum(model.y[k, i] for k in range(CL))
        model.c.add(expr >= rhs)
        expr = 0
        rhs = 0

    # Constraint 11: Total cost must not exceed €60,000
    expr = 0
    for i in range(N):
        expr = expr + 100 * model.z[i] + 2000 * model.x[i, 0] + 6000 * model.x[i, 1]
    model.c.add(expr <= 60000)

    # Define cost matrix for the objective function (customer-to-node distances)
    model.obj_coeff = np.zeros((CL, N))

    for i in range(CL):
        for j in range(N):
            model.obj_coeff[i, j] = clSpDistances[i][j]  # shortest path

    # Objective function: Minimize total customer service distance
    def obj_expression(m):
        return pyomo.summation(m.obj_coeff, m.y)

    model.objective = pyomo.Objective(sense=pyomo.minimize, rule=obj_expression)

    # Solve the model using Gurobi
    solver = pyomo.SolverFactory("gurobi_persistent")
    solver.set_instance(model)

    # Set timeout of 1 hour for Gurobi
    solver.options["TimeLimit"] = TIME_LIMIT

    # Solve/Model the model
    results = solver.solve(tee=True)

    # Print non-zero variable values
    print("Print values for non-zero variables")

    # Access all variables of our model
    for v in model.component_data_objects(pyomo.Var):
        if v.value is not None and v.value != 0:
            print(str(v), v.value)

    # Print objective value
    print(f"\nObjective value = {pyomo.value(model.objective)}")

    # Check if the model is infeasible
    if results.solver.status == pyomo.SolverStatus.ok:
        if results.solver.termination_condition == pyomo.TerminationCondition.optimal:
            print("Model solved to optimality")
            print(f"Objective value = {pyomo.value(model.objective)}\n\n")
            return (
                results.solver.status,
                results.solver.termination_condition,
                pyomo.value(model.objective),
                results.solver.wallclock_time,
            )
        else:
            print(
                "Solver terminated with condition:",
                results.solver.termination_condition,
            )
            print(f"Objective value = {pyomo.value(model.objective)}\n\n")
            return (
                results.solver.status,
                results.solver.termination_condition,
                pyomo.value(model.objective),
                results.solver.wallclock_time,
            )
    else:
        if (
            results.solver.termination_condition
            == pyomo.TerminationCondition.infeasible
        ):
            print("Model is infeasible\n\n")
            return (
                results.solver.status,
                results.solver.termination_condition,
                -1,
                results.solver.wallclock_time,
            )
        else:
            print("Solver status:\n\n", results.solver.status)
            return (
                results.solver.status,
                results.solver.termination_condition,
                -1,
                results.solver.wallclock_time,
            )


# Function that handles the input arguments from the user
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Facility Location Problem Generator\nUsage: <grid size> <total points (N)> <clients (CL)> <facilities (P)> <# problems>"
    )

    # Define each positional argument
    parser.add_argument("dir", type=str, help="A directory constaining problems")
    parser.add_argument(
        "--problems", nargs="*", type=str, help="Solve specific problems"
    )

    # Parse arguments
    args = parser.parse_args()

    return args


# Class to store problem results
class Problem_Results:
    def __init__(
        self,
        name="",
        objective=-1,
        status=None,
        termination_condition=None,
        solving_time=0,
    ):
        self.name = name
        self.objective = objective
        self.status = status
        self.termination_condition = termination_condition
        self.solving_time = solving_time


# To use the parser
if __name__ == "__main__":
    print("\n\n")
    problem_results = list()

    # Handle input arguments
    args = parse_arguments()

    dir = args.dir

    # Solve specific problems in directory, passed as arguments
    if args.problems is not None:
        problems = args.problems

        for problem in problems:
            try:
                print(f"****************Solving problem {problem}****************\n\n")
                st, tc, obj, solving_time = run_model(dir + problem)
                problem_results.append(
                    Problem_Results(problem, obj, st, tc, solving_time)
                )
            except Exception as e:
                print(f"{str(e)}\n\n")

    # Solve all problems in directory
    else:
        for name in sorted(os.listdir(dir)):
            print(f"****************Solving problem {name}****************\n\n")
            st, tc, obj, solving_time = run_model(dir + name)
            problem_results.append(Problem_Results(name, obj, st, tc, solving_time))

    infeasible = ""
    feasible = ""
    aborted = ""
    objectives = list()
    solving_time = list()

    # Store results for feasible problems and mark the infeasible and aborted ones.
    for pr in problem_results:
        if pr.status == pyomo.SolverStatus.ok:
            feasible += pr.name + ", "
            objectives.append(pr.objective)
            solving_time.append(pr.solving_time)
            print(solving_time)
        else:
            if pr.termination_condition == pyomo.TerminationCondition.infeasible:
                infeasible += pr.name + ", "
            else:
                aborted += pr.name + ", "

    infeasible = infeasible[:-2]
    feasible = feasible[:-2]
    aborted = aborted[:-2]

    # Calculate mean value of the objective values and mean run time
    mean_obj = np.mean(objectives)
    sum_solving_time = np.mean(solving_time)

    # Create/update results file
    file_name = "results.txt"

    # Write results in CSV format
    with open(file_name, mode="a", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([mean_obj, sum_solving_time])

    # Print results
    print(
        "\n\n=============================== RESULTS ==================================\n\n"
    )
    print(f"Average objective value for all feasible problems: {round(mean_obj, 3)}.")
    print(
        f"Average run time for all feasible problems: {round(sum_solving_time,3)} s.\n"
    )
    print(f"\nInfeasible problems: {infeasible}")
    print(f"Feasible problems: {feasible}")
    print(f"Aborted problems: {aborted}")
    print()
    print(
        "==========================================================================\n\n"
    )
