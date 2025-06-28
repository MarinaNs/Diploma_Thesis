# Code developed by Marina Nousi implementing Branch and Bound with DFS and Minimum Distance heuristic.

# Import necessary libraries
import gurobipy as gp  # Used for creating and solving optimization models
from gurobipy import GRB  # Imports constants and enums from Gurobi
import numpy as np  # For numerical operations and array/matrix handling
import time  # For tracking execution time
import problems as pr  # Custom module that defines the p-median optimization model
from model import read_data  # Function to read input data from files
import os  # For interacting with the file system (directories, paths)
import sys  # Access to system-specific parameters and command-line arguments
import csv  # For reading and writing CSV files

F = [2, 3]  # Types of chargers (Level 2 and Level 3)

# Define global variables
isMax = None  # Indicates whether the optimization problem is a maximization (True) or minimization (False)
DEBUG_MODE = True  # Flag to enable or disable debug mode
nodes = 0  # Counter to track the number of nodes explored in the branch-and-bound tree
lower_bound = -np.inf  # Initial value for the global lower bound
upper_bound = np.inf  # Initial value for the global upper bound


# Returns True if the value is very close to an integer (within the specified tolerance)
def is_nearly_integer(value, tolerance=1e-6):
    return abs(value - round(value)) <= tolerance


# Α class 'Node' that holds information of a node
class Node:
    def __init__(self, ub, lb, depth, vbasis, cbasis, branching_var, label=""):
        self.ub = ub  # Upper bounds of variables at this node
        self.lb = lb  # Lower bounds of variables at this node
        self.depth = depth  # Depth of this node in the search tree
        self.vbasis = vbasis  # Variable basis information
        self.cbasis = cbasis  # Constraint basis information
        self.branching_var = (
            branching_var  # Variable used to branch and create child nodes
        )
        self.label = label  # Ιdentifier for the node


# Print debugging info
def debug_print(node: Node = None, x_obj=None, sol_status=None):
    print("\n\n-----------------  DEBUG OUTPUT  -----------------\n\n")
    print(f"UB:{upper_bound}")  # Current upper bound
    print(f"LB:{lower_bound}")  # Current lower bound
    if node is not None:
        print(
            f"Branching Var: {node.branching_var}"
        )  # Branching variable at the current node
    if node is not None:
        print(f"Child: {node.label}")  # Label of the current node
    if node is not None:
        print(f"Depth: {node.depth}")  # Depth of the current node
    if x_obj is not None:
        print(
            f"Simplex Objective: {x_obj}"
        )  # Objective value of the current simplex solution
    if sol_status is not None:
        print(f"Solution status: {sol_status}")  # Status of the current solution

    print("\n\n--------------------------------------------------\n\n")


# Function to calculate the total distance from a given location (idx) to all customers
def calculate_total_distance(idx, cl_sp_distances):
    total_distance = 0
    for cl in range(len(cl_sp_distances)):  # Iterate over all customers
        total_distance += cl_sp_distances[
            cl, idx
        ]  # Sum the distance from customer cl to location idx
    return total_distance  # Return the total accumulated distance


# Branch & Bound algorithm
def branch_and_bound(
    model,
    ub,
    lb,
    integer_var,
    best_bound_per_depth,
    nodes_per_depth,
    vbasis=[],
    cbasis=[],
    depth=0,
):
    global nodes, lower_bound, upper_bound

    # Create stack (LIFO)
    stack = []

    # Initialize solution list
    solutions = list()
    solutions_found = 0  # Counter for the number of solutions found
    best_sol_idx = 0  # Index of the best solution found so far

    # Initialize best solution
    if isMax:
        best_sol_obj = (
            -np.inf
        )  # For maximization, start with worst possible (negative infinity)
    else:
        best_sol_obj = (
            np.inf
        )  # For minimization, start with worst possible (positive infinity)

    # Create root node
    root_node = Node(ub, lb, depth, vbasis, cbasis, -1, "root")
    nodes_per_depth[
        0
    ] -= 1  # Each time a node is processed, decrement the number of remaining nodes at the current depth by one

    # ===============  Root node  ==========================
    if DEBUG_MODE:
        debug_print()

    # Solve relaxed problem
    model.optimize()

    # Check if the model was solved to optimality. If not then return (infeasible)
    if model.status != GRB.OPTIMAL:
        if isMax:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return [], -np.inf, depth
        else:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return [], np.inf, depth

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
        fp_euc_distances,
        cl_euc_distances,
        cl_sp_distances,
    ) = read_data(file_path)

    # Get the solution (variable assignments)
    x_candidate = model.getAttr("X", model.getVars())

    # Restrict branching to the subset of variables representing facility openings (z)
    start_idx = N * len(F) + CL * N
    end_idx = start_idx + N

    # Get the objective value
    x_obj = model.ObjVal
    best_bound_per_depth[0] = x_obj

    # Verify whether all z variables have taken integer values; if not, choose the one with the smallest total distance
    # to all customers as the branching variable (Min Distance Heuristic)
    vars_have_integer_vals = True
    min_total_distance = float("inf")  # Initialize the minimum total distance

    for idx in range(start_idx, end_idx):
        if integer_var[idx] and not is_nearly_integer(x_candidate[idx]):
            vars_have_integer_vals = False

            # Compute total distance of the current variable to all customers using the calculate_total_distance function
            total_distance = calculate_total_distance(idx - start_idx, cl_sp_distances)

            # Update the branching variable to the one with the minimum total distance to customers
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                selected_var_idx = idx

    # If all integer variables are indeed integers, we've found a feasible solution
    if vars_have_integer_vals:
        # if we have a feasible solution in root, then terminate
        solutions.append([x_candidate, x_obj, depth])
        solutions_found += 1  # Increment the solution counter

        if DEBUG_MODE:  # Print debug information for this solution
            debug_print(node=root_node, x_obj=x_obj, sol_status="Integer")
        return solutions, best_sol_idx, solutions_found

    # If the solution is fractional, update bounds accordingly based on optimization type
    else:
        if isMax:
            upper_bound = x_obj
        else:
            lower_bound = x_obj

    if DEBUG_MODE:
        debug_print(node=root_node, x_obj=x_obj, sol_status="Fractional")

    # Retrieve vbasis and cbasis
    vbasis = model.getAttr("VBasis", model.getVars())
    cbasis = model.getAttr("CBasis", model.getConstrs())

    # Create lower bounds and upper bounds for the variables of the child nodes
    left_lb = np.copy(lb)
    left_ub = np.copy(ub)
    right_lb = np.copy(lb)
    right_ub = np.copy(ub)

    # Create left and right branches (e.g. set left: x = 0, right: x = 1 in a binary problem)
    left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
    right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

    # Create child nodes
    left_child = Node(
        left_ub,
        left_lb,
        root_node.depth + 1,
        vbasis.copy(),
        cbasis.copy(),
        selected_var_idx,
        "Left",
    )
    right_child = Node(
        right_ub,
        right_lb,
        root_node.depth + 1,
        vbasis.copy(),
        cbasis.copy(),
        selected_var_idx,
        "Right",
    )

    # Add child nodes in stack
    stack.append(left_child)
    stack.append(right_child)

    # Solving sub problems
    while len(stack) != 0:
        print(
            "\n********************************  NEW NODE BEING EXPLORED  ******************************** "
        )

        # Increment the total number of explored nodes by 1
        nodes += 1

        # Retrieve the current node from the top of the stack (LIFO)
        current_node = stack[-1]

        # Remove the node from the top of the stack
        stack.pop()

        # Decrease the number of remaining nodes to be explored at the current depth
        nodes_per_depth[current_node.depth] -= 1

        # Warm start solver. Use the vbasis and cbasis that parent node passed to the current one
        if (len(current_node.vbasis) != 0) and (len(current_node.cbasis) != 0):
            model.setAttr("VBasis", model.getVars(), current_node.vbasis)
            model.setAttr("CBasis", model.getConstrs(), current_node.cbasis)

        # Update the state of the model, passing the new lower bounds/upper bounds for the vars. Basically, we only change the ub/lb for the branching variable
        model.setAttr("LB", model.getVars(), current_node.lb)
        model.setAttr("UB", model.getVars(), current_node.ub)
        model.update()

        if DEBUG_MODE:
            debug_print()

        # Optimize the model
        model.optimize()

        # Check if the problem is infeasible or was not solved successfully. In such a case, no child nodes will be generated
        infeasible = False
        if model.status != GRB.OPTIMAL:
            if isMax:
                infeasible = True
                x_obj = (
                    -np.inf
                )  # Set objective value to negative infinity for infeasible maximization problems
            else:
                infeasible = True
                x_obj = (
                    np.inf
                )  # Set objective value to positive infinity for infeasible minimization problems

            # For all levels deeper than the current one, decrease the expected number of nodes,  as child nodes will not be generated due to infeasibility
            for i in range(current_node.depth + 1, len(nodes_per_depth)):
                nodes_per_depth[i] -= 2 * (i - current_node.depth)

        else:
            # Store the values of the decision variables from the current solution
            x_candidate = model.getAttr("X", model.getVars())

            # Store the objective value of the current solution
            x_obj = model.ObjVal

            # Update best bound per depth if a better solution was found
            if isMax == True and x_obj > best_bound_per_depth[current_node.depth]:
                best_bound_per_depth[current_node.depth] = x_obj

            elif isMax == False and x_obj < best_bound_per_depth[current_node.depth]:
                best_bound_per_depth[current_node.depth] = x_obj

            # If we reached the final node of a depth, then update the bounds
            if nodes_per_depth[current_node.depth] == 0:
                # Set upper or lower bound based on the best objective value found at this depth
                if isMax == True:
                    upper_bound = best_bound_per_depth[current_node.depth]
                else:
                    lower_bound = best_bound_per_depth[current_node.depth]

        # If the solution is infeasible, skip generating child nodes and continue with the next node
        if infeasible:
            if DEBUG_MODE:
                debug_print(node=current_node, sol_status="Infeasible")
            continue

        # Verify whether all z variables have taken integer values; if not, choose the one with the smallest total distance
        # to all customers as the branching variable (Min Distance Heuristic)
        vars_have_integer_vals = True
        min_total_distance = float("inf")  # Initialize the minimum total distance

        for idx in range(start_idx, end_idx):
            if integer_var[idx] and not is_nearly_integer(x_candidate[idx]):

                vars_have_integer_vals = False

                # Compute total distance of the current variable to all customers using the calculate_total_distance function
                total_distance = calculate_total_distance(
                    idx - start_idx, cl_sp_distances
                )

                # Update the branching variable to the one with the minimum total distance to customers
                if total_distance < min_total_distance:
                    min_total_distance = total_distance
                    selected_var_idx = idx

        # Found feasible solution
        if vars_have_integer_vals:
            if isMax:
                if lower_bound < x_obj:  # A better solution was found
                    lower_bound = x_obj  # Update lower bound
                    if abs(lower_bound - upper_bound) < 1e-6:  # Optimal solution
                        # Store solution, number of solutions and best sol index (and return)
                        solutions.append([x_candidate, x_obj, current_node.depth])
                        solutions_found += 1
                        if (abs(x_obj - best_sol_obj) < 1e-6) or solutions_found == 1:
                            best_sol_obj = x_obj
                            best_sol_idx = solutions_found - 1

                            if DEBUG_MODE:
                                debug_print(
                                    node=current_node,
                                    x_obj=x_obj,
                                    sol_status="Integer/Optimal",
                                )

                        return solutions, best_sol_idx, solutions_found

                    # Not optimal. Store solution, number of solutions and best sol index (and do not expand children)
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1

                    # Remove the children nodes from each next depth
                    for i in range(current_node.depth + 1, len(nodes_per_depth)):
                        nodes_per_depth[i] -= 2 * (i - current_node.depth)

                    if DEBUG_MODE:
                        debug_print(
                            node=current_node, x_obj=x_obj, sol_status="Integer"
                        )
                    continue

            else:
                if upper_bound > x_obj:  # Found better solution
                    upper_bound = x_obj  # Update bound

                    if (
                        abs(lower_bound - upper_bound) < 1e-6
                    ):  # Check if optimality reached
                        # Store solution, number of solutions and best sol index (and return)
                        solutions.append([x_candidate, x_obj, current_node.depth])
                        solutions_found += 1
                        if x_obj < best_sol_obj or solutions_found == 1:
                            best_sol_obj = x_obj
                            best_sol_idx = solutions_found - 1

                            if DEBUG_MODE:
                                debug_print(
                                    node=current_node,
                                    x_obj=x_obj,
                                    sol_status="Integer/Optimal",
                                )

                        return solutions, best_sol_idx, solutions_found

                    # Not optimal. Store solution, number of solutions and best sol index (and do not expand children)
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    if x_obj < best_sol_obj or solutions_found == 1:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1

                    # Remove the children nodes from each next depth
                    for i in range(current_node.depth + 1, len(nodes_per_depth)):
                        nodes_per_depth[i] -= 2 * (i - current_node.depth)

                    if DEBUG_MODE:
                        debug_print(
                            node=current_node, x_obj=x_obj, sol_status="Integer"
                        )
                    continue

            # Do not branch further if is an equal solution and remove the children nodes from each next depth
            for i in range(current_node.depth + 1, len(nodes_per_depth)):
                nodes_per_depth[i] -= 2 * (i - current_node.depth)

            if DEBUG_MODE:
                debug_print(
                    node=current_node,
                    x_obj=x_obj,
                    sol_status="Integer (Rejected -- Doesn't improve incumbent)",
                )
            continue

        if isMax:
            if x_obj < lower_bound or abs(x_obj - lower_bound) < 1e-6:  # Cut
                # Remove the children nodes from each next depth
                for i in range(current_node.depth + 1, len(nodes_per_depth)):
                    nodes_per_depth[i] -= 2 * (i - current_node.depth)
                if DEBUG_MODE:
                    debug_print(
                        node=current_node,
                        x_obj=x_obj,
                        sol_status="Fractional -- Cut by bound",
                    )
                continue
        else:
            if x_obj > upper_bound or abs(x_obj - upper_bound) < 1e-6:  # Cut
                # Remove the children nodes from each next depth
                for i in range(current_node.depth + 1, len(nodes_per_depth)):
                    nodes_per_depth[i] -= 2 * (i - current_node.depth)
                if DEBUG_MODE:
                    debug_print(
                        node=current_node,
                        x_obj=x_obj,
                        sol_status="Fractional -- Cut by bound",
                    )
                continue

        if DEBUG_MODE:
            debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional")

        # Retrieve vbasis and cbasis
        vbasis = model.getAttr("VBasis", model.getVars())
        cbasis = model.getAttr("CBasis", model.getConstrs())

        # Create lower bounds and upper bounds for child nodes
        left_lb = np.copy(current_node.lb)
        left_ub = np.copy(current_node.ub)
        right_lb = np.copy(current_node.lb)
        right_ub = np.copy(current_node.ub)

        # create left and right branches  ( set left: x = 0, right: x = 1 )
        left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
        right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

        # Create child nodes
        left_child = Node(
            left_ub,
            left_lb,
            current_node.depth + 1,
            vbasis.copy(),
            cbasis.copy(),
            selected_var_idx,
            "Left",
        )
        right_child = Node(
            right_ub,
            right_lb,
            current_node.depth + 1,
            vbasis.copy(),
            cbasis.copy(),
            selected_var_idx,
            "Right",
        )

        # Add child nodes in stack
        stack.append(left_child)
        stack.append(right_child)

    return solutions, best_sol_idx, solutions_found


if __name__ == "__main__":
    folder_path = sys.argv[1]

    # Check if the specified folder exists
    if not os.path.isdir(folder_path):
        print(f"Ο φάκελος '{folder_path}' δεν υπάρχει.")
        sys.exit(1)

    # Get a list of all .txt files in the folder
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    # If no .txt files are found, exit the program
    if not txt_files:
        print(f"Δεν βρέθηκαν .txt αρχεία στον φάκελο '{folder_path}'.")
        sys.exit(1)

    # Define the name of the results file based on the folder name
    results_file = f"results_DFS_MinDistance_{os.path.basename(folder_path)}.txt"

    for file in txt_files:
        file_path = os.path.join(folder_path, file)
        print(f"\n========= Processing file: {file_path} =========")

        # Initialize the lower and upper bounds for the problem
        lower_bound = -np.inf
        upper_bound = np.inf

        # Create a new Gurobi model for each input file
        model = gp.Model()

        model.reset()

        # Load problem data and initialize variables from the current file
        model, ub, lb, integer_var, num_vars, c = pr.pmedian(file_path)
        isMax = False  # Problem is a minimization problem

        # Initialize best bounds per depth depending on problem type
        if isMax == True:
            best_bound_per_depth = np.array([-np.inf for i in range(num_vars)])
        else:
            best_bound_per_depth = np.array([np.inf for i in range(num_vars)])

        # Initialize an array to track the number of nodes per tree depth
        nodes_per_depth = np.zeros(num_vars + 1, dtype=float)
        nodes_per_depth[0] = 1
        for i in range(1, num_vars + 1):
            nodes_per_depth[i] = (
                nodes_per_depth[i - 1] * 2
            )  # Number of nodes doubles at each level of the search tree

        print(
            "************************    Solving problem...    ************************"
        )

        # Reset solutions and related variables before starting branch and bound
        solutions = []
        best_sol_idx = -1
        solutions_found = False
        optimal_solution = []
        best_z_values = []

        # Start timer for branch and bound process
        start_bb = time.time()

        # Run branch and bound algorithm to solve the problem
        solutions, best_sol_idx, solutions_found = branch_and_bound(
            model, ub, lb, integer_var, best_bound_per_depth, nodes_per_depth
        )

        # End timer and calculate branch and bound duration
        end_bb = time.time()
        time_bb = end_bb - start_bb

        # Read problem data from the input file
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
            fp_euc_distances,
            cl_euc_distances,
            cl_sp_distances,
        ) = read_data(file_path)

        # Calculate index range for z variables in the solution vector
        start_idx = N * len(F) + CL * N
        end_idx = start_idx + N

        # Extract the best integer solution found by branch and bound
        optimal_solution = solutions[int(best_sol_idx)][0]
        best_z_values = optimal_solution[start_idx:end_idx]

        # Fix the z variables in the model to the values from the best solution (set lower and upper bounds equal)
        for i in range(start_idx, end_idx):
            model.getVars()[i].LB = best_z_values[i - start_idx]
            model.getVars()[i].UB = best_z_values[i - start_idx]

        # Set all variables in the model to integer type before solving with Gurobi
        all_vars = model.getVars()
        int_types = [GRB.INTEGER] * len(all_vars)
        model.setAttr("VType", all_vars, int_types)

        # Update the model to apply variable type changes
        model.update()

        print("ΛΥΣΗ ΜΕ GUROBI\n")

        # Start timer for Gurobi optimization
        start_gurobi = time.time()

        # Optimize the model using Gurobi
        model.optimize()

        # End timer and calculate Gurobi solving time
        end_gurobi = time.time()
        time_gurobi = end_gurobi - start_gurobi

        # Calculate total solving time combining Βranch and Βound and Gurobi phases
        total_time = time_bb + time_gurobi

        # If an optimal solution is found, print the objective value and variable values
        if model.Status == 2:
            print("\n========= Optimal Solution Found =========")
            print(f"Objective Value: {model.ObjVal}")

            # Retrieve all variable values from the solution
            var_values = np.array([v.X for v in model.getVars()])

            # Reshape the variables according to problem structure: x, y, and z
            x_values = var_values[: N * len(F)].reshape(N, len(F))
            y_values = var_values[N * len(F) : N * len(F) + CL * N].reshape(CL, N)
            z_values = var_values[N * len(F) + CL * N :]

            print("Print values for nonzero variables")

            # Print nonzero x variable values
            print("x[N][len(F)]:")
            for i in range(N):
                for j in range(len(F)):
                    if abs(x_values[i, j]) > 1e-6:
                        print(f"x[{i}][{j}] = {x_values[i, j]}")

            # Print nonzero y variable values
            print("\ny[CL][N]:")
            for i in range(CL):
                for j in range(N):
                    if abs(y_values[i, j]) > 1e-6:
                        print(f"y[{i}][{j}] = {y_values[i, j]}")

            # Print nonzero z variable values
            print("\nz[N]:")
            for i in range(N):
                if abs(z_values[i]) > 1e-6:
                    print(f"z[{i}] = {z_values[i]}")

        else:
            # Print message if Gurobi did not find an optimal solution
            print("No optimal solution found. Status:", model.Status)

        # Write results to the output file
        with open(results_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            # If an optimal solution was found, log the objective value
            if model.Status == 2:
                writer.writerow([file_path, total_time, model.ObjVal])
            else:
                # Otherwise, note that no optimal solution was found
                writer.writerow([file_path, total_time, "No optimal solution"])

        print(f"Time Elapsed: {total_time}")
