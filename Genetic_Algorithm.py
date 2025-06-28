# Code developed by Marina Nousi implementing a genetic algorithm for the Charging Station Location Problem

# Import necessary libraries
import pygad  # Importing PyGAD library for implementing genetic algorithms
import random  # For stochastic behavior
import os  # For interacting with the file system (directories, paths)
import time  # For tracking execution time
import csv  # For reading and writing CSV files
import numpy as np  # For numerical operations and array/matrix handling
from model import (
    run_model,
)  # Function that runs the optimization model using Gurobi solver
from model import read_data  # Function to read input data from files


# Define constants
F = [2, 3]  # Types of chargers (Level 2 and Level 3)
budget_limit = 60000  # Budget constraint for the problem
c_level = [22, 100]  # Power of chargers in kW (Level 2 -> 22 kW, Level 3 -> 100 kW)
D = 5  # Maximum distance a customer can travel to reach a facility


# Check distance constraints between facilities (Constraint 8)
def check_distance(solution, binaryCstrs, fp_euc_distances, K):
    for i in range(K - 1):
        for j in range(i + 1, K):
            if fp_euc_distances[solution[i], solution[j]] <= binaryCstrs[i][j]:
                return False
    return True


# Function that checks partial feasibility
def partial_check_feasibility(
    solution,
    P,
    N,
    CL,
    F,
    D,
    place,
    capacity,
    c_level,
    binaryConstraint,
    clEucDistances,
    fpEucDistances,
):

    # Extract and reshape the solution vector into variables
    x = np.array(solution[: N * len(F)]).reshape((N, len(F)))
    y = np.array(solution[N * len(F) : N * len(F) + CL * N]).reshape((CL, N))
    z = np.array(solution[-N:])

    # Constraint 1: Max number of chargers per location
    for i in range(N):
        if z[i] == 1 and np.sum(x[i]) > place[i]:
            return False

    # Constraint 2: Power limit per installation
    for i in range(N):
        if z[i] == 1:
            power = sum(c_level[j] * x[i][j] for j in range(len(F)))
            if power > capacity[i]:
                return False

    # Constraint 3: Open installation must have at least one charger
    for i in range(N):
        if z[i] == 1 and np.sum(x[i]) < 1:
            return False

    # Constraint 4: No chargers unless installation is open
    for i in range(N):
        if z[i] == 0 and np.sum(x[i]) > 0:
            return False

    # Constraint 7: A customer can only be served by an open installation
    for k in range(CL):
        for i in range(N):
            if y[k][i] == 1 and z[i] == 0:
                return False

    # Constraint 9: Customer-installation distance must be <= D
    for k in range(CL):
        for i in range(N):
            if y[k][i] == 1 and clEucDistances[k][i] > D:
                return False

    # Constraint 10: Charger availability must meet customer demand
    for i in range(N):
        if z[i] == 1:
            capacity_coverage = 12 * x[i][0] + 48 * x[i][1]
            clients_served = sum(y[k][i] for k in range(CL))
            if capacity_coverage < clients_served:
                return False

    # Constraint 11: Total cost must not exceed €60,000
    total_cost = 0
    for i in range(N):
        total_cost += 100 * z[i] + 2000 * x[i][0] + 6000 * x[i][1]
    if total_cost > 60000:
        return False

    return True  # Return True if all constraints are satisfied, indicating the solution is feasible


# Function which implements a myopic heuristic to select P facility locations and assign clients with distance and capacity constraints
def myopic_heuristic(
    P,
    N,
    CL,
    binaryConstraint,
    fp_euc_distances,
    cl_euc_distances,
    cl_sp_distances,
    F,
    D,
    place,
    capacity,
    c_level,
):

    z = np.zeros(N, dtype=int)  # Initialize binary variables for open facilities
    x = np.zeros(
        (N, len(F)), dtype=int
    )  # Initialize number of chargers of each type per location
    y = np.zeros((CL, N), dtype=int)  # Initialize client-to-facility assignment matrix
    tempsol = np.zeros(
        P, dtype=int
    )  # Temporary array to store chosen facility indices for P facilities
    used_values = []  # Track locations already selected to avoid repetition

    for p in range(P):  # Iterate to select each of the P facilities
        best_cost = (
            np.inf
        )  # Track the best cost found for the current facility placement
        best_x = None
        best_y = None
        best_z = None
        best_j = None

        # Check all candidate locations for this facility
        for j in range(N):
            if j in used_values:
                continue  # Skip already selected locations

            tempsol[p] = j  # Tentatively assign location j for the current facility

            # Verify distance constraints between selected facilities with the tentative location
            if not check_distance(tempsol, binaryConstraint, fp_euc_distances, p + 1):
                continue  # Skip if distance constraints are violated

            # Try all combinations of chargers (l2, l3) at location j under capacity constraints
            for l2 in range(place[j]):
                for l3 in range(place[j] - l2):
                    if l2 == 0 and l3 == 0:
                        continue  # At least one charger must be placed

                    temp_x = x.copy()  # Copy current charger allocations
                    temp_x[j][0] = l2  # Assign chargers of type 2
                    temp_x[j][1] = l3  # Assign chargers of type 3

                    temp_z = z.copy()  # Copy open facility vector
                    temp_z[j] = 1  # Open facility at location j

                    # Assign clients to the nearest open facility within distance D
                    temp_y = np.zeros((CL, N), dtype=int)
                    for k in range(CL):
                        best_facility = -1
                        best_dist = np.inf
                        for i in range(N):
                            if temp_z[i] == 1 and cl_sp_distances[k][i] <= D:
                                if cl_sp_distances[k][i] < best_dist:
                                    best_dist = cl_sp_distances[k][i]
                                    best_facility = i
                        if best_facility != -1:
                            temp_y[k][
                                best_facility
                            ] = 1  # Assign client k to closest open facility

                    # Concatenate current solution variables for feasibility check
                    temp_solution = np.concatenate(
                        [temp_x.flatten(), temp_y.flatten(), temp_z]
                    )

                    # Check partial feasibility of this tentative solution
                    if not partial_check_feasibility(
                        temp_solution,
                        P,
                        N,
                        CL,
                        F,
                        D,
                        place,
                        capacity,
                        c_level,
                        binaryConstraint,
                        cl_euc_distances,
                        fp_euc_distances,
                    ):
                        continue  # Skip infeasible solutions

                    # Compute cost of current assignment
                    cost = getAssignmentCost(temp_solution, P, CL, cl_sp_distances)

                    # Update best solution if current cost is lower
                    if cost < best_cost:
                        best_cost = cost
                        best_x = temp_x
                        best_y = temp_y
                        best_z = temp_z
                        best_j = j

        # If a valid location was found for this facility, update solution variables
        if best_j is not None:
            tempsol[p] = best_j
            x = best_x
            y = best_y
            z = best_z
            used_values.append(
                best_j
            )  # αποθηκεύει την θέση που χρησιμοποιήθηκε στον πίνακα used_values για να μην χρησιμοποιηθεί ξανά

    # Build final solution vector by concatenating charger allocations, client assignments, and facility openings
    final_solution = np.concatenate([x.flatten(), y.flatten(), z])

    # Check overall feasibility of the final solution
    if check_feasibility(
        final_solution,
        P,
        N,
        CL,
        F,
        D,
        place,
        capacity,
        c_level,
        binaryConstraint,
        cl_euc_distances,
        fp_euc_distances,
        budget_limit,
    ):

        final_cost = getAssignmentCost(final_solution, P, CL, cl_sp_distances)
        heur_status = 1  # Feasible solution found
        return heur_status, final_solution, final_cost
    else:

        heur_status = 0  # No feasible solution found
        return (
            heur_status,
            final_solution,
            np.inf,
        )  # Return infinite cost for infeasibility


# Decodes a flat solution vector into structured integer arrays representing x, y and z variables
def decode_solution(solution, N, F, CL):
    idx = 0
    x_flat = solution[idx : idx + N * len(F)]
    x = (
        np.array(x_flat).reshape((N, len(F))).astype(int)
    )  # Extract and reshape the first segment into x matrix (N x len(F))
    idx += N * len(F)

    y_flat = solution[idx : idx + CL * N]
    y = (
        np.array(y_flat).reshape((CL, N)).astype(int)
    )  # Extract and reshape the next segment into y matrix (CL x N)
    idx += CL * N

    z = np.array(solution[idx : idx + N]).astype(
        int
    )  # Extract the final segment into z vector (length N)

    return x, y, z  # Return the decoded variables x, y, z as integer arrays


# Function that checks feasibility
def check_feasibility(
    solution,
    P,
    N,
    CL,
    F,
    D,
    place,
    capacity,
    c_level,
    binaryConstraint,
    cl_euc_distances,
    fp_euc_distances,
    budget_limit,
    return_penalty=False,
):

    x, y, z = decode_solution(solution, N, F, CL)
    penalty = 0  # counts the number of violated constraints

    feasible = True  # flag indicating whether the solution is feasible

    # Constraint 1: Max number of chargers per location
    for i in range(N):
        if sum(x[i]) > place[i]:
            feasible = False
            penalty += 1

    # Constraint 2: Power limit per installation
    for i in range(N):
        if np.dot(x[i], c_level) > capacity[i]:
            feasible = False
            penalty += 1

    # Constraint 3: Open installation must have at least one charger +   Constraint 4: No chargers unless installation is open
    for i in range(N):
        if z[i] == 1 and np.sum(x[i]) == 0:
            feasible = False
            penalty += 1
        if z[i] == 0 and np.sum(x[i]) > 0:
            feasible = False
            penalty += 1

    # Constraint 5: Exactly P facilities must be opened
    if np.sum(z) != P:
        feasible = False
        penalty += 1

    # Constraint 6: Each customer must be served by exactly one installation + Constraint 7: A customer can only be served by an open installation + Constraint 9: Customer-installation distance must be <= D
    for k in range(CL):
        if np.sum(y[k]) != 1:
            feasible = False
            penalty += 1
        for i in range(N):
            if y[k][i] == 1 and z[i] == 0:
                feasible = False
                penalty += 1
            if y[k][i] == 1 and cl_euc_distances[k][i] > D:
                feasible = False
                penalty += 1

    # Constraint 8: Close proximity installations cannot be opened simultaneously
    for i in range(N):
        for j in range(i + 1, N):
            if z[i] == 1 and z[j] == 1:
                for p in range(P - 1):
                    for l in range(p + 1, P):
                        if fp_euc_distances[i][j] <= binaryConstraint[p][l]:
                            feasible = False
                            penalty += 1

    # Constraint 10: Charger availability must meet customer demand
    for i in range(N):
        station_capacity = 12 * x[i][0] + 48 * x[i][1]
        assigned_clients = sum(y[k][i] for k in range(CL))
        if station_capacity < assigned_clients:
            feasible = False
            penalty += 1

    # Constraint 11: Total cost must not exceed €60,000
    total_cost = sum(100 * z[i] + 2000 * x[i][0] + 6000 * x[i][1] for i in range(N))
    if total_cost > budget_limit:
        feasible = False
        penalty += 1

    # If penalty values are requested, return both feasibility status and penalty
    if return_penalty:
        return feasible, penalty

    return feasible  # Otherwise, return only the feasibility status


# Returns a fitness function for evaluating candidate solutions to the Charging Station Location Problem
def fitness_func(
    P,
    N,
    CL,
    F,
    D,
    place,
    capacity,
    c_level,
    binaryConstraint,
    cl_sp_distances,
    cl_euc_distances,
    fp_euc_distances,
    budget_limit,
):

    def fitness_func(ga_instance, solution, solution_idx):

        # Decode the flat solution into decision variables x, y, z
        x, y, z = decode_solution(solution, N, F, CL)

        # Calculate total shortest path distance between assigned clients and facilities
        total_distance = sum(
            cl_sp_distances[k][i] * y[k][i] for k in range(CL) for i in range(N)
        )

        # Initialize total penalty and penalty weights
        penalty = 0
        penalty_weights = {
            "budget": 10000,
            "power": 1000,
            "slots": 1000,
            "distance": 1000,
            "assignment": 1000,
            "facility_opening": 1000,
            "wrong_facility_count": 5000,
            "capacity_coverage": 1000,
            "fp_min_distance": 1000,
        }

        # Penalty if the number of opened facilities is not equal to P
        if sum(z) != P:
            penalty += penalty_weights["wrong_facility_count"] * abs(sum(z) - P)

        # Penalty if z[i]=1 (facility is open) but x[i]=0 (no chargers), or vice versa
        for i in range(N):
            if z[i] == 1 and sum(x[i]) == 0:
                penalty += penalty_weights["facility_opening"]
            if z[i] == 0 and sum(x[i]) > 0:
                penalty += penalty_weights["facility_opening"]

        # Penalty if number of chargers exceeds slot limits at each facility
        for i in range(N):
            if sum(x[i]) > place[i]:
                penalty += penalty_weights["slots"] * (sum(x[i]) - place[i])

        # Penalty if total power required exceeds the facility's power capacity
        for i in range(N):
            power = sum(x[i][j] * c_level[j] for j in range(len(F)))
            if power > capacity[i]:
                penalty += penalty_weights["power"] * (power - capacity[i])

        # Penalty if total cost exceeds the allowed budget
        total_cost = sum(100 * z[i] + 2000 * x[i][0] + 6000 * x[i][1] for i in range(N))
        if total_cost > budget_limit:
            penalty += penalty_weights["budget"] * (total_cost - budget_limit)

        # Penalties for incorrect client assignments
        for k in range(CL):
            assigned = sum(y[k][i] for i in range(N))
            if assigned != 1:
                penalty += penalty_weights["assignment"] * abs(assigned - 1)

            for i in range(N):
                if y[k][i] == 1:
                    if z[i] == 0:  # Client assigned to closed facility
                        penalty += penalty_weights["facility_opening"]
                    if cl_euc_distances[k][i] > D:  # Assignment exceeds max distance
                        penalty += penalty_weights["distance"] * (
                            cl_euc_distances[k][i] - D
                        )

        #  Penalty if any two opened facilities are too close (based on constraints)
        active_indices = [i for i in range(N) if z[i] == 1]

        if len(active_indices) >= 2:
            for idx1 in range(len(active_indices) - 1):
                for idx2 in range(idx1 + 1, len(active_indices)):
                    i = active_indices[idx1]
                    j = active_indices[idx2]
                    p = idx1
                    l = idx2
                    if p < P and l < P:  # Ensure safe indexing
                        if fp_euc_distances[i][j] <= binaryConstraint[p][l]:
                            penalty += penalty_weights["fp_min_distance"]

        # Penalty if the facility cannot serve all clients assigned to it
        for i in range(N):
            station_capacity = 12 * x[i][0] + 48 * x[i][1]  # Based on charger types
            assigned_clients = sum(y[k][i] for k in range(CL))
            if station_capacity < assigned_clients:
                penalty += penalty_weights["capacity_coverage"] * (
                    assigned_clients - station_capacity
                )

        # The final fitness value (to be maximized) is the negative of the total distance and penalties
        return -total_distance - penalty

    return fitness_func


# Function that generates an initial population solution by randomly selecting facility sites, assigning clients to them,
# and configuring charger levels while considering feasibility and penalties for constraint violations.
def initialize_population_with_random_solution(
    P,
    N,
    CL,
    F,
    D,
    place,
    capacity,
    c_level,
    binaryConstraint,
    clEucDistances,
    fpEucDistances,
    budget_limit,
):

    # Initialize counters and best solution tracking variables
    attempts = 0  # Counter for the number of random solution attempts
    max_attempts = 100  # Maximum number of attempts allowed to find a feasible solution
    best_attempt = None  # Stores the best solution found
    best_cost = float(
        "inf"
    )  # Tracks the lowest penalized cost among all attempted solutions

    # Loop until a feasible solution is found or max attempts exhausted
    while attempts < max_attempts:
        z = np.zeros(
            N, dtype=int
        )  # Initialize binary vector indicating facility openings
        selected = random.sample(range(N), P)  # Randomly select P facility locations
        for idx in selected:
            z[idx] = 1  # Mark the selected facilities as open

        y = np.zeros(
            (CL, N), dtype=int
        )  # Initialize client-to-facility assignment matrix
        assigned_clients_per_node = np.zeros(
            N, dtype=int
        )  # Track number of clients assigned to each facility
        feasible = True  # Flag to indicate feasibility of client assignments

        # Assign clients to facilities within allowed distance D
        for client in range(CL):
            min_dist = float("inf")
            assigned = None
            for facility in selected:
                if (
                    clEucDistances[client][facility] <= D
                    and clEucDistances[client][facility] < min_dist
                ):
                    min_dist = clEucDistances[client][facility]
                    assigned = facility
            if assigned is not None:
                y[client][assigned] = 1  # Assign client to facility
                assigned_clients_per_node[
                    assigned
                ] += 1  # Increment assigned client count
            else:
                # If a client cannot be assigned to any facility within D, mark infeasible and break
                feasible = False
                break

        if not feasible:
            attempts += 1  # Increase attempt count and try again
            continue

        x = np.zeros((N, len(F)), dtype=int)  # Initialize charger allocation matrix

        # Determine charger configurations for each selected facility to satisfy client demand and capacity constraints
        for i in selected:
            clients = assigned_clients_per_node[
                i
            ]  # Number of clients assigned to this facility
            if clients == 0:
                continue  # Skip if no clients assigned

            best_l2, best_l3 = None, None  # Variables to store best charger allocation
            min_cost = float("inf")  # Track minimum installation cost found
            found = False  # Flag to check if any valid charger allocation was found

            # Explore all combinations of Level 2 and Level 3 chargers within space limits
            for l2 in range(place[i] + 1):
                for l3 in range(place[i] - l2 + 1):
                    if l2 == 0 and l3 == 0:
                        continue  # Skip no charger case
                    total_chargers = l2 + l3
                    if total_chargers > place[i]:
                        continue  # Exceeds place capacity, skip
                    served_clients = (
                        l2 * 12 + l3 * 48
                    )  # Calculate how many clients can be served
                    if served_clients < clients:
                        continue  # Not enough chargers to serve all clients
                    power_used = l2 * 22 + l3 * 100  # Calculate power consumption
                    if power_used > capacity[i]:
                        continue  # Power exceeds capacity, skip
                    cost = 2000 * l2 + 6000 * l3  # Installation cost for chargers
                    if cost < min_cost:  # Update best charger allocation
                        best_l2, best_l3 = l2, l3
                        min_cost = cost
                        found = True

            if not found:
                feasible = (
                    False  # No valid charger configuration found, mark infeasible
                )
                break
            else:
                # Assign best charger allocation to facility i
                x[i][0] = best_l2
                x[i][1] = best_l3

        if not feasible:
            attempts += 1  # Retry with another random selection
            continue

        # Flatten matrices and concatenate to form a full solution vector
        solution = list(x.flatten()) + list(y.flatten()) + list(z)

        # Check overall feasibility and calculate penalties for violations
        feasible, penalty = check_feasibility(
            solution,
            P,
            N,
            CL,
            F,
            D,
            place,
            capacity,
            c_level,
            binaryConstraint,
            clEucDistances,
            fpEucDistances,
            budget_limit,
            return_penalty=True,
        )

        # Calculate assignment cost based on client-to-facility distances
        assignment_cost = getAssignmentCost(solution, P, CL, clEucDistances)

        # If feasible, return solution immediately; otherwise keep track of best penalized solution
        if feasible:
            return solution, assignment_cost
        else:
            # If infeasible, calculate penalized cost and update best solution if improved
            assignment_cost = getAssignmentCost(solution, P, CL, clEucDistances)
            penalty_cost = assignment_cost + penalty * 1e6
            if penalty_cost < best_cost:
                best_cost = penalty_cost
                best_attempt = solution

        attempts += 1  # Increment attempt counter

    return best_attempt, np.inf  # Return the best solution found after max attempts


# Function that calculates the cost of a partial/complete assignment (implementation for p-median)
def getAssignmentCost(tempsol, P, CL, cl_sp_distances):
    total_distance = 0
    for cl in range(CL):
        min_distance = np.inf
        for i in range(P):
            cost = cl_sp_distances[cl, tempsol[i]]
            if min_distance > cost:
                min_distance = cost
        total_distance += min_distance

    return total_distance


# Function that executes the genetic algorithm (GA) on a single input data file for the Charging Station Location Problem.
def run_genetic_algorithm(filename):

    # Read input data from the given file
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
    ) = read_data(filename)

    num_charger_types = len(F)
    num_genes = (
        (N * num_charger_types) + (CL * N) + N
    )  # Total number of genes in each chromosome

    # Define gene space for GA: charger types use integer range, assignments and openings are binary
    gene_space = [{"low": 0, "high": 5, "step": 1}] * (N * num_charger_types)
    gene_space += [[0, 1]] * ((CL * N) + N)

    # Solve using Gurobi for optimal reference solution
    print("Solving with Gurobi...")
    start_time = time.time()
    st, tc, gurobi_obj, solving_time = run_model(filename)
    gurobi_time = time.time() - start_time
    print(f"Gurobi objective: {gurobi_obj:.2f} in {gurobi_time:.2f} sec")

    # Initialize GA population using myopic / random heuristic
    start_initialization = time.time()
    initial_population = []
    max_attempts = 100

    for _ in range(max_attempts):
        # Try to generate a feasible solution using the myopic heuristic
        heur_status, initial_solution, initial_cost = myopic_heuristic(
            P,
            N,
            CL,
            binaryConstraint,
            fp_euc_distances,
            cl_euc_distances,
            cl_sp_distances,
            F,
            D,
            place,
            capacity,
            c_level,
        )

        # If heuristic fails, fall back to a random feasible initializer
        if heur_status == False:
            initial_solution = initialize_population_with_random_solution(
                P,
                N,
                CL,
                F,
                D,
                place,
                capacity,
                c_level,
                binaryConstraint,
                cl_euc_distances,
                fp_euc_distances,
                budget_limit,
            )

        # Avoid duplicate individuals in initial population
        already_exists = any(
            np.array_equal(initial_solution, existing)
            for existing in initial_population
        )
        if not already_exists:
            initial_population.append(initial_solution)

    end_initialization = time.time()
    init_time = end_initialization - start_initialization

    # Run Genetic Algorithm using PyGAD
    ga_instance = pygad.GA(
        num_generations=1000,
        num_parents_mating=min(100, len(initial_population)),
        sol_per_pop=len(initial_population),
        num_genes=num_genes,
        fitness_func=fitness_func(
            P,
            N,
            CL,
            F,
            D,
            place,
            capacity,
            c_level,
            binaryConstraint,
            cl_sp_distances,
            cl_euc_distances,
            fp_euc_distances,
            budget_limit,
        ),
        parent_selection_type="rank",
        keep_parents=min(20, len(initial_population)),
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=10,
        initial_population=initial_population,
        stop_criteria=["saturate_200"],
        parallel_processing=["thread", 4],  # Multithreaded evaluation
    )

    # Start timing GA
    start_ga = time.time()
    ga_instance.run()

    # End timing and caclulate total GA time
    end_ga = time.time()
    ga_time = end_ga - start_ga

    # Total execution time including both initialization and genetic algorithm run
    total_time = init_time + ga_time

    # Decode and report best solution
    solution, solution_fitness, _ = ga_instance.best_solution()
    x, y, z = decode_solution(solution, N, F, CL)

    print(f"\nFile: {filename}")
    print("Best solution fitness:", solution_fitness)
    print("Number of open facilities:", np.sum(z))
    print("Total distance (negative fitness):", -solution_fitness)

    print("\n--- x[i][j] (chargers per facility) ---")
    for i in range(N):
        for j in range(len(F)):
            if x[i][j] != 0:
                print(f"x[{i}][{j}] = {x[i][j]}")

    print("\n--- y[k][i] (client-facility assignments) ---")
    for k in range(CL):
        for i in range(N):
            if y[k][i] == 1:
                print(f"y[{k}][{i}] = 1")

    print("\n--- z[i] (open facilities) ---")
    for i in range(N):
        if z[i] == 1:
            print(f"z[{i}] = 1")

    # Calculate GAP between GA and Gurobi results
    if -solution_fitness > 1000:
        gap = "-"
        print("\nNo feasible solution found by the genetic algorithm.")
    else:
        best_cost = -solution_fitness
        gap = (best_cost - gurobi_obj) / gurobi_obj * 100
        print(f"\nGA best cost: {best_cost:.2f}")
        print(f"Gurobi optimal: {gurobi_obj:.2f}")
        print(f"GAP: {gap:.2f}%\n")

    # Write results to output file
    folder_name = os.path.basename(os.path.dirname(filename))
    output_file = f"results_GA_{folder_name}.txt"

    with open(output_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                filename,  # Input file path
                init_time,  # Initialization time
                ga_time,  # GA execution time
                solution_fitness,  # Final fitness (negative total distance)
                gap,  # GAP between GA and optimal
                total_time,  # Total execution time
                gurobi_time,  # Gurobi solve time
            ]
        )


# Executes the genetic algorithm for all input files in a specified folder
def run_all_genetic_algorithms_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            full_path = os.path.join(folder_path, filename)
            run_genetic_algorithm(full_path)


# Main
def main():
    folder_path = input("Enter the folder path containing the .txt files: ")
    run_all_genetic_algorithms_in_folder(folder_path)


if __name__ == "__main__":
    main()
