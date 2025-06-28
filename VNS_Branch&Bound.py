# Code developed by Marina Nousi. This code implements Variable Neighborhood Search, whose solution is used as a warm start for the Branch & Bound method.
# It also includes a Myopic heuristic and a Random Generator for initial solution generation.

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
import random  # For stochastic behavior

# Define constants
F = [2, 3]  # Types of chargers (Level 2 and Level 3)
D = 5  # Maximum distance a customer can travel to reach a facility
c_level = [22, 100]  # Power of chargers in kW (Level 2 -> 22 kW, Level 3 -> 100 kW)
budget_limit = 60000  # Budget constraint for the problem


# Define global variables
isMax = None  # Indicates whether the optimization problem is a maximization (True) or minimization (False)
DEBUG_MODE = True  # Flag to enable or disable debug mode
nodes = 0  # Counter to track the number of nodes explored in the branch-and-bound tree
lower_bound = -np.inf  # Initial value for the global lower bound
upper_bound = np.inf  # Initial value for the global upper bound


# Check distance constraints between facilities
def check_distance(solution, binaryCstrs, fp_euc_distances, K):
    for i in range(K - 1):
        for j in range(i + 1, K):
            if fp_euc_distances[solution[i], solution[j]] <= binaryCstrs[i][j]:
                return False
    return True


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
            used_values.append(best_j)

    # Build final solution vector by concatenating charger allocations, client assignments, and facility openings
    final_solution = np.concatenate([x.flatten(), y.flatten(), z])

    # Check overall feasibility of the final solution
    if check_feasibility(
        final_solution, P, N, CL, F, D, place, capacity, c_level, cl_euc_distances
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


# Function that checks feasibility
def check_feasibility(
    solution, P, N, CL, F, D, place, capacity, c_level, clEucDistances
):

    # Extract x, y, and z from the solution vector
    x = np.array(solution[: N * len(F)]).reshape((N, len(F)))
    y = np.array(solution[N * len(F) : N * len(F) + CL * N]).reshape((CL, N))
    z = np.array(solution[-N:])

    feasible = True  # Flag to track overall feasibility status

    # Constraint 1: Max number of chargers per location
    for i in range(N):
        if np.sum(x[i]) > place[i]:
            feasible = False

    # Constraint 2: Power limit per installation
    for i in range(N):
        power = sum(c_level[j] * x[i][j] for j in range(len(F)))
        if power > capacity[i]:
            feasible = False

    # Constraint 3: Open installation must have at least one charger
    for i in range(N):
        if z[i] == 1 and np.sum(x[i]) < 1:
            feasible = False

    # Constraint 4: No chargers unless installation is open
    for i in range(N):
        if z[i] == 0 and np.sum(x[i]) > 0:
            feasible = False

    # Constraint 5: Exactly P facilities must be opened
    if np.sum(z) != P:
        feasible = False

    # Constraint 6: Each customer must be served by exactly one installation
    for k in range(CL):
        if np.sum(y[k]) != 1:
            feasible = False

    # Constraint 7: A customer can only be served by an open installation
    for k in range(CL):
        for i in range(N):
            if y[k][i] == 1 and z[i] == 0:
                feasible = False

    # Constraint 9: Customer-installation distance must be <= D
    for k in range(CL):
        for i in range(N):
            if y[k][i] == 1 and clEucDistances[k][i] > D:
                feasible = False

    # Constraint 10: Charger availability must meet customer demand
    for i in range(N):
        capacity_coverage = 12 * x[i][0] + 48 * x[i][1]
        clients_served = sum(y[k][i] for k in range(CL))
        if capacity_coverage < clients_served:
            feasible = False

    # Constraint 11: Total cost must not exceed €60,000
    total_cost = 0
    for i in range(N):
        total_cost += 100 * z[i] + 2000 * x[i][0] + 6000 * x[i][1]
    if total_cost > 60000:
        feasible = False

    return feasible  # Return  the feasibility status


# Function that perturbs the current solution by trying to open or close facilities in the given neighborhood,
# attempting to explore new regions of the solution space while maintaining feasibility.
def shaking(
    solution,
    neighborhood,
    P,
    N,
    CL,
    binaryConstraint,
    clEucDistances,
    fpEucDistances,
    cl_sp_distances,
):

    # Copy the current solution to avoid modifying the original
    new_solution = solution.copy()

    # Extract subcomponents from the solution vector
    x = np.array(new_solution[: N * len(F)]).reshape(
        (N, len(F))
    )  # Charger distribution
    y = np.array(new_solution[N * len(F) : N * len(F) + CL * N]).reshape(
        (CL, N)
    )  # Client assignments
    z = np.array(new_solution[-N:])  # Facility open/close indicators

    max_attempts = 10  # Max retries per facility

    for facility_idx in neighborhood:
        # Skip invalid facility indices
        if facility_idx < 0 or facility_idx >= N:
            continue

        success = False
        attempts = 0

        while not success and attempts < max_attempts:
            # Backup current state in case the move fails
            x_old = x.copy()
            y_old = y.copy()
            z_old = z.copy()

            # Try to close the current facility and open another one
            if z[facility_idx] == 1:
                closed_facilities = [j for j in range(N) if z[j] == 0]
                if not closed_facilities:
                    break

                facility_to_close = facility_idx
                facility_to_open = random.choice(closed_facilities)

                old_chargers = x[facility_to_close].copy()
                total_old = np.sum(old_chargers)
                capacity_l = place[facility_to_open]

                # Attempt to move chargers to the new facility, respecting capacity
                if total_old <= capacity_l:
                    x[facility_to_open] = old_chargers
                else:
                    x[facility_to_open] = np.floor(
                        (old_chargers / total_old) * capacity_l
                    ).astype(int)

                # Close the old and open the new
                x[facility_to_close] = np.zeros(len(F), dtype=int)
                z[facility_to_close] = 0
                z[facility_to_open] = 1

            # Try to open a closed facility
            else:
                capacity_l = place[facility_idx]

                if np.sum(z) >= P:
                    # Enough facilities already open: randomly close another
                    open_facilities = [j for j in range(N) if z[j] == 1]
                    if not open_facilities:
                        break

                    facility_to_close = random.choice(open_facilities)
                    old_chargers = x[facility_to_close].copy()
                    total_old = np.sum(old_chargers)

                    # Move chargers from the closed facility to the new one
                    if total_old <= capacity_l:
                        x[facility_idx] = old_chargers
                    else:
                        x[facility_idx] = np.floor(
                            (old_chargers / total_old) * capacity_l
                        ).astype(int)

                    x[facility_to_close] = np.zeros(len(F), dtype=int)
                    z[facility_to_close] = 0

                else:
                    # Fewer than P facilities are open: just open this one
                    open_facilities = [
                        j for j in range(N) if z[j] == 1 and j != facility_idx
                    ]

                    if open_facilities:
                        donor = random.choice(open_facilities)
                        old_chargers = x[donor].copy()
                        total_old = np.sum(old_chargers)

                        if total_old <= capacity_l:
                            x[facility_idx] = old_chargers
                        else:
                            x[facility_idx] = np.floor(
                                (old_chargers / total_old) * capacity_l
                            ).astype(int)
                    else:
                        # If no donor exists, assign a random charger distribution
                        x[facility_idx] = np.random.randint(
                            1, capacity_l + 1, size=len(F)
                        )

                z[facility_idx] = 1

            # Reassign clients to the new facility layout
            feasible_assignment = True
            for client in range(CL):
                min_dist = float("inf")
                assigned_facility = None

                for f in range(N):
                    if z[f] == 1 and clEucDistances[client][f] <= D:
                        if clEucDistances[client][f] < min_dist:
                            min_dist = clEucDistances[client][f]
                            assigned_facility = f

                if assigned_facility is not None:
                    y[client] = np.zeros(N)
                    y[client][assigned_facility] = 1
                else:
                    # No valid assignment found for this client
                    feasible_assignment = False
                    break

            if not feasible_assignment:
                # Revert and retry
                x, y, z = x_old, y_old, z_old
                attempts += 1
                continue

            # Create new candidate solution
            new_sol_candidate = list(x.flatten()) + list(y.flatten()) + list(z)

            # Check binary constraints and total number of open facilities
            open_facilities_new = [j for j in range(N) if z[j] == 1]
            if len(open_facilities_new) != P or not check_distance(
                np.array(open_facilities_new), binaryConstraint, fpEucDistances, P
            ):
                x, y, z = x_old, y_old, z_old
                attempts += 1
                continue

            # Check complete feasibility of the new solution
            if check_feasibility(
                new_sol_candidate,
                P,
                N,
                CL,
                F,
                D,
                place,
                capacity,
                c_level,
                clEucDistances,
            ):
                new_solution = new_sol_candidate
                success = True
            else:
                # Revert and retry
                x, y, z = x_old, y_old, z_old
                attempts += 1

    # Calculate and return the cost of the shaken solution
    new_cost = getAssignmentCost(new_solution, P, CL, cl_sp_distances)
    return new_solution, new_cost


# Function to create 'k' neighborhoods for the Variable Neighborhood Search (VNS) algorithm.
def create_neighborhoods(k, P, neighborhood_size):
    vars = list(range(0, P))  # List of variable indices
    neighborhoods = []  # List to store all neighborhoods

    for i in range(k):
        # Randomly select a subset of 'neighborhood_size' facility indices to form a neighborhood
        random_sample = random.sample(vars, neighborhood_size)

        # Add the sampled neighborhood to the list
        neighborhoods.append(random_sample)

    # Return the list of all neighborhoods
    return neighborhoods


# Function that attempts to improve the current solution by swapping the status (open/closed) of a given facility x with a compatible candidate,
# adjusting charger allocations and reassigning clients and returns the best feasible solution found.
def find_best_val(
    x, solution, P, CL, N, clEucDistances, binaryConstraint, fpEucDistances
):

    # Save the original solution and its cost for comparison
    best_original_solution = np.copy(solution)
    best_original_cost = getAssignmentCost(solution, P, CL, clEucDistances)

    # Compute index boundaries for z and y
    z_start = N * len(F) + N * CL
    y_start = N * len(F)

    # Extract facility open/closed status
    z = solution[z_start : z_start + N]
    is_open = z[x] == 1  # Check if facility x is currently open

    # Determine candidate facilities to swap with:
    # If x is open, consider closed ones; if closed, consider open ones (excluding x itself)
    if is_open:
        candidates = [i for i in range(N) if z[i] == 0 and i != x]
    else:
        candidates = [i for i in range(N) if z[i] == 1 and i != x]

    best_candidate_solution = None
    best_candidate_cost = float("inf")

    # Evaluate each candidate for a potential swap
    for i in candidates:
        temp = np.copy(solution)  # Work on a copy of the solution

        # Swap the open/closed status
        if is_open:
            temp[z_start + x] = 0
            temp[z_start + i] = 1
            giver = x
            receiver = i
        else:
            temp[z_start + x] = 1
            temp[z_start + i] = 0
            giver = i
            receiver = x

        # Transfer chargers from giver to receiver considering capacity
        old_chargers = np.array([temp[giver * len(F) + j] for j in range(len(F))])
        total_old = np.sum(old_chargers)
        capacity_l = capacity[receiver]

        if total_old <= capacity_l:
            for j in range(len(F)):
                temp[receiver * len(F) + j] = old_chargers[j]
                temp[giver * len(F) + j] = 0
        else:
            # Scale down proportionally if it exceeds receiver's capacity
            scaled = np.floor((old_chargers / total_old) * capacity_l).astype(int)
            for j in range(len(F)):
                temp[receiver * len(F) + j] = scaled[j]
                temp[giver * len(F) + j] = 0

        # Reassign all clients to their nearest open facility within allowed distance D
        feasible_assignment = True
        for client in range(CL):
            min_dist = float("inf")
            best_fac = -1
            for f in range(N):
                if temp[z_start + f] == 1 and clEucDistances[client][f] <= D:
                    if clEucDistances[client][f] < min_dist:
                        min_dist = clEucDistances[client][f]
                        best_fac = f

            # Assign client to best available facility
            if best_fac >= 0:
                for f in range(N):
                    temp[y_start + client * N + f] = 0
                temp[y_start + client * N + best_fac] = 1
            else:
                # If no valid facility is found, mark assignment as infeasible
                feasible_assignment = False
                break

        if not feasible_assignment:
            continue  # Skip this candidate if assignment is invalid

        # Check if the number of open facilities remains valid
        new_open = [f for f in range(N) if temp[z_start + f] == 1]
        if len(new_open) != P:
            continue

        # Check binary (distance-based) constraints
        if not check_distance(new_open, binaryConstraint, fpEucDistances, P):
            continue

        # Final feasibility check of the candidate solution
        if check_feasibility(
            temp, P, N, CL, F, D, place, capacity, c_level, clEucDistances
        ):
            cost = getAssignmentCost(temp, P, CL, clEucDistances)
            # Keep the best candidate with the lowest cost
            if cost < best_candidate_cost:
                best_candidate_cost = cost
                best_candidate_solution = temp

    # Return the improved solution if found, else return the original
    if best_candidate_solution is not None and best_candidate_cost < best_original_cost:
        return best_candidate_solution
    else:
        return best_original_solution


# Function that performs a local search within a given neighborhood by evaluating and applying beneficial facility status swaps,
# aiming to reduce the overall assignment cost while maintaining feasibility.
def local_search_vns(
    neighborhood, solution, P, N, CL, cl_euc_distances, binaryConstraint, fpEucDistances
):
    # Make a copy of the current solution to work on
    current_solution = np.copy(solution)

    # Evaluate the cost of the current solution
    current_cost = getAssignmentCost(current_solution, P, CL, cl_euc_distances)

    # Iterate over each facility index in the given neighborhood
    for x in neighborhood:
        # Try to improve the solution by modifying facility x
        candidate = find_best_val(
            x,
            current_solution,
            P,
            CL,
            N,
            cl_euc_distances,
            binaryConstraint,
            fpEucDistances,
        )

        # If the candidate solution is invalid, skip to next
        if candidate is None:
            continue

        # Compute the cost of the new candidate solution
        candidate_cost = getAssignmentCost(candidate, P, CL, cl_euc_distances)

        # If the candidate has a lower cost, update current solution
        if candidate_cost < current_cost:
            current_solution = candidate
            current_cost = candidate_cost

    # Return the improved solution and its corresponding cost
    return current_solution, current_cost


# Function that randomly generates a feasible solution by opening P facilities, assigning chargers, and allocating clients within distance D.
# Returns the first valid solution or a fallback if none is found within the attempt limit.
def generate_random_feasible_solution(
    P, N, CL, F, D, place, capacity, c_level, clEucDistances
):
    attempts = 0
    max_attempts = 1000
    last_solution = None  # Store the last attempted solution

    while attempts < max_attempts:
        # Randomly open P facilities
        z = np.zeros(N, dtype=int)
        selected = random.sample(range(N), P)
        for idx in selected:
            z[idx] = 1

        # Assign chargers randomly within capacity limits to the open facilities
        x = np.zeros((N, len(F)), dtype=int)
        for idx in selected:
            total = random.randint(1, place[idx])  # Total chargers at this facility
            first = random.randint(0, total)  # Random split between charger types
            x[idx][0] = first
            x[idx][1] = total - first

        # Assign each client to the nearest open facility within distance D
        y = np.zeros((CL, N), dtype=int)
        feasible = True
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
                y[client][assigned] = 1
            else:
                feasible = False  # No feasible facility found for this client
                break

        # If client assignment is infeasible, retry
        if not feasible:
            attempts += 1
            continue

        # Construct the full solution vector
        open_facilities_new = [i for i in range(N) if z[i] == 1]
        solution = list(x.flatten()) + list(y.flatten()) + list(z)
        last_solution = solution  # Save the last constructed solution

        # Check full feasibility of the generated solution
        if check_feasibility(
            solution, P, N, CL, F, D, place, capacity, c_level, clEucDistances
        ) and check_distance(
            open_facilities_new, binaryConstraint, fp_euc_distances, P
        ):
            print("Βρέθηκε αρχική εφικτή λύση απο το Random Generator")
            return solution, getAssignmentCost(solution, P, CL, clEucDistances)

        attempts += 1

    # Fallback if no feasible solution found
    print("Δεν βρέθηκε αρχική εφικτή λύση έπειτα απο", max_attempts, "προσπάθειες")
    return last_solution if last_solution else [0] * (N * len(F) + CL * N + N), float(
        "inf"
    )


# Executes the Variable Neighborhood Search (VNS) algorithm to iteratively improve a given solution.
def VNS_algorithm(
    kmax,
    max_iterations,
    neighborhood_size,
    solution,
    solution_cost,
    P,
    N,
    CL,
    binaryConstraint,
    fp_euc_distances,
    cl_euc_distances,
):

    print(
        "************************    Running Variable Neighborhood Search    ************************\n\n"
    )

    iterations = 0
    current_sol = np.array(solution)
    current_sol_cost = solution_cost

    # Extract solution components: x, y, z
    x = np.array(current_sol[: N * len(F)]).reshape((N, len(F)))
    y = np.array(current_sol[N * len(F) : N * len(F) + CL * N]).reshape((CL, N))
    z = np.array(current_sol[-N:])

    # Check if the initial solution is invalid
    if np.sum(z) == 0:
        print("Initial solution not feasible. Generating random feasible solution...")
        current_sol, current_sol_cost = generate_random_feasible_solution(
            P, N, CL, F, D, place, capacity, c_level, cl_euc_distances
        )
        print(current_sol_cost)

    # Create kmax neighborhoods with size = neighborhood_size
    neighborhoods = create_neighborhoods(kmax, P, neighborhood_size)

    # If neighborhoods couldn't be created, terminate early
    if neighborhoods is None:
        print("Cannot create neighborhoods. VNS terminated.")
        return None, float("inf")

    # Main VNS loop
    while iterations < max_iterations:
        k = 0  # Start from the first neighborhood

        while k < kmax:
            # Shaking: randomly perturbs the current solution using neighborhood[k]
            new_sol, new_sol_cost = shaking(
                current_sol,
                neighborhoods[k],
                P,
                N,
                CL,
                binaryConstraint,
                cl_euc_distances,
                fp_euc_distances,
                cl_sp_distances,
            )

            # Skip to next neighborhood if the new solution is infeasible
            if not check_feasibility(
                new_sol, P, N, CL, F, D, place, capacity, c_level, cl_euc_distances
            ):
                k += 1
                continue

            # Apply local search to the shaken solution
            new_sol, new_sol_cost = local_search_vns(
                neighborhoods[k],
                new_sol,
                P,
                N,
                CL,
                cl_euc_distances,
                binaryConstraint,
                fp_euc_distances,
            )

            # If a better solution is found, accept it and restart neighborhood index
            if new_sol_cost < current_sol_cost:
                current_sol = np.array(new_sol)
                current_sol_cost = new_sol_cost
                print(
                    f"-- Improved bound: Iteration {iterations}, k {k}, Best Evaluation {current_sol_cost:.5f}"
                )
                k = 0
            else:
                k += 1  # Move to the next neighborhood

        # Print progress every 10 iterations
        if iterations % 10 == 0:
            print(f"Iteration {iterations}, Best Evaluation {current_sol_cost:.5f}")

        iterations += 1

    print(f"Last Iteration {iterations-1}, Best Evaluation {current_sol_cost:.5f}")

    return current_sol, current_sol_cost


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

    # Use of a list to store nodes
    node_list = []

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

    # check if the model was solved to optimality. If not then return (infeasible)
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
    ) = read_data(
        file_path
    )  # "C:/Users/hp-i3/25_15_5/9.txt"

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

    # Create left and right branches ( set left: x = 0, right: x = 1 )
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

    # Add child nodes in list
    node_list.append(left_child)
    node_list.append(right_child)

    # Solving sub problems
    while node_list:
        print(
            "\n********************************  NEW NODE BEING EXPLORED  ******************************** "
        )

        # Increment the total number of explored nodes by 1
        nodes += 1

        # Select the node with the smallest lower bound. If multiple nodes have the same lower bound, prioritize the right child by using a secondary sorting key
        # In the tuple (np.min(node.lb), node.label != "Right"), 'False' (i.e., right child) is treated as smaller than 'True' (left child)
        current_node = min(
            node_list, key=lambda node: (np.min(node.lb), node.label != "Right")
        )

        node_list.remove(current_node)

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

        # Create left and right branches  ( set left: x = 0, right: x = 1 )
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

        # Add child nodes in list
        node_list.append(left_child)
        node_list.append(right_child)

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
    results_file = f"results_BestFS_MinDistance_VNS_{os.path.basename(folder_path)}.txt"

    for file in txt_files:
        file_path = os.path.join(folder_path, file)
        print(f"\n========= Processing file: {file_path} =========")

        # Initialize the lower and upper bounds for the problem
        lower_bound = -np.inf
        upper_bound = np.inf

        # Create a new Gurobi model for each input file
        model = gp.Model()
        model.reset()
        model, ub, lb, integer_var, num_vars, c = pr.pmedian(file_path)
        isMax = False
        model2 = model.copy()

        # Initialize best bounds per depth depending on problem type
        if isMax == True:
            best_bound_per_depth = np.array([-np.inf for i in range(num_vars)])
        else:
            best_bound_per_depth = np.array([np.inf for i in range(num_vars)])

        nodes_per_depth = np.zeros(num_vars + 1, dtype=float)
        nodes_per_depth[0] = 1
        for i in range(1, num_vars + 1):
            nodes_per_depth[i] = (
                nodes_per_depth[i - 1] * 2
            )  # Number of nodes doubles at each level of the search tree

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
            fp_euc_distances,
            cl_euc_distances,
            cl_sp_distances,
        ) = read_data(file_path)

        start_idx = N * len(F) + CL * N
        end_idx = start_idx + N

        print(
            "************************    Running heuristic search    ************************\n\n"
        )

        # Start timer for heuristic execution time
        start_heuristic = time.time()

        # Run the myopic  heuristic to get an initial feasible solution and its cost
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

        # End timer
        end_heuristic = time.time()

        # Calculate total time spent on heuristic
        time_heuristic = end_heuristic - start_heuristic

        # Store the objective value and corresponding solution
        heur_obj = initial_cost
        heur_sol = initial_solution

        # If the heuristic did not return a feasible solution
        if not heur_status:
            print(
                "Η λύση του μυωπικού αλγορίθμου είναι μη εφικτή. Προσπαθούμε με τυχαία εφικτή λύση."
            )

            # Try generating a feasible solution randomly
            heur_sol, heur_obj = generate_random_feasible_solution(
                P, N, CL, F, D, place, capacity, c_level, cl_euc_distances
            )

            # If the random generator also failed, fall back to a default zero solution
            if heur_obj == float("inf"):
                print(
                    "Αποτυχία δημιουργίας εφικτής λύσης μέσω τυχαίας μεθόδου. Χρήση μηδενικής λύσης."
                )
                heur_sol = [0] * (N * len(F) + CL * N + N)  # Zeroed solution vector
                heur_obj = 1e10  # Assign a large objective cost as penalty

        # ---------------------- Run Variable Neighborhood Search (VNS) ----------------------

        # Start timing the VNS metaheuristic
        start_metaheuristic = time.time()

        # Execute the VNS algorithm with:
        # kmax = 20 neighborhoods, 50 max iterations, and initial solution from heuristic
        best_sol, best_sol_cost = VNS_algorithm(
            20,
            50,
            int(2 * P / 3),
            heur_sol,
            heur_obj,
            P,
            N,
            CL,
            binaryConstraint,
            fp_euc_distances,
            cl_euc_distances,
        )

        # End timing the VNS metaheuristic
        end_metaheuristic = time.time()

        # Calculate total VNS execution time
        time_metaheuristic = end_metaheuristic - start_metaheuristic

        # Feasibility status flag
        status2 = False

        # If VNS returns no feasible solution
        if best_sol is None:
            print("VNS returned no feasible solution.")
        else:
            # Check feasibility of the returned solution
            status2 = check_feasibility(
                best_sol, P, N, CL, F, D, place, capacity, c_level, cl_euc_distances
            )

            # Print the solution vector and its objective cost
            print(f"\n\nSolution: {list(map(int, best_sol))}")
            print(f"Objective value: {best_sol_cost}")

        # Print final feasibility status
        if status2:
            print("Status: Feasible")
        else:
            print("Status: Infeasible")
            metaheur_obj = best_sol_cost  # Save cost for infeasible solution case

        # Final message after completing the VNS search
        print(f"\n Variable Neighborhood Search Finished. Best cost: {best_sol_cost}")

        # -------------------- Gurobi WITH Warm Start (metaheuristic) --------------------
        print(
            "\n************************ Gurobi WITH Warm Start ************************"
        )

        # Set the upper bound of the objective function based on the best cost found from the metaheuristic
        upper_bound = best_sol_cost

        # Update the Gurobi model to ensure all recent changes are reflected before solving
        model2.update()

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
            model2, ub, lb, integer_var, best_bound_per_depth, nodes_per_depth
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
            model2.getVars()[i].LB = best_z_values[i - start_idx]
            model2.getVars()[i].UB = best_z_values[i - start_idx]

        # Set all variables in the model to integer type before solving with Gurobi
        all_vars = model2.getVars()
        int_types = [GRB.INTEGER] * len(all_vars)
        model2.setAttr("VType", all_vars, int_types)

        # Update the model to apply variable type changes
        model2.update()

        print("ΛΥΣΗ ΜΕ GUROBI\n")

        # Start timer for Gurobi optimization
        start_gurobi = time.time()

        # Optimize the model using Gurobi
        model2.optimize()

        # End timer and calculate Gurobi solving time
        end_gurobi = time.time()
        time_gurobi = end_gurobi - start_gurobi

        # Calculate total solving time combining Βranch and Βound and Gurobi phases
        total_time = time_bb + time_gurobi

        # If an optimal solution is found, print the objective value and variable values
        if model2.Status == 2:
            print("\n========= Optimal Solution Found =========")
            print(f"Objective Value: {model2.ObjVal}")

            # Retrieve all variable values from the solution
            var_values = np.array([v.X for v in model2.getVars()])

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
            print("No optimal solution found. Status:", model2.Status)

        # Write results to the output file
        with open(results_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            # If an optimal solution was found, log the objective value
            if model2.Status == 2:
                writer.writerow(
                    [
                        file_path,
                        time_heuristic,
                        time_metaheuristic,
                        total_time,
                        model2.ObjVal,
                        best_sol_cost,
                    ]
                )
            else:
                # Otherwise, note that no optimal solution was found
                writer.writerow(
                    [
                        file_path,
                        time_heuristic,
                        time_metaheuristic,
                        total_time,
                        "No optimal solution",
                    ]
                )

        print(f"Time Elapsed: {total_time}")
