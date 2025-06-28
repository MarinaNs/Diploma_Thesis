# Code developed by Nousi Marina for the problem generator

# Import necessary libraries
import random  # Generate random numbers and make random selections
import numpy  # Library for arrays and numerical computations
import argparse  # Helps in parsing command-line arguments
import os  # For operating system operations (files, directories)

# Global parameters
N = 0  # Number of potential installation sites
CL = 0  # Number of clients
P = 0  # Number of facilities that must be opened
PR = 0  # Number of problems to generate
max_euclideian = 0  # Max Euclidean distance
grid_size = 0  # Size of the square grid
F_F_STRICTNESS = 7  # Strictness for facility-to-facility constraints
F_CL_STRICTNESS = 1  # Strictness for facility-to-client constraints
FACILITY_POINT = 1  # Marker for facility location in grid
CLIENT_POINT = -1  # Marker for client location in grid


# Class representing a point with x, y coordinates
class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


# Function to initialize the grid
def initialize_grid(grid, points):
    print("--Initializing grid... (step 1)")
    i = 0
    # In this case, select all points from the grid
    if (grid_size * grid_size) == N:
        for i in range(N):
            points[i].x = i // grid_size
            points[i].y = i % grid_size
            grid[points[i].x][points[i].y] = FACILITY_POINT
    # Otherwise, randomly select coordinates for the points that will be selected
    else:
        while i < N:
            points[i].x = random.randint(0, grid_size - 1)
            points[i].y = random.randint(0, grid_size - 1)

            # If the randomly selected point has not been selected before, add it to the grid.
            if grid[points[i].x][points[i].y] == 0:
                grid[points[i].x][points[i].y] = FACILITY_POINT
                i += 1


# Function that calculates the euclideian distances between ALL points on the grid
# Point_coords contains all pairs of points (even with themselves)
def compute_euclideian(point_coords):
    global max_euclideian
    print("--Computing Euclideian distances... (step 2)")

    # Compute the squared differences for all pairs using broadcasting
    diff_x = point_coords[:, 0].reshape(-1, 1) - point_coords[:, 0]
    diff_y = point_coords[:, 1].reshape(-1, 1) - point_coords[:, 1]
    squared_distances = diff_x**2 + diff_y**2

    # Calculate Euclidean distances
    euclidean_distances = numpy.sqrt(squared_distances)

    # Find the maximum Euclidean distance (this will be used for the distance constraints)
    max_euclideian = numpy.max(euclidean_distances)

    return euclidean_distances


# Function that calculates the shortest path distances between ALL points on the grid
# Notice that in our case, shortest paths are equal to the Manhattan distances
# Point_coords contains all pairs of points (even with themselves)
def compute_shortestpath(point_coords):
    print("--Computing Shotest Path distances... (step 3)")

    # Compute the squared differences for all pairs using broadcasting
    diff_x = point_coords[:, 0].reshape(-1, 1) - point_coords[:, 0]
    diff_y = point_coords[:, 1].reshape(-1, 1) - point_coords[:, 1]

    # Shortest path distance is equal to the Manhattan distance
    sp_distances = numpy.abs(diff_x) + numpy.abs(diff_y)

    return sp_distances


# Function that randomly selects clients from N points
# that were previously selected as FACILITY_POINT
def select_clients(grid, points):
    print("--Selecting clients... (step 4)")
    client_list = list()
    clients = numpy.zeros(CL, dtype=int)
    clients_in_grid = 0
    while clients_in_grid < CL:
        # Generate a potential client index (out of N points) in a random way
        cl_index = random.randint(0, N - 1)

        # Check if client has already been selected
        if cl_index not in client_list:
            client_list.append(cl_index)

            # If not then generate a label for this client based on the coordinates
            clients[clients_in_grid] = (
                points[cl_index].x * grid_size + points[cl_index].y
            )

            # Mark the corresponding point on the grid as a client (CLIENT_POINT)
            grid[points[cl_index].x][points[cl_index].y] = CLIENT_POINT

            # Increment clients_in_grid counter, as we succesfully generated a client
            clients_in_grid += 1

    return clients


# Function that generates distance constraints between facilities
# The lower bound is generated randomly from the range [0, max_euclideian/denom]
# where 'denom' can be modified from the input arguments that the user passes
def generate_fac_to_fac_constraints():
    print("--Generating distance constraints between facilities... (step 5)")
    fac_to_fac_cstrs = list()
    for i in range(P - 1):
        for j in range(i + 1, P):
            distance = random.randint(0, round(max_euclideian / F_F_STRICTNESS))
            fac_to_fac_cstrs.append([i, j, distance])

    return fac_to_fac_cstrs


# Function that generates distance constraints between facilities and clients
# The lower bound is generated randomly from the range [0, strictness]
# where 'strictness' can be modified from the input arguments that the user passes
def generate_fac_to_cl_constraints():
    print(
        "--Generating distance constraints between facilities and clients... (step 6)"
    )
    fac_to_cl_cstrs = list()
    for i in range(P):
        distance = random.randint(0, F_CL_STRICTNESS)
        fac_to_cl_cstrs.append(distance)

    return fac_to_cl_cstrs


# Create an array indicating how many chargers each facility can host
def create_place_array(size):
    print("--Creating place array... (step 7)")
    place = [random.randint(1, 5) for i in range(size)]
    return place


# Create an array representing the power capacity of each facility (in KW)
def create_capacity_array(size):
    print("--Creating capacity array... (step 8)")
    capacity = [random.randint(22, 200) for i in range(size)]
    return capacity


# Function that saves a problem on the disk
def save_on_disk(
    grid,
    clients,
    euclideian,
    sp_distances,
    point_coords,
    fac_to_fac_cstrs,
    fac_to_cl_cstrs,
    filename="0",
    place=[],
    capacity=[],
):
    FP = int(abs(N - CL))
    folder_name = str(N) + "_" + str(CL) + "_" + str(P)
    file_extension = ".txt"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    try:
        file_handler = open(folder_name + "/" + filename + file_extension, "w")

        print("--Writing problems on disk... (step 9)")
        text = list()
        text.append(f"{grid_size*grid_size} {CL} {FP} {P}\n")

        text.append(f"{CL} clients:\n")  # Write client data
        for i in clients:
            text.append(f"{i}\n")

        text.append(f"{FP} candidate facilities:\n")  # Write candidate facility data
        for i in range(grid_size):
            for j in range(grid_size):
                if grid[i][j] == FACILITY_POINT:
                    text.append(f"{i * grid_size + j}\n")

        if place:  # is not None
            text.append(
                f"{FP} values corresponding to the charging positions for each candidate facility:\n"
            )  # Write charger slots per facility
            for value in place:
                text.append(f"{value}\n")

        if capacity:  # is not None
            text.append(
                f"{FP} values corresponding to the total capacity for each candidate facility:\n"
            )  # Write capacity values per facility
            for value in capacity:
                text.append(f"{value}\n")

        # Write constraints between facilities and clients
        text.append(f"{P} constraints between facilities and clients:\n")
        [text.append(f"{i} {fac_to_cl_cstrs[i]}\n") for i in range(P)]

        # Write constraints between facilities
        text.append(f"{int(P * (P - 1) / 2)} constraints between facilities:\n")
        [text.append(f"{cstr[0]} {cstr[1]} {cstr[2]}\n") for cstr in fac_to_fac_cstrs]

        # Write distance matrix between candidate facilities
        text.append(
            f"{int(FP * FP - FP)} shortest paths and Euclidean distances between candidate facilities:\n"
        )
        for i in range(grid_size * grid_size):
            [
                text.append(f"{i} {j} {sp_distances[i, j]} {euclideian[i, j]:.6f}\n")
                for j in range(grid_size * grid_size)
                if grid[i // grid_size, i % grid_size] == FACILITY_POINT
                and grid[j // grid_size, j % grid_size] == FACILITY_POINT
                and i != j
            ]

        # Write distance matrix between clients and candidate facilities
        text.append(
            f"{int(CL * FP)} shortest paths and Euclidean distances between clients and candidate facilities:\n"
        )
        for i in range(grid_size * grid_size):
            [
                text.append(f"{i} {j} {sp_distances[i, j]} {euclideian[i, j]:.6f}\n")
                for j in range(grid_size * grid_size)
                if grid[i // grid_size, i % grid_size] == CLIENT_POINT
                and grid[j // grid_size, j % grid_size] == FACILITY_POINT
                and i != j
            ]

        # Remove last \n character
        text[-1] = text[-1][:-1]
        file_handler.write("".join(text))
        file_handler.close()
        print(f"\nSuccesfully created file {filename+file_extension}.\n\n")
    except Exception as e:
        print(str(e))
        print(f"\nError while creating file {filename+file_extension}!\n\n")


# Function that handles the input arguments from the user
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Facility Location Problem Generator\nUsage: <grid size> <total points (N)> <clients (CL)> <facilities (P)> <# problems>"
    )

    # Define each positional argument
    parser.add_argument(
        "grid_size", type=int, help="Size of the grid (e.g., 10 for a 10x10 grid)"
    )
    parser.add_argument("total_points", type=int, help="Total number of points (N)")
    parser.add_argument("clients", type=int, help="Number of clients (CL)")
    parser.add_argument("facilities", type=int, help="Number of facilities (P)")
    parser.add_argument("num_problems", type=int, help="Number of problems")
    parser.add_argument(
        "--f_f_strictness",
        type=int,
        help="Define strictness of constraints between facilities (0, max_euclideian])",
    )
    parser.add_argument(
        "--f_cl_strictness",
        type=int,
        help="Define strictness of constraints between facilities and clients [0,+inf]",
    )

    # Parse arguments
    args = parser.parse_args()

    global grid_size, N, CL, P, PR, F_F_STRICTNESS, F_CL_STRICTNESS
    grid_size = args.grid_size
    N = args.total_points
    CL = args.clients
    P = args.facilities
    PR = args.num_problems
    if args.f_f_strictness is not None:
        if args.f_f_strictness <= 0:
            print("f_f_strictness cannot <= zero. Using default (=2)")
        else:
            F_F_STRICTNESS = args.f_f_strictness
    if args.f_cl_strictness is not None:
        F_CL_STRICTNESS = args.f_cl_strictness
    return args


# To use the parser
if __name__ == "__main__":

    # Handle input arguments
    args = parse_arguments()

    for i in range(PR):
        # Define grid
        grid = numpy.zeros(shape=(grid_size, grid_size), dtype=int)

        # Define points list
        points = [Point() for i in range(N)]

        initialize_grid(grid, points)

        # Generate (x, y) coordinates for all points in the grid
        point_coords = numpy.array(
            [(i // grid_size, i % grid_size) for i in range(grid_size * grid_size)]
        )

        # Calculate Euclideian distances
        euclidean_distances = compute_euclideian(point_coords)

        # Calculate shortest path distances
        sp_distances = compute_shortestpath(point_coords)

        # Select clients from the grid
        clients = select_clients(grid, points)

        # Generate distance constraints between facilities
        fac_to_fac_cstrs = generate_fac_to_fac_constraints()

        # Generate distance constraints between facilities and clients
        fac_to_cl_cstrs = generate_fac_to_cl_constraints()

        # Create place array
        place = create_place_array(abs(N - CL))

        # Create place array
        capacity = create_capacity_array(abs(N - CL))

        # Save problem on disk
        save_on_disk(
            grid,
            clients,
            euclidean_distances,
            sp_distances,
            point_coords,
            fac_to_fac_cstrs,
            fac_to_cl_cstrs,
            str(i),
            place,
            capacity,
        )

        # Delete data structures
        del grid
        del points
        del point_coords
        del euclidean_distances
        del sp_distances
        del clients
        del fac_to_fac_cstrs
        del fac_to_cl_cstrs
        del place
        del capacity
