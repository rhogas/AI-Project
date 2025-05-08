# Required imports
import numpy as np
import networkx as nx
from Boundaries import Boundaries
from Map import EPSILON
from networkx import astar_path

# Number of nodes expanded in the heuristic search (stored in a global variable to be updated from the heuristic functions)
NODES_EXPANDED = 0

def h1(current_node, objective_node) -> np.float32:
    """ First heuristic to implement """
    global NODES_EXPANDED

    # Extract current node and goal node coordinates
    i, j = current_node
    i_goal, j_goal = objective_node

    # Compute Euclidean distance
    h = np.sqrt((i - i_goal)**2 + (j - j_goal)**2)

    NODES_EXPANDED += 1
    return h

def h2(current_node, objective_node) -> np.float32:
    """ Second heuristic to implement """
    global NODES_EXPANDED

    # Extract current node and goal node coordinates
    i, j = current_node
    i_goal, j_goal = objective_node
    
    # Compute Manhattan distance
    h = abs(i - i_goal) + abs(j - j_goal)

    NODES_EXPANDED += 1
    return h

def build_graph(detection_map: np.array, tolerance: np.float32) -> nx.DiGraph:
    """ Builds an adjacency graph (not an adjacency matrix) from the detection map """
    # The only possible connections from a point in space (now a node in the graph) are:
    #   -> Go up
    #   -> Go down
    #   -> Go left
    #   -> Go right
    # Not every point has always 4 possible neighbors
    G = nx.DiGraph()
    H, W = detection_map.shape

    for i in range(H):
        for j in range(W):
            if detection_map[i, j] > tolerance:
                continue

            current = (i, j)

            # Check 4 neighbors (up, down, left, right)
            neighbors = [
                (i - 1, j),  # up
                (i + 1, j),  # down
                (i, j - 1),  # left
                (i, j + 1)   # right
            ]

            for ni, nj in neighbors:
                if 0 <= ni < H and 0 <= nj < W:
                    if detection_map[ni, nj] < tolerance:
                        G.add_edge(current, (ni, nj), weight=detection_map[ni, nj])

    return G


def discretize_coords(high_level_plan: np.array, boundaries: Boundaries, map_width: np.int32, map_height: np.int32) -> np.array:
    """ Converts coordiantes from (lat, lon) into (x, y) """
    min_lat, max_lat = boundaries.min_lat, boundaries.max_lat
    min_lon, max_lon = boundaries.min_lon, boundaries.max_lon

    # Result array for discrete coordinates
    discrete_coords = []

    for lat, lon in high_level_plan:
        # Normalize latitude and longitude to [0, 1]
        norm_lat = (lat - min_lat) / (max_lat - min_lat)
        norm_lon = (lon - min_lon) / (max_lon - min_lon)

        # Map normalized coordinates to grid size
        y = round((1 - norm_lat) * (map_height - 1))  # Y axis goes from top (0) to bottom (map_height - 1)
        x = round(norm_lon * (map_width - 1))         # X axis goes left (0) to right (map_width - 1)

        discrete_coords.append((y, x))

    return np.array(discrete_coords, dtype=np.int32)

def path_finding(G: nx.DiGraph,
                 heuristic_function,
                 locations: np.array, 
                 initial_location_index: np.int32, 
                 boundaries: Boundaries,
                 map_width: np.int32,
                 map_height: np.int32) -> tuple:
    """ Implementation of the main searching / path finding algorithm """
    # Step 1: Convert (lat, lon) into grid (i, j) positions
    discrete_coords = discretize_coords(locations, boundaries, map_width, map_height)

    # Step 2: Build the path following the order of POIs
    current_index = initial_location_index
    final_path = [] # Initialize the final plan 

    for next_index in range(len(discrete_coords)):
        # Skip the current index (already visited)
        if next_index == current_index:
            continue

        source = tuple(discrete_coords[current_index])
        target = tuple(discrete_coords[next_index])

        # Use A* algorithm to find the path from source to target
        try:
            # Both heuristics must take two arguments: the current node and the target node, that is why
            # we use a lambda function to pass the heuristic function
            path_segment = astar_path(G, source, target, heuristic=lambda u, v: heuristic_function(u, v))
            
            if not final_path: # If final_path is empty, initialize it with the first segment
                final_path.extend(path_segment)
            else:
                final_path.extend(path_segment[1:])  # Avoid duplicating the target node
        except nx.NetworkXNoPath: # This exception is raised by astar_path if no path exists
            print(f"No path found from {source} to {target}.")
            continue

    return np.array(final_plan, dtype=np.int32), discrete_coords

def compute_path_cost(G: nx.DiGraph, solution_plan: list) -> np.float32:
    """ Computes the total cost of the whole planning solution """

    total_cost = 0.0

    for i in range(len(solution_plan) - 1):
        u = solution_plan[i]
        v = solution_plan[i + 1]
        
        # Add the weight of the edge (u -> v)
        if G.has_edge(u, v):
            total_cost += G[u][v]['weight']
        else:
            print(f"Error: Edge from {u} to {v} does not exist in the graph.")

    return np.float32(total_cost)