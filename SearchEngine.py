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

    graph = nx.DiGraph()
    H, W = detection_map.shape

    for i in range(H):
        for j in range(W):
            if detection_map[i, j] > tolerance:
                continue
            current = (i, j)
            graph.add_node(current, weight=detection_map[i, j]) # Ensure the node is added to the graph even if it has no neighbors

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
                        if (i, j) != (ni, nj):  # Ensure no self-loops
                            graph.add_edge(current, (ni, nj), weight=detection_map[ni, nj])

    return graph


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
    # Convert (lat, lon) into grid (i, j) positions
    discrete_coords = discretize_coords(locations, boundaries, map_width, map_height)
    
    print(f"POIs to visit: {', '.join(map(str, discrete_coords))}")
    # Create the visiting order of the POIs
    visiting_order = create_visiting_order(locations, heuristic_function, G, boundaries, map_width, map_height)

    if visiting_order is None:
        return None, 0, None
    print(f"Visiting order: {', '.join(map(str, discrete_coords[visiting_order]))}")

    # Build the path following the order of POIs
    current_index = initial_location_index
    final_path = [] # Initialize the final plan 
    pois_in_path = [] # Initialize the POIs in the path


    for next_index in visiting_order:
        # Skip the current index (already visited)
        if next_index == current_index:
            continue

        source = tuple(discrete_coords[visiting_order[current_index]])
        target = tuple(discrete_coords[visiting_order[next_index]])

        pois_in_path.append(source) # Save the source POI in the path

        # Use A* algorithm to find the path from source to target
        try:
            # Both heuristics must take two arguments: the current node and the target node, that is why
            # we use a lambda function to pass the heuristic function
            path_segment = astar_path(G, source, target, heuristic=lambda u, v: heuristic_function(u, v))
            if not final_path: # If final_path is empty, initialize it with the first segment
                final_path.extend(path_segment)
            else:
                final_path.extend(path_segment[1:])  # Avoid duplicating the target node
                

            # Add the POI to the pois_in_path list
            pois_in_path.append(target)  # Save the discrete target grid coordinate

            current_index = next_index  # Update the current index to the next one

        except nx.NodeNotFound:
            print(f"ERROR. One of the POIs ({source} or {target}) is not in the graph.\nThe POI may be outside the map or has no valid neighbors due to the tolerance setting (it is too low).")
            return None, 0, None
        except nx.NetworkXNoPath: # This exception is raised by astar_path if no path exists
            print(f"ERROR. No path exists from {source} POI to {target} POI with the current map and tolerance.")
            return None, 0, None
            
    return final_path, NODES_EXPANDED, pois_in_path



def compute_path_cost(G: nx.DiGraph, solution_plan: list) -> np.float32:
    """ Computes the total cost of the whole planning solution """

    total_cost = 0.0

    for i in range(len(solution_plan) - 1):
        u = tuple(solution_plan[i]) # Convert numpy array to tuple
        v = tuple(solution_plan[i + 1]) # Convert numpy array to tuple
        
        # Add the weight of the edge (u -> v)
        if G.has_edge(u, v):
            total_cost += G[u][v]['weight']
        else:
            print(f"Error: Edge from {u} to {v} does not exist in the graph.")

    return np.float32(total_cost)


def create_visiting_order(locations, heuristic_function, G, boundaries, map_width, map_height):
    # Initialize the visiting order and the set of unvisited POIs
    remaining_pois = list(range(len(locations)))  # List of all POIs to visit
    current_index = 0  # Start from the initial POI (index 0, for example)
    visiting_order = [current_index]  # Visiting order, starting with the initial POI

    while remaining_pois:
        remaining_pois.remove(current_index)  # Remove the current POI from the list of remaining POIs

        # Find the closest POI based on the heuristic function
        next_index = None
        min_cost = float('inf')  # Initialize with a large cost value

        for next_poi in remaining_pois:
            source = tuple(discretize_coords([locations[current_index]], boundaries, map_width, map_height)[0])
            target = tuple(discretize_coords([locations[next_poi]], boundaries, map_width, map_height)[0])

            try: 
                if nx.has_path(G, source, target):
                    # Calculate the heuristic cost to move from the current POI to the next POI
                    cost = heuristic_function(source, target)

                    # If this POI is the closest, update next_index
                    if cost < min_cost:
                        min_cost = cost
                        next_index = next_poi
                else:
                    print(f"No path between {source} and {target}")
            except nx.NodeNotFound:
                print(f"ERROR. One of the POIs ({source} or {target}) is not in the graph.\nThe POI may be outside the map or has no valid neighbors due to the tolerance setting (it is too low).")
                return None

            except nx.NetworkXNoPath:
                print(f"ERROR. No path exists from {source} POI to {target} POI with the current map and tolerance.")
                return None

        # Append the next POI to the visiting order
        if next_index is not None:
            visiting_order.append(next_index)
            current_index = next_index  # Update current POI to the selected one

    return visiting_order