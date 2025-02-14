import numpy as np
import random


def find_all_triangles(adjmat):
    """
    Finds all triangles in the undirected graph represented by adjmat.
    Each triangle is returned as a tuple (i, j, k) with i < j < k.
    """
    n = adjmat.shape[0]
    triangles = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                if adjmat[i, j] == 1 and adjmat[j, k] == 1 and adjmat[i, k] == 1:
                    triangles.append((i, j, k))
    return triangles

def convert_adjmat_to_string(adjmat):
    """
    Converts the upper-triangular entries of the adjacency matrix to a string.
    (Edges are read row by row, skipping the diagonal and lower-triangular part.)
    """
    n = adjmat.shape[0]
    entries = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            entries.append(str(adjmat[i, j]))
    return "".join(entries)

def string_to_adjmat(obj, N):
    # Create an empty adjacency matrix
    adjmat = np.zeros((N, N), dtype=int)
    
    # Fill the upper triangular matrix from the input string `obj`
    index = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            # Convert the current character to an integer (0 or 1)
            value = int(obj[index])
            adjmat[i, j] = value
            adjmat[j, i] = value  # Ensure the matrix is symmetric
            index += 1
    return adjmat

def greedy_search_from_startpoint(db, obj, N):
    """
    Main greedy search algorithm.
    Input:
      - db: (unused in this code)
      - obj: A string representing the upper-triangle of an adjacency matrix.
             The length should be N*(N-1)//2.
    The algorithm:
      1. Builds a symmetric N x N adjacency matrix from obj.
      2. Greedily removes edges that participate in triangles until no triangle exists.
      3. Greedily adds random edges that do not create any triangle until no such edge remains.
    Returns:
      A string representation of the final upper-triangular adjacency matrix.
    """
    # Create an empty adjacency matrix
    adjmat = np.zeros((N, N), dtype=int)
    
    # Fill the upper triangular matrix from the input string `obj`
    index = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            # Convert the current character to an integer (0 or 1)
            value = int(obj[index])
            adjmat[i, j] = value
            adjmat[j, i] = value  # Ensure the matrix is symmetric
            index += 1

    # Remove triangles by deleting the most frequent edge in any triangle until no triangles remain.
    triangles = find_all_triangles(adjmat)
    while triangles:
        edge_count = {}
        for (i, j, k) in triangles:
            for edge in [(i, j), (j, k), (i, k)]:
                edge_count[edge] = edge_count.get(edge, 0) + 1

        # Find the edge that appears in the most triangles
        most_frequent_edge = max(edge_count, key=edge_count.get)
        i_edge, j_edge = most_frequent_edge

        # Remove this edge from the graph
        adjmat[i_edge, j_edge] = 0
        adjmat[j_edge, i_edge] = 0

        # Update triangles by removing those that contain the removed edge
        triangles = [
            t for t in triangles 
            if most_frequent_edge not in [(t[0], t[1]), (t[1], t[2]), (t[0], t[2])]
        ]

    # Now add allowed edges (those that do not create a triangle) one by one at random.
    allowed_edges = []
    # Compute A^2 to count two-step connections
    adjmat2 = np.dot(adjmat, adjmat)
    for i in range(N - 1):
        for j in range(i + 1, N):
            if adjmat[i, j] == 0 and adjmat2[i, j] == 0:
                allowed_edges.append((i, j))

    while allowed_edges:
        # Randomly select an edge to add
        edge = random.choice(allowed_edges)
        i_edge, j_edge = edge
        adjmat[i_edge, j_edge] = 1
        adjmat[j_edge, i_edge] = 1

        # Update allowed_edges by removing those that would form a triangle with the newly added edge
        new_allowed_edges = []
        for (a, b) in allowed_edges:
            # Skip the newly added edge if it's still in the list
            if (a, b) == (i_edge, j_edge):
                continue

            # Check if adding (a,b) now would create a triangle with the new edge (i_edge, j_edge)
            if (a == i_edge and adjmat[b, j_edge] == 1) or \
               (a == j_edge and adjmat[b, i_edge] == 1) or \
               (b == i_edge and adjmat[a, j_edge] == 1) or \
               (b == j_edge and adjmat[a, i_edge] == 1):
                continue

            new_allowed_edges.append((a, b))
        allowed_edges = new_allowed_edges

    return convert_adjmat_to_string(adjmat)

def reward_calc(obj, N):
    """
    Calculates the reward of a construction.
    For example, counts the number of edges (i.e. '1's) in the string representation.
    """
    ## differs from paper, there it is num of edges - 2 * num of triangles
    #return obj.count('1')
    edges = obj.count('1')
    triangles = len(find_all_triangles(string_to_adjmat(obj, N)))
    return edges - 2 * triangles

def empty_starting_point(N):
    """
    Returns an empty starting point as a string.
    The string consists of "0" repeated for every possible upper-triangle entry.
    """
    return "0" * (N * (N - 1) // 2)

if __name__ == '__main__':
    pass