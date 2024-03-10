import torch

EPS = 1e-6  # Define EPS constant

def read_dat_file(file_path,eps):
    with open(file_path, 'r') as f:
        data = f.read().split()

    # Extract size of the matrices
    N = int(data.pop(0))

    # Read distance matrix
    distance_matrix = []
    for _ in range(N):
        row = []
        for _ in range(N):
            value = float(data.pop(0))
            if value == 0:
                value += eps
            row.append(value)
        distance_matrix.append(row)
    distance_matrix = torch.tensor(distance_matrix)

    # Read flow matrix
    flow_matrix = []
    for _ in range(N):
        row = []
        for _ in range(N):
            value = float(data.pop(0))
            if value == 0:
                value += eps
            row.append(value)
        flow_matrix.append(row)
    flow_matrix = torch.tensor(flow_matrix)

    return N, distance_matrix, flow_matrix

# Example usage:
N,distance_matrix, flow_matrix = read_dat_file("/datasets/qapdata/els19.dat")
print(distance_matrix)
print(flow_matrix)
