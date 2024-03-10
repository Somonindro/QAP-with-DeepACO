import torch
from torch_geometric.data import Data

# CAPACITY = 50
FLOW_LOW = 1
FLOW_HIGH = 9
EPS = 1e-10

def gen_instance(n, device):
    locations = torch.rand(size=(n, 2), device=device)
    # flows = torch.randint(low=FLOW_LOW, high=FLOW_HIGH+1, size=(n,n), device=device)
    flows = torch.rand(n,n) * (FLOW_HIGH-FLOW_LOW) + FLOW_LOW
    flows = torch.floor(0.5 *(flows+flows.T))
    flows[torch.arange(n),torch.arange(n)] = EPS # flow to itself is 0
    # depot = torch.tensor([DEPOT_COOR], device=device)
    all_locations = locations
    all_flows = flows
    distances = gen_distance_matrix(all_locations)
    return all_flows, distances # (n,n), (n, n)

def gen_distance_matrix(location_coordinates):
    n_nodes = len(location_coordinates)
    distances = torch.norm(location_coordinates[:, None] - location_coordinates, dim=2, p=2)
    # note, here
    # setting the distance of a node to itself to a small value
    distances[torch.arange(n_nodes), torch.arange(n_nodes)] = EPS
    return distances

def gen_pyg_data(flows, distances, device):
    
    n = flows.shape[0]

    flow_nodes = torch.arange(n)
    # location_nodes = torch.arange(n,2*n)

    flow_u = flow_nodes.repeat(n)
    flow_v = torch.repeat_interleave(flow_nodes,n)

    flow_edge_index = torch.stack((flow_u,flow_v))

    # location_u = location_nodes.repeat(n)
    # location_v = torch.repeat_interleave(location_nodes,n)

    # location_edge_index = torch.stack((location_u,location_v))

    # edge_index  = torch.concat([flow_edge_index,location_edge_index],dim = 1)
    edge_index = flow_edge_index

    flow_edge_attr = flows.reshape((n*n,1))
    # location_edge_attr = distances.reshape((n*n,1))

    # edge_attr = torch.concat([flow_edge_attr,location_edge_attr],dim=0)
    edge_attr = flow_edge_attr

    # print(f"flows {flows}")
    # print(f"distances {distances}")
    # print(f"flow nodes {flow_nodes}")
    # print(f"location nodes {location_nodes}")
    # print(f"flow_edge_index {flow_edge_index}")
    # print(f"location edge index {location_edge_index}")
    # print(f"edge index {edge_index}")
    # print(f"edge attr {edge_attr}")

    # TODO - fix x
    x = torch.ones(n,1)

    # print(torch.tensor([[1,2],[3,4]]).reshape((4,1)))

    qap_pyg_data = Data(x,edge_index = edge_index,edge_attr=edge_attr,num_nodes=n)
    return qap_pyg_data

def load_test_dataset(problem_size, device):
    test_list = []
    dataset = torch.load(f'../data/cvrp/testDataset-{problem_size}.pt', map_location=device)
    for i in range(len(dataset)):
        test_list.append((dataset[i, 0, :], dataset[i, 1:, :]))
    return test_list


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

if __name__ == '__main__':
    import pathlib
    pathlib.Path('../data/cvrp').mkdir(parents=False, exist_ok=True) 
    torch.manual_seed(123456)
    for n in [20, 100, 500]:
        inst_list = []
        for _ in range(100):
            demands, distances = gen_instance(n, 'cpu')
            inst = torch.cat((demands.unsqueeze(0), distances), dim=0) # (n+2, n+1)
            inst_list.append(inst)
        testDataset = torch.stack(inst_list)
        torch.save(testDataset, f'../data/cvrp/testDataset-{n}.pt')
        