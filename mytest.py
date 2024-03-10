from utils import *
import torch
from torch_geometric.data import Data
from net import Net

device='cpu'

# flows,distances = gen_instance(5,device)

# print(flows)

# print(distances)

# def gen_pyg_data(demands, distances, device):
#     n = demands.size(0)
#     nodes = torch.arange(n, device=device)
#     u = nodes.repeat(n)
#     v = torch.repeat_interleave(nodes, n)
#     edge_index = torch.stack((u, v))
#     edge_attr = distances.reshape(((n)**2, 1))
#     x = demands
#     pyg_data = Data(x=x.unsqueeze(1), edge_attr=edge_attr, edge_index=edge_index)
#     return pyg_data

# def gen_qap_pyg_data(flows,distances,device):
#     n = flows.shape[0]
    

# n = 5
# nodes = torch.arange(n)

# u = nodes.repeat(n)

# v = torch.repeat_interleave(nodes,n)

# edge_index = torch.stack((u,v))

# print(f"u {u}")
# print(f"v {v}")
# print(f"edge index {edge_index}")


# QAP

n = 2

flows,distances=gen_instance(n,device)

flow_nodes = torch.arange(n)
location_nodes = torch.arange(n,2*n)

flow_u = flow_nodes.repeat(n)
flow_v = torch.repeat_interleave(flow_nodes,n)

flow_edge_index = torch.stack((flow_u,flow_v))

location_u = location_nodes.repeat(n)
location_v = torch.repeat_interleave(location_nodes,n)

location_edge_index = torch.stack((location_u,location_v))

edge_index  = torch.concat([flow_edge_index,location_edge_index],dim = 1)


flow_edge_attr = flows.reshape((n*n,1))
location_edge_attr = distances.reshape((n*n,1))

edge_attr = torch.concat([flow_edge_attr,location_edge_attr],dim=0)

print(f"flows {flows}")
print(f"distances {distances}")
print(f"flow nodes {flow_nodes}")
print(f"location nodes {location_nodes}")
print(f"flow_edge_index {flow_edge_index}")
print(f"location edge index {location_edge_index}")
print(f"edge index {edge_index}")
print(f"edge attr {edge_attr}")

# TODO - fix x
x = torch.randn(n*2,1)

# print(torch.tensor([[1,2],[3,4]]).reshape((4,1)))

qap_pyg_data = Data(x,edge_index = edge_index,edge_attr=edge_attr,num_nodes=2 * n)

print(f"QAP pyg data: {qap_pyg_data}")

print(f"QAP pyg x: {qap_pyg_data.x}")

print(f" num of nodes in QAP pyg data: {qap_pyg_data.num_nodes}")

net = Net()

heu = net(qap_pyg_data)

print(heu.shape)
