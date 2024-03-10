import torch
from net import Net
from qap_aco_copy import QAP_ACO
from utils import gen_instance,gen_pyg_data,EPS,read_dat_file

device = 'cpu'
lr = 0.001

def train_instance(model, optimizer, pyg_data, distances,flows, n_ants):
    model.train()
    heu_vec = model(pyg_data)
    heu_mat = model.reshape(pyg_data, heu_vec) + EPS
    
    aco = QAP_ACO(
        distances,
        flows,
        n_ants,
        100,
        heu_mat
    )
    
    costs, log_probs = aco.sample()
    baseline = costs.mean()
    reinforce_loss = torch.sum((costs - baseline) * log_probs.sum(dim=0)) / aco.n_ants
    # print(f"reinforce loss: {reinforce_loss}")
    optimizer.zero_grad()
    reinforce_loss.backward()
    optimizer.step()
    

def train_epoch(n_node,
                n_ants, 
                epoch, 
                steps_per_epoch, 
                net, 
                optimizer
                ):
    for _ in range(steps_per_epoch):
        flows,distances = gen_instance(n_node,device)
        data= gen_pyg_data(flows,distances,device)
        train_instance(net,optimizer,data,distances,flows,n_ants)
        print(f"{_}/{steps_per_epoch} done")
        

def train(n_node, n_ants, steps_per_epoch, epochs):
    net = Net().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    
    for epoch in range(0, epochs):
        train_epoch(n_node, n_ants, epoch, steps_per_epoch, net, optimizer)
    
    with torch.no_grad():
        n,flows,distances = read_dat_file("./datasets/qapdata/els19.dat",EPS)
        data= gen_pyg_data(flows,distances,device)
        heu_vec = net(data)
        heu_mat = net.reshape(data, heu_vec) + EPS
        
        aco_without_heu = QAP_ACO(
            distances,
            flows,
            n_ants,
            100
        )
        
        aco_with_heu = QAP_ACO(
            distances,
            flows,
            n_ants,
            100,
            heu_mat
        )
        
        best_solution_no_heu,best_cost_no_heu = aco_without_heu.run()
        best_solution_heu,best_cost_heu = aco_with_heu.run()
        
        print(f"Without heu: {best_solution_no_heu} \n cost:  {best_cost_no_heu}")
        print(f"With heu: {best_solution_heu} \n cost:  {best_cost_heu}")

train(19,10,500,10)