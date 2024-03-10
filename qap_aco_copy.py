import numpy as np
import torch
from utils import *
from net import Net
np.random.seed = 1

EPS = 1e-10

class QAP_ACO:
    def __init__(self, distance_matrix, flow_matrix, n_ants=20, n_iterations=100,heuristic=None):
        self.distance_matrix = distance_matrix
        self.flow_matrix = flow_matrix
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.n = distance_matrix.shape[0]
        self.pheromone = torch.ones_like(flow_matrix)
        self.heuristic = heuristic if heuristic!=None else flow_matrix
    
    def run(self):
        # pheromone = np.ones((self.n, self.n))
        best_solution = None
        best_cost = float('inf')
        
        for _ in range(self.n_iterations):
            solutions,log_prob_list = self.generate_solutions()
            costs = [self.calculate_cost(solution) for solution in solutions]
            
            local_best_index = np.argmin(costs)
            local_best_solution = solutions[local_best_index]
            local_best_log_prob = log_prob_list[local_best_index]
            local_best_cost = costs[local_best_index]
            
            if local_best_cost < best_cost:
                best_solution = local_best_solution
                best_cost = local_best_cost
            
            pheromone = self.update_pheromone( solutions, costs)
        
        return best_solution, best_cost
    
    def sample(self):
        paths, log_probs = self.generate_solutions()
        costs = [self.calculate_cost(path) for path in paths]
        costs = torch.tensor(costs,requires_grad=True)
        log_probs = torch.tensor(log_probs,requires_grad=True)
        return costs, log_probs
    
    def generate_solutions(self):
        solutions = []
        log_prob_list =  []
        for _ in range(self.n_ants):
            solution,log_prob= self.construct_solution()
            solutions.append(solution)
            log_prob_list.append(log_prob)
        return solutions,log_prob_list
    
    def construct_solution(self):
        visited = set()
        solution = []
        log_prob = 0
        for _ in range(self.n):
            # print(f"len visited {len(visited)}")
            valid_choices = [i for i in range(self.n) if i not in visited]
            # print(f"valid choices: {valid_choices}")
            # print(f"pheromones {self.pheromone[solution[-1],:] if solution else None}")
            # print(f"heuristic {self.heuristic[solution[-1],:] if solution else None}")
            probabilities = [self.pheromone[solution[-1], j].item() * self.heuristic[solution[-1],j].item() \
                for j in valid_choices] if solution else torch.ones(self.n)
            # print(f"probablity raw: {probabilities}")
            probabilities = torch.tensor(probabilities,requires_grad=True)
            

            probabilities = probabilities / torch.sum(probabilities)
            # print(f"probabilites: {probabilities}")
            # print(f"Length of single probability {probabilities[0]}")
            choice = np.random.choice(valid_choices, p=probabilities.detach().numpy())
            choice_probability = probabilities[valid_choices.index(choice)]
            log_prob+=torch.log(choice_probability)
            solution.append(choice)
            visited.add(choice)
        return solution,log_prob
    
    def calculate_cost(self, solution):
        cost = 0
        for i in range(self.n):
            for j in range(self.n):
                cost += self.distance_matrix[i, j] * self.flow_matrix[solution[i], solution[j]]
        return cost
    
    # def update_pheromone(self, solutions, costs):
    #     evaporation_rate = 0.5
    #     new_pheromone = (1 - evaporation_rate) * self.pheromone
    #     best_solution_index = np.argmin(costs)
    #     best_solution = solutions[best_solution_index]
    #     for i in range(self.n):
    #         new_pheromone[i, best_solution[i]] += 1 / costs[best_solution_index]
    #     return new_pheromone
    
    def update_pheromone(self, solutions, costs):
        evaporation_rate = 0.2
        new_pheromone = (1 - evaporation_rate) * self.pheromone
        best_solution_index = np.argmin(costs)
        best_solution = solutions[best_solution_index]
        # print(f"best solution cost {costs[best_solution_index]}")
        # print(f"best solution {best_solution}")
        for i in range(self.n):
            for j in range(self.n):
                new_pheromone[i, j] += (1.0 / costs[best_solution_index]) if best_solution[i] == j else 0 
        # print(f"new pheromone: {new_pheromone}")
        return new_pheromone

def main():
    # Generate a QAP instance
    n = 10
    print(f"n = {n}")
    flow_matrix,distance_matrix = gen_instance(n,'cpu')
    print(flow_matrix)
    pyg_data = gen_pyg_data(flow_matrix,distance_matrix,'cpu')
    # flow_matrix=torch.tensor([[.1,100,2],[100,.1,3],[2,3,.1]])
    # distance_matrix=torch.tensor([[.1,2,100],[2,.1,105],[100,105,.1]])
    
    # Initialize and run ACO algorithm
    aco_solver = QAP_ACO(distance_matrix, flow_matrix)
    best_solution, best_cost = aco_solver.run()
    print("without deep aco")
    print("Best solution found: location to facility: ", best_solution)
    print("Cost of best solution:", best_cost)
    
    model = Net()
    heu = model(pyg_data)
    heu = model.reshape(pyg_data, heu) + EPS
    
    print(f"heu shape: {heu.shape}")
    print(f"heu: {heu}")
    
    aco_solver = QAP_ACO(distance_matrix, flow_matrix,20,100,heu)
    best_solution, best_cost = aco_solver.run()
    print("with deep aco")
    print("Best solution found: location to facility: ", best_solution)
    print("Cost of best solution:", best_cost)

if __name__ == "__main__":
    main()