import numpy as np

class ACO_QAP:
    def __init__(self, dist_matrix, flow_matrix, num_ants, num_iterations, alpha=1.0, beta=2.0, rho=0.5, q=100):
        self.dist_matrix = dist_matrix
        self.flow_matrix = flow_matrix
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q

    def initialize_pheromone_matrix(self):
        n = len(self.dist_matrix)
        return np.ones((n, n))

    def run(self):
        n = len(self.dist_matrix)
        pheromone_matrix = self.initialize_pheromone_matrix()
        best_solution = None
        best_cost = float('inf')

        for iter in range(self.num_iterations):
            solutions = []
            costs = []

            for ant in range(self.num_ants):
                visited = [False] * n
                solution = []
                total_cost = 0

                start = np.random.randint(0, n)
                current_node = start
                visited[current_node] = True
                solution.append(current_node)
                
                unvisited = [i for i in range(n) if not visited[i]]  # Compute unvisited nodes outside the loop

                for _ in range(n - 1):
                    probs = np.zeros(len(unvisited))  # Initialize probs with the correct size
                    for j in unvisited:
                        probs[j] = (pheromone_matrix[current_node, j] ** self.alpha) * \
                                   (1.0 / self.dist_matrix[current_node, j] ** self.beta)
                    
                    probs = probs / np.sum(probs)
                    next_node = np.random.choice(unvisited, p=probs)
                    visited[next_node] = True
                    solution.append(next_node)
                    total_cost += self.flow_matrix[current_node, next_node] * self.dist_matrix[current_node, next_node]
                    current_node = next_node

                    # Update unvisited nodes
                    unvisited = [i for i in range(n) if not visited[i]]

                total_cost += self.flow_matrix[solution[-1], solution[0]] * self.dist_matrix[solution[-1], solution[0]]

                solutions.append(solution)
                costs.append(total_cost)

                if total_cost < best_cost:
                    best_solution = solution
                    best_cost = total_cost

            pheromone_matrix *= (1 - self.rho)

            for i in range(self.num_ants):
                for j in range(n - 1):
                    pheromone_matrix[solutions[i][j], solutions[i][j + 1]] += self.q / costs[i]
                pheromone_matrix[solutions[i][-1], solutions[i][0]] += self.q / costs[i]

        return best_solution, best_cost

# Example usage:
dist_matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
flow_matrix = np.array([[0, 5, 2], [5, 0, 1], [2, 1, 0]])
num_ants = 10
num_iterations = 100
alpha = 1.0
beta = 2.0
rho = 0.5
q = 100

aco = ACO_QAP(dist_matrix, flow_matrix, num_ants, num_iterations, alpha, beta, rho, q)
best_solution, best_cost = aco.run()
print("Best solution:", best_solution)
print("Best cost:", best_cost)
