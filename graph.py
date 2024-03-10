import json
import matplotlib.pyplot as plt

# Load data from JSON
with open('data.json', 'r') as file:
    data = json.load(file)

# Filter data for specific points on x-axis
filtered_data = [entry for entry in data if entry['ants'] in [10, 20, 30]]

# Extracting relevant data
ants = [entry['ants'] for entry in filtered_data]
opt_costs = [entry['opt_cost'] for entry in filtered_data]
vanilla_aco_costs = [entry['vanilla_aco'] for entry in filtered_data]
aco_with_heu_costs = [entry['aco_with_heu_cost'] for entry in filtered_data]

# Plotting the graph
plt.plot(ants, opt_costs, label='Optimal Cost', marker='o')
plt.plot(ants, vanilla_aco_costs, label='Vanilla ACO', marker='o')
plt.plot(ants, aco_with_heu_costs, label='Deep ACO', marker='o')

# Adding labels and title
plt.xlabel('Number of Ants')
plt.ylabel('Cost')
plt.title('Comparison of Costs with Different Number of Ants(nug14.dat)')
plt.legend()

# Displaying the plot
plt.grid(True)
plt.xticks(ants)  # Set x-ticks to show only specified points

# Save the plot
plt.savefig('plot_nug14.png')

# Show the plot
plt.show()
