# %% [markdown]
# functions to create and draw graphs

# %%
import networkx as nx
import matplotlib.pyplot as plt
import random

def create_graph(V, E, name):
    """cria o grafo e da peso das edges"""
    # The limit for x and y coordinates in the 2D space
    space_limit = 1000
    # Create a random graph with n nodes and e edges
    G = nx.gnm_random_graph(V, E)
    # Assign random weights to each edge
    for (u, v) in G.edges():
        G[u][v]['weight'] = random.randint(1, 20)  # Random weight between 1 and 20
    nx.write_graphml(G, name + ".graphml")
    return G

def draw_graph(G, name, show = False):
    """salva desenho do grafo, com posicoes random"""
    space_limit = 1000 #might me important
    # Assign random (x, y) positions to each node in a 1000x1000 grid
    positions = {i: (random.randint(0, space_limit), random.randint(0, space_limit)) for i in G.nodes()} #might me important
    # Draw the graph with the node positions based on the coordinates
    plt.figure(figsize=(7, 7))
    plt.grid(True, alpha=0.3)
    # Set transparent background for the figure and axes
    plt.gcf().patch.set_alpha(0)  # Make the figure background transparent
    plt.gca().patch.set_alpha(0)  # Make the axis background transparent
    # Get edge weights
    edge_weights = nx.get_edge_attributes(G, 'weight')
    # Draw nodes and edges
    nx.draw(G, pos=positions, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_weights)
    # Set grid, axis limits, and axis ticks
      # Enable grid with some transparency
    plt.xticks(range(0, space_limit + 1, 100))  # Set X-axis ticks from 0 to 1000, every 100 units
    plt.yticks(range(0, space_limit + 1, 100))  # Set Y-axis ticks from 0 to 1000, every 100 units
    plt.xlim(0, space_limit)  # Set X-axis limit
    plt.ylim(0, space_limit)  # Set Y-axis limit
    # Make sure tick labels (numbers) are visible and properly formatted
    plt.tick_params(axis='both', which='both', labelsize=10)  # Set the size of tick labels
    # Re-enable axis visibility since networkx disables it by default
    plt.gca().set_axis_on()
    plt.tight_layout()
    plt.savefig(name + ".svg", format="svg", transparent=True)  # or png
    if show:
        plt.show()
    else:
        plt.close()

# %% [markdown]
# function to automate graphs creating and saving them

# %%
V = list(range(4,50)) + list(range(50,100,5)) + list(range(100,500,50)) + list(range(500,1001, 100)) # #Nodes
E = [0.125, 0.25, 0.5, 0.75] # %Edges

for v in V:
    for e in E:
        random.seed(124348)
        name = f"graphs/{str(v).zfill(4)}_{int(e*1000)}"
        G = create_graph(v, int(e * ((v * (v-1)) / 2)), name)
        draw_graph(G, name, show = False)
    print(f"v={v} OK", end = " | ")

# %%
#nx.adjacency_matrix(G).todense()

#G_loaded_graphml = nx.read_graphml("h.graphml")
#nx.adjacency_matrix(G_loaded_graphml).todense()


