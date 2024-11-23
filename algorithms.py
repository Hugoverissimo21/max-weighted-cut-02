# %%
def read_graph_ELEARNING(filename):
    graph = []
    with open(filename, 'r') as file:
        # Skip the first line
        if "gset" in filename:
            nodes, edges = file.readline().split()
            nodes = int(nodes)
            edges = int(edges)
        else:
            file.readline()
            file.readline()
            nodes = int(file.readline().split()[0])
            edges = int(file.readline().split()[0])
        for line in file:
            # Split each line into components and convert to appropriate types
            node1, node2, weight = line.split()
            graph.append((int(node1)-1, int(node2)-1, float(weight)))
    return graph, nodes, edges

#path = """graphs (elearning)/SW10000EWD.txt"""
#path = """graphs (gset)/G59.txt"""
#edges, n_nodes, n_edges = read_graph_ELEARNING(path)
#n_nodes, n_edges

#edges = [(0,1,5), (0,3,3), (1,3,4)]
#n_nodes = 4

# %% [markdown]
# ## algoritmo 1: random random PRONTO
# 
# gera `max_iters` soluções aleatórias; tlvz meter esse max para 2^n / qq coisa ?
# 
# escrever complexidade em funcao do max iter, e se usar 2^n / qq coisa, assim posso reescrever com tudo, mas convem dizer q foi escolha minha mas o utilizador pode mudar

# %%
import random

def max_weight_cut(edges, n_nodes, solutions=10000):

    best_solution = None
    best_cut_weight = 0
    seen_solutions = set()

    for _ in range(solutions):
        # Generate a random candidate solution
        partition = {node: random.choice([0, 1]) for node in range(n_nodes)}

        if len(seen_solutions) == 2**(n_nodes): # max possible solutions
            break

        # avoid calculating the same solution multiple times
        partition_hash = frozenset(partition.items())
        if partition_hash in seen_solutions:
            continue

        seen_solutions.add(partition_hash)
        new_cut_weight = sum(weight for node1, node2, weight in edges if partition[node1] != partition[node2])
        if new_cut_weight > best_cut_weight:
            best_cut_weight = new_cut_weight
            best_solution = partition.copy()


    S = set([node for node, part in best_solution.items() if part == 0])
    T = set(range(n_nodes)) - S
    return S, T, best_cut_weight

#S, T, best_cut_weight = max_weight_cut(edges, n_nodes)
#S, T, best_cut_weight


# %% [markdown]
# ## algoritmo 2: simmulated annealing PRONTO
# 
# gera sol inicial, vai alterando os nós, se melhorar altera, se piorar altera com prob exponencial
# 
# dps dizer q pode ser interessenta executar o algoritmo mais que uma vez para testar diferentes solucoes iniciais, pq têm um impacto na convergencia para a sol otima
# 
# é altamente improvael o SA escolher duas vezes consecutivas o mesmo node em grafos grandes, ent n será relevante evitar "solucoes" repertidas

# %%
import random
import math

def simulated_annealing_partition(edges, n_nodes, temperature=1000, cooling_rate=0.995):
    # Step 1: Extract unique nodes
    nodes = range(n_nodes)
    
    # Step 2: Initialize partitions randomly
    partition = {node: random.choice([0, 1]) for node in nodes}
    
    # Initialize current cost
    current_cut = sum(weight for node1, node2, weight in edges if partition[node1] != partition[node2])

    best_partition = partition.copy()
    best_cut = current_cut
    while temperature > 1e-3:
        # Step 3: Randomly select a node to move to the opposite partition
        node = random.choice(nodes)
        partition[node] = 1 - partition[node]  # Flip partition
        
        # Calculate the new cost after swapping
        new_cut = sum(weight for node1, node2, weight in edges if partition[node1] != partition[node2])
        
        # Step 4: Determine if we should accept the new solution
        cost_diff = new_cut - current_cut
        if cost_diff > 0 or random.random() < math.exp(cost_diff / temperature):
            # Accept the move
            current_cut = new_cut
            # Update the best partition if new cost is lower
            if new_cut > best_cut:
                best_cut = new_cut
                best_partition = partition.copy()
        else:
            # Reject the move (revert the change)
            partition[node] = 1 - partition[node]
        
        # Step 5: Cool down the temperature
        temperature *= cooling_rate
    
    S = set([node for node, part in best_partition.items() if part == 0])
    T = set(range(n_nodes)) - S
    return S, T, best_cut

#S, T, best_cut = simulated_annealing_partition(edges, n_nodes)
#S, T, best_cut

# %% [markdown]
# ## algoritmo 3: Random Greedy PRONTO
# 
# faz heuristicas (mandar para o outro lado o nó q melhora mais a sol otima) a partir de uma solucao inicial gerada aleatoriamente
# 
# pode ser util gerar multiplas solucoes iniciais

# %%
import random

def random_greedy(edges, n_nodes, itLim = 2):
    """Performs Random Greedy optimization for the Max Weight Cut problem."""

    partition = {node: random.choice([0, 1]) for node in range(n_nodes)}
    cut_weight = sum(weight for node1, node2, weight in edges if partition[node1] != partition[node2])

    improved = True
    it_limit = len(edges) * itLim
    while improved and it_limit > 0:
        it_limit -= 1
        improved = False
        for node in range(n_nodes):
            # Flip the node to the other set
            partition[node] = 1 - partition[node]  
            new_cut_weight = sum(weight for node1, node2, weight in edges if partition[node1] != partition[node2])

            # If this move improves the cut weight, keep it; otherwise, revert
            if new_cut_weight > cut_weight:
                cut_weight = new_cut_weight
                improved = True  # Continue improving
                break
            else:
                partition[node] = 1 - partition[node]  # Revert the change

    S = set([node for node, part in partition.items() if part == 0])
    T = set(range(n_nodes)) - S
    return S, T, cut_weight

#S, T, cut_weight = random_greedy(edges, n_nodes)
#S, T, cut_weight

# %%



