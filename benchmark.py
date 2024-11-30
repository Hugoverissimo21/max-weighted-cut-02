# %%
import networkx as nx
import random
import math

# %% [markdown]
# # funcoes para ler grafos

# %%
def adjacency_matrix_to_edges(G_graphml):
    matrix = nx.adjacency_matrix(nx.read_graphml(G_graphml)).todense()
    edges = []
    n = len(matrix)  # assuming matrix is square
    for u in range(n):
        for v in range(u, n):
            if matrix[u][v] != 0:  # assuming 0 means no edge
                edges.append((u, v, matrix[u][v]))
    return edges, n, len(edges)

#G = 'graphs/0004_125.graphml'
#edges, n_nodes, m = adjacency_matrix_to_edges(G)

# %%
def edges_from_Bgraph(filename):
    edges = []
    with open(filename, 'r') as file:
        # Skip the first line
        if "gset" in filename:
            n_nodes, n_edges = file.readline().split()
            n_nodes, n_edges = int(n_nodes), int(n_edges)
        else:
            file.readline()
            file.readline()
            n_nodes = int(file.readline().split()[0])
            n_edges = int(file.readline().split()[0])
        for line in file:
            # Split each line into components and convert to appropriate types
            node1, node2, weight = line.split()
            edges.append((int(node1)-1, int(node2)-1, float(weight)))
    return edges, n_nodes, n_edges

#path = """graphs (gset)/G59.txt"""
#edges_from_Bgraph(path)


# %% [markdown]
# # algorimos

# %%
def random_solution(edges, n_nodes, solutions=10000):
    SOLTESTED, OPSEXEC = 0, 0

    best_solution = {node: 0 for node in range(n_nodes)}
    best_cut_weight = 0
    seen_solutions = set()

    MEAN_WEIGHT = 0
    for _ in range(solutions):
        # Generate a random candidate solution
        partition = {node: random.choice([0, 1]) for node in range(n_nodes)}
        OPSEXEC += n_nodes
        # avoid calculating the same solution multiple times
        partition_hash = frozenset(partition.items()) # hash
        # nao vou contabilizar o custo de calcular o hash pq não seria necessário se só testasse uma solucao
        # e em grafos de maior dimensao, a probabilidade de ter solucoes iguais tende para 0 (2^n possiblidades)

        if len(seen_solutions) == 2**(n_nodes): # max possible solutions
            break
        
        if partition_hash in seen_solutions:
            continue

        
        

        seen_solutions.add(partition_hash)
        OPSEXEC += 1
        SOLTESTED += 1
        new_cut_weight = sum(weight for node1, node2, weight in edges if partition[node1] != partition[node2])
        MEAN_WEIGHT += new_cut_weight
        OPSEXEC += len(edges)
        if new_cut_weight > best_cut_weight:
            best_cut_weight = new_cut_weight
            best_solution = partition.copy()
            OPSEXEC += 2

    actual_it = _ + 1
    S = set([node for node, part in best_solution.items() if part == 0]) #pos processamento, depende do q ser quer, n conta
    T = set(range(n_nodes)) - S #pos processamento, depende do q ser quer, n conta
    return S, T, best_cut_weight, SOLTESTED, OPSEXEC, actual_it, MEAN_WEIGHT/actual_it

#S, T, best_cut_weight, SOLTESTED, OPSEXEC, actual_it, MEAN_WEIGHT = random_solution(edges, n_nodes)
#S, T, best_cut_weight, SOLTESTED, OPSEXEC, actual_it, MEAN_WEIGHT


# %%
def sim_anlng(edges, n_nodes, initial_temp=1000, cooling_rate=0.995, min_temp=1e-3):
    SOLTESTED, OPSEXEC = 0, 0

    nodes = range(n_nodes)
    
    partition = {node: random.choice([0, 1]) for node in nodes}
    OPSEXEC += n_nodes
    
    # Initialize current cost
    current_cut = sum(weight for node1, node2, weight in edges if partition[node1] != partition[node2])
    OPSEXEC += len(edges)
    SOLTESTED += 1

    temperature = initial_temp

    best_partition = partition.copy()
    best_cut = current_cut
    while temperature > min_temp:

        node = random.choice(nodes)
        partition[node] = 1 - partition[node] 
        OPSEXEC += 2
        
        new_cut = sum(weight for node1, node2, weight in edges if partition[node1] != partition[node2])
        OPSEXEC += len(edges)
        SOLTESTED += 1
        
        cost_diff = new_cut - current_cut
        if cost_diff > 0 or random.random() < math.exp(cost_diff / temperature):
            current_cut = new_cut
            if new_cut > best_cut:
                best_cut = new_cut
                best_partition = partition.copy()
                OPSEXEC += 5
        else:
            partition[node] = 1 - partition[node]
            OPSEXEC += 3
        
        temperature *= cooling_rate
        OPSEXEC += 1
    
    S = set([node for node, part in best_partition.items() if part == 0])
    T = set(range(n_nodes)) - S
    return S, T, best_cut, SOLTESTED, OPSEXEC

#S, T, best_cut, SOLTESTED, OPSEXEC = sim_anlng(edges, n_nodes)
#S, T, best_cut, SOLTESTED, OPSEXEC

# %%
def random_greedy(edges, n_nodes, itLim = 2):
    SOLTESTED, OPSEXEC = 0, 0

    partition = {node: random.choice([0, 1]) for node in range(n_nodes)}
    cut_weight = sum(weight for node1, node2, weight in edges if partition[node1] != partition[node2])
    SOLTESTED += 1
    OPSEXEC += len(edges) + n_nodes

    improved = True
    it_limit = len(edges) * itLim
    OPSEXEC += 1
    while improved and it_limit > 0:
        it_limit -= 1
        OPSEXEC += 1
        improved = False
        for node in range(n_nodes):
            # Flip the node to the other set
            partition[node] = 1 - partition[node]  
            new_cut_weight = sum(weight for node1, node2, weight in edges if partition[node1] != partition[node2])
            SOLTESTED += 1
            OPSEXEC += len(edges) + 1

            # If this move improves the cut weight, keep it; otherwise, revert
            if new_cut_weight > cut_weight:
                cut_weight = new_cut_weight
                improved = True  # Continue improving
                OPSEXEC += 2
                break
            else:
                partition[node] = 1 - partition[node]
                OPSEXEC += 1

    S = set([node for node, part in partition.items() if part == 0])
    T = set(range(n_nodes)) - S
    return S, T, cut_weight, SOLTESTED, OPSEXEC

#S, T, cut_weight, SOLTESTED, OPSEXEC = random_greedy(edges, n_nodes)
#S, T, cut_weight, SOLTESTED, OPSEXEC

# %% [markdown]
# # benchmarks

# %%
import pandas as pd
from datetime import datetime
import time

df = pd.read_excel('results v00.xlsx', index_col=0, header=[0,1]) # do trabalho 1
#df.head()

# %%
import numpy as np
dateFile = str(datetime.now().strftime("%m%d%H%M%S")) 

alg1_ops = [np.nan for _ in range(len(df))]
alg1_time = [np.nan for _ in range(len(df))]
alg1_sols = [np.nan for _ in range(len(df))]
alg1_prec = [np.nan for _ in range(len(df))]
alg1_it = [np.nan for _ in range(len(df))]
alg1_prec_it = [np.nan for _ in range(len(df))]

alg2_ops = [np.nan for _ in range(len(df))]
alg2_time = [np.nan for _ in range(len(df))]
alg2_sols = [np.nan for _ in range(len(df))]
alg2_prec = [np.nan for _ in range(len(df))]

alg3_ops = [np.nan for _ in range(len(df))]
alg3_time = [np.nan for _ in range(len(df))]
alg3_sols = [np.nan for _ in range(len(df))]
alg3_prec = [np.nan for _ in range(len(df))]

indices = np.arange(len(df))
np.random.shuffle(indices)

for i in indices: # range(len(df)) range(len(df)-1, -1, -1) indices
    graph = df[("Graph", "name")][i]
    n = df[("Graph", "n")][i]
    m = df[("Graph", "m")][i]
    weight = df[("Graph", "weight")][i]
    print(f"Solving {graph}: {datetime.now().strftime('%H%M')}")

    # prepare the graph
    if "gset" in graph:
        edges, n_nodes, n_edges = edges_from_Bgraph(graph)
    elif "graphs/" in graph:
        edges, n_nodes, n_edges = adjacency_matrix_to_edges(graph)



    
    max_solutions = 10000       #################################### <--- mudar se quiser
    ALG1 = f"Random Solutions (MS: {max_solutions})"
    start_time = time.time()
    #S, T, best_cut_weight, SOLTESTED, OPSEXEC, ACTUAL_IT, MEAN_WEIGHT = random_solution(edges, n_nodes, solutions=max_solutions)
    timee = time.time() - start_time
    alg1_ops[i] = np.nan #OPSEXEC #OPSEXEC
    alg1_time[i] = np.nan #timee #timee
    alg1_sols[i] = np.nan #SOLTESTED #SOLTESTED
    alg1_prec[i] = np.nan #best_cut_weight/weight #best_cut_weight/weight
    alg1_it[i] = np.nan #ACTUAL_IT #ACTUAL_IT
    alg1_prec_it[i] = np.nan #MEAN_WEIGHT/weight #MEAN_WEIGHT/weight

    
    temperature = 1000          #################################### <--- mudar se quiser
    cooling_rate = 0.99         #################################### <--- mudar se quiser
    ALG2 = f"Simulated Annealing (T: {temperature}, CR: {cooling_rate})"
    start_time = time.time()
    S, T, best_cut, SOLTESTED, OPSEXEC = sim_anlng(edges, n_nodes, initial_temp=temperature, cooling_rate=cooling_rate)
    timee = time.time() - start_time
    alg2_ops[i] = OPSEXEC
    alg2_time[i] = timee
    alg2_sols[i] = SOLTESTED
    alg2_prec[i] = best_cut/weight

    it_limite = 2               #################################### <--- mudar se quiser
    ALG3 = f"Random Greedy (IT: {it_limite})"
    start_time = time.time()
    S, T, cut_weight, SOLTESTED, OPSEXEC = random_greedy(edges, n_nodes, itLim = it_limite)
    timee = time.time() - start_time
    alg3_ops[i] = OPSEXEC #np.nan #OPSEXEC
    alg3_time[i] = timee #timee
    alg3_sols[i] = SOLTESTED #SOLTESTED
    alg3_prec[i] = cut_weight/weight #cut_weight/weight


    df[(ALG1, "#ops")] = alg1_ops
    df[(ALG1, "time")] = alg1_time
    df[(ALG1, "#sols")] = alg1_sols
    df[(ALG1, "prec.")] = alg1_prec
    df[(ALG1, "it.")] = alg1_it
    df[(ALG1, "prec./it.")] = alg1_prec_it

    df[(ALG2, "#ops")] = alg2_ops
    df[(ALG2, "time")] = alg2_time
    df[(ALG2, "#sols")] = alg2_sols
    df[(ALG2, "prec.")] = alg2_prec

    df[(ALG3, "#ops")] = alg3_ops
    df[(ALG3, "time")] = alg3_time
    df[(ALG3, "#sols")] = alg3_sols
    df[(ALG3, "prec.")] = alg3_prec


    df.to_excel(f"results_{dateFile}.xlsx")

print("Done! i hope...")

# %%



