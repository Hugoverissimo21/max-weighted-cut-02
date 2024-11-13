# %%
import networkx as nx

#teste para os Gset
G = nx.adjacency_matrix(nx.read_graphml("graphs/0004_500.graphml")).todense()
G[0,1] = -1
G[1,0] = -1
G[1,-1] = -1
G[-1,1] = -1
G


# %%
import itertools

def exhaustive_search(G):
    SOLTESTED = 0
    OPSEXEC = 0
    input_set = set(range(len(G)))
    subsets = []
    n = len(input_set)
    # becasue of the set operation: input_set
    OPSEXEC += n
    
    # generate all subsets (complexy 2^V to generate * V to convert to set = O(V^2 * V))
    for r in range(n + 1):
        for subset in itertools.combinations(input_set, r):
            subsets.append(set(subset))
            OPSEXEC += 1

    best = input_set
    weight = 0
    for subset in subsets: # 2^n resultados para percorrer
        new_weight = 0
        for s in subset: # n^2 para calcular o peso
            for t in input_set - subset:
                new_weight += G[s][t]
                OPSEXEC += 1
        SOLTESTED += 1
        if new_weight > weight:
            best = subset
            weight = new_weight
            OPSEXEC += 2
    
    return best, input_set-best, weight, SOLTESTED, OPSEXEC

exhaustive_search(G) # teste a ver se tudo ok


# %%
def max_weighted_cut_greedy(G):
    OPSEXEC = 0
    num_vertices = len(G)
    
    # Step 1: Extract edges and their weights
    edges = []
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):  # To avoid duplicate edges
            edges.append((i, j, G[i, j]))
            OPSEXEC += 1

    # Step 2: Sort edges in descending order based on their weights
    edges.sort(key=lambda e: e[2], reverse=True)
    OPSEXEC += len(edges) * (len(edges).bit_length() - 1) #n log n
    
    seen, S, T = set(), set(), set()

    cut_weight = 0
    # Step 3: Process each edge
    for u, v, weight in edges:
        if u not in seen and v not in seen: 
            # nenhum vertice visto
            cut_weight += weight
            seen.update({u,v})
            S.add(u)
            T.add(v)
            OPSEXEC += 4
        elif u in S and v not in seen:
            # u no primeiro set, v não visto
            cut_weight += weight
            seen.add(v)
            T.add(v)
            OPSEXEC += 3
        elif u in T and v not in seen:
            # u no segundo set, v não visto
            cut_weight += weight
            seen.add(v)
            S.add(v)
            OPSEXEC += 3
        elif v in S and u not in seen:
            # v no primeiro set, u não visto
            cut_weight += weight
            seen.add(u)
            T.add(u)
            OPSEXEC += 3
        elif v in T and u not in seen:
            # v no segundo set, u não visto
            cut_weight += weight
            seen.add(u)
            S.add(u)
            OPSEXEC += 3
        elif v in T and u in S:
            cut_weight += weight
            OPSEXEC += 1
        elif v in S and u in T:
            cut_weight += weight
            OPSEXEC += 1
        # v and u in the same set

    return S, T, cut_weight, OPSEXEC

max_weighted_cut_greedy(G) # teste a ver se tudo ok

# %% [markdown]
# ---
# ---
# ---
# BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS
# BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS
# BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS BENCHMARKS
# ---
# ---
# ---

# %% [markdown]
# ## My graphs

# %%
import os
import numpy as np
import time
import pandas as pd
from datetime import datetime
import random

graphs = sorted([f"graphs/{x}" for x in os.listdir("graphs") if x[-7:] == "graphml"])#[20*3:30*4]
#graphs = []
#random.shuffle(graphs)

def countEdges(G):
    return int(np.count_nonzero(G) / 2)

data = {
    ("Graph", "name"): [],
    ("Graph", "n"): [],
    ("Graph", "m"): [],
    ("Exhaustive", "#operations"): [],
    ("Exhaustive", "exec_time"): [],
    ("Exhaustive", "#solutions tested"): [],
    ("Heuristic", "#operations"): [],
    ("Heuristic", "exec_time"): [],
    ("Heuristic", "#solutions tested"): [],
    ("greedy_prec", " "): []
}

dateFile = str(datetime.now().strftime("%m%d%H%M%S"))
for graph in graphs:
    print(f"Solving {graph}: {datetime.now().strftime('%H%M')}")
    G = nx.adjacency_matrix(nx.read_graphml(graph)).todense()
    start_time = time.time()
    S1, T1, w1, SOLTEST1, OPS1 = exhaustive_search(G)
    timeEx = time.time() - start_time
    start_time = time.time()
    S2, T2, w2, OPS2 = max_weighted_cut_greedy(G)
    timeHeu = time.time() - start_time
    #
    data[("Graph", "name")].append(graph)
    data[("Graph", "n")].append(len(G))
    data[("Graph", "m")].append(countEdges(G))
    data[("Exhaustive", "#operations")].append(OPS1) #OPS1 #np.nan
    data[("Exhaustive", "exec_time")].append(timeEx) #timeEx #np.nan
    data[("Exhaustive", "#solutions tested")].append(SOLTEST1) #SOLTEST1 #np.nan
    data[("Heuristic", "#operations")].append(OPS2)
    data[("Heuristic", "exec_time")].append(timeHeu)
    data[("Heuristic", "#solutions tested")].append(1)
    data["greedy_prec", " "].append(w2/w1 if w1 != 0 else 1) #w2/w1 if w1 != 0 else 1 #np.nan
    

    df = pd.DataFrame(data)
    df.to_excel(f"results_{dateFile}.xlsx")

print("Done! i hope...")

# %% [markdown]
# ## Gset Graphs

# %%

data = {
    ("Graph", "name"): [],
    ("Graph", "n"): [],
    ("Graph", "m"): [],
    ("Exhaustive", "#operations"): [],
    ("Exhaustive", "exec_time"): [],
    ("Exhaustive", "#solutions tested"): [],
    ("Heuristic", "#operations"): [],
    ("Heuristic", "exec_time"): [],
    ("Heuristic", "#solutions tested"): [],
    ("greedy_prec", " "): []
}

df = pd.read_csv("gset/description.csv", index_col=0)
dateFile = str(datetime.now().strftime("%m%d%H%M%S"))
for row in df.iterrows():
    graph = row[1]["instance"]
    n = row[1]["#vertices"]
    m = row[1]["#edges"]
    optimal = row[1]["Best known"]
    
    with open(f"gset/{graph}.txt", 'r') as file:
        first_line = file.readline().split()
        if int(first_line[0]) == int(n):
            pass
        else:
            print(f"Error in {graph}!!!")
            break
        adjacency = np.zeros((n,n))
        for line in file:
            i,j,w = line.strip().split(" ")
            adjacency[int(i)-1, int(j)-1] = int(w)
            adjacency[int(j)-1, int(i)-1] = int(w)
    
    print(f"Solving {graph}: {datetime.now().strftime('%H%M')}")
    start_time = time.time()
    S2, T2, w2, OPS2 = max_weighted_cut_greedy(adjacency)
    timeHeu = time.time() - start_time
    #
    data[("Graph", "name")].append(f"gset/{graph}")
    data[("Graph", "n")].append(n)
    data[("Graph", "m")].append(m)
    data[("Exhaustive", "#operations")].append(np.nan)
    data[("Exhaustive", "exec_time")].append(np.nan)
    data[("Exhaustive", "#solutions tested")].append(np.nan)
    data[("Heuristic", "#operations")].append(OPS2)
    data[("Heuristic", "exec_time")].append(timeHeu)
    data[("Heuristic", "#solutions tested")].append(1)
    data["greedy_prec", " "].append(w2/optimal if optimal != 0 else np.nan)
    

    df = pd.DataFrame(data)
    df.to_excel(f"results_{dateFile}.xlsx")

print("Done! i hope...")

# %%
#dfs = [pd.read_excel(x, header=[0,1], index_col=0) for x in os.listdir() if x[:7] == "results" and x != "results.xlsx"]
#merged_df = pd.concat(dfs, ignore_index=True)
#merged_df.to_excel("Aresults.xlsx")
#merged_df

# %%



