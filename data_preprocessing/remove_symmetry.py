import pandas as pd

df = pd.read_csv("example_dataset.csv")
df.head()

import pickle

with open("example_graphs_5A.pkl", "rb") as f:
    example_graphs = pickle.load(f)

## remove duplicate homo. but if the epitopes show symmetry. both chains are selected!!!

def homo_chain(graphs, df):
    chain_dict = {}
    for graph, epitope in zip(graphs, df["Epitope"].values):
        print(graph)
        print(graph.graph["chain_ids"])
        chain_list = graph.graph["chain_ids"]
        chain_sequence = []
        for chain in chain_list:
            chain_sequence.append(graph.graph[f"sequence_{chain}"])
        print("# of chains:", len(chain_sequence))
        print("# of unique chain:", len(set(chain_sequence)))
        print()

        if len(set(chain_sequence)) == 1:
            epi_list = []
            for epi in epitope.split(", "):
                print("epi.", epi)
                epi_list.append(epi[0])
            epi_chain_list = list(set(epi_list))
            print()
            print("epi_chain_list: ", epi_chain_list)
            chain_dict[graph.graph["pdb_id"]] = epi_chain_list
        else:
            chain_dict[graph.graph["pdb_id"]] = graph.graph["chain_ids"]

        print("="*50)
    return chain_dict

with open("example_chain_dict.pkl", "wb") as f:
    pickle.dump(example_chain_dict, f)

from graphein.protein.subgraphs import extract_subgraph_from_chains

def subgraph(graphs, chain_dict):
    s_g_list = []
    for g in graphs:
        pdb = g.graph["pdb_id"]
        print(pdb)
        epi_chain = chain_dict[pdb]
        print(epi_chain)
        if len(epi_chain) == 1:
            s_g = extract_subgraph_from_chains(g, epi_chain)
            assert list(s_g.nodes)[0][0] == list(s_g.nodes)[-1][0]
            print("chain:", list(s_g.nodes)[0][0])
            print("="*50)
            s_g_list.append(s_g)
        else:
            s_g = g
            chain_list = []
            for n, d in s_g.nodes(data=True):
                chain_list.append(n[0])
            print(list(set(chain_list)))
            print("="*50)
            s_g_list.append(s_g)
    return s_g_list


example_subgraph = subgraph(example_graphs, example_chain_dict)


with open("example_nonsym_graphs_5A.pkl", "wb") as f:
    pickle.dump(example_subgraph, f)