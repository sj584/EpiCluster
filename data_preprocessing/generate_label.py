import pandas as pd

df = pd.read_csv("example_dataset.csv")
df.head()

import pickle
with open("example_graphs_5A.pkl", "rb") as f:
    graphs = pickle.load(f)

pdb_epi_label_list = []
for graph, epitopes in zip(graphs, df["Epitope"]):
    epi_label_dict = {}
    print(epitopes.split(", "))
    epitope_list = epitopes.split(", ")

    for n, d in graph.nodes(data=True):
        if n in epitope_list:
            epi_label_dict[n] = 1
        else:
            epi_label_dict[n] = 0
    pdb_epi_label_list.append(epi_label_dict)


with open("example_label_dict.pkl", "wb") as f:
    pickle.dump(pdb_epi_label_list, f)