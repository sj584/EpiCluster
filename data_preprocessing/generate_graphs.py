import pandas as pd

df = pd.read_csv("example_dataset.csv")
df.head()

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import add_distance_threshold
from graphein.protein.visualisation import plotly_protein_structure_graph
from functools import partial
from graphein.protein.features.nodes.aaindex import aaindex1
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot
from graphein.protein.features.nodes.amino_acid import expasy_protein_scale

# edge construction within 5 Angstrome
edge_funcs = {"edge_construction_functions": [partial(add_distance_threshold, long_interaction_threshold=1, threshold=5)]}

node_funcs = {"node_metadata_functions": [expasy_protein_scale, 
                                          amino_acid_one_hot]}

config = ProteinGraphConfig(**edge_funcs, **node_funcs) #**node_funcs

graphs_list = []
pdb_dir = "example_pdb"

for pdb in df["PDB"]:
    print("pdb :", pdb)
    g = construct_graph(config=config, pdb_path=f"{pdb_dir}/{pdb.lower()}.pdb")
    graphs_list.append(g)

import pickle
with open("./example_graphs_5A.pkl", "wb") as f:
    pickle.dump(graphs_list, f)