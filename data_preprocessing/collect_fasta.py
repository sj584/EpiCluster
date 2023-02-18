import pandas as pd

df = pd.read_csv("example_dataset.csv")
df.head()

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph

config = ProteinGraphConfig()

pdb_dir = "example_pdb"
fasta_dir = "example_fasta"

import os
if os.path.isdir(fasta_dir):
    pass
else:
    os.mkdir(fasta_dir)

for pdb in df["PDB"]:
    print("pdb :", pdb)
    pdb_path = f"./{pdb_dir}/{pdb.lower()}.pdb"
    g = construct_graph(config=config, pdb_path=pdb_path)
    for chain in g.graph["chain_ids"]:
        print("chain :", chain)
        print(g.graph[f"sequence_{chain}"])
        print("="*50)
        with open(f"{fasta_dir}/{pdb.lower()}_{chain}.fasta", "w") as f:
            f.write(f">{pdb}_{chain}\n")
            f.write(g.graph[f"sequence_{chain}"])
            f.write("\n")

