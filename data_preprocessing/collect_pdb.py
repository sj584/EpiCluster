import pandas as pd

df = pd.read_csv("example_dataset.csv")
df.head()

import os
import wget
import shutil

pdb_dir = "example_pdb"

if os.path.isdir(pdb_dir):
    pass
else:
    os.mkdir(pdb_dir)
        
for pdb in df["PDB"]:
    wget.download(f"https://files.rcsb.org/download/{pdb}.pdb")
    shutil.move(f"{pdb.lower()}.pdb", pdb_dir)