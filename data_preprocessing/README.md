This example is data preprocessing from Epitope3D external test set "epitope3d_dataset_45_Blind_Test.csv"
=============================

Version
=============================
PSAIA 1.0
NCBI-BLAST 2.13

Tutorials 
=============================
Prerequisits: csv file of PDB ID in Data_processing directory 

.csv file comes with column names: PDB, Epitope

Epitopes should be {Chain:Amino_acid(3):residue_number}", "{Chain:Amino_acid(3):residue_number}

1. collect_pdb.py --> example_pdb/*.pdb

2. collect_fasta.py --> example_fasta/*.fasta

3. Make PSAIA data and PSSM data by... 

PSAIA: Structure Analyser: Accessible Surface Area, Depth Index, Protrusion Index, Hydrophobicity 
Analyse as Bound, Table Output

PSI-BLAST. psi_blast -query example.fasta -db swissprot -num_iterations 3 -out_ascii_pssm example.pssm 

After this process, you'll get example_psaia/*.tbl, example_pssm/*.pssm

4. generate_graphs.py --> example_graphs_5A.pkl

5. remove_symmetry.py --> example_nonsym_graphs_5A.pkl

6. PSAIA_PSSM_2_pkl.py  --> example_psaia.pkl, example_pssm.pkl

7. generate_label.py  (not necessary when you don't have epitope labels) --> example_label_dict.pkl

8. generate_dataset.py --> example_pyg_surface_"RSA".pkl


The files from 4-8 you can download them in the google drive link below:
https://drive.google.com/drive/folders/1erW8dht3YB6dAH6Z1YkbXHefgxfR_qjd?usp=share_link
