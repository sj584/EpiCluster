import pandas as pd

df = pd.read_csv("example_dataset.csv")
df.head()

import pickle


with open("example_label_dict.pkl", "rb") as f:
     y_example = pickle.load(f)

with open("example_pssm.pkl", "rb") as f:
     example_pssm = pickle.load(f)

with open("example_nonsym_graphs_5A.pkl", "rb") as f:
    example_graphs = pickle.load(f)

with open("example_psaia.pkl", "rb") as f:
     example_psaia = pickle.load(f)

import torch.nn.functional as F

def normalize(d, max_val, min_val):
    d = float(d)
    norm = (d - min_val) / (max_val - min_val)
    return norm

def distance_bond(d):
    num = 0
    if "distance_threshold" in list(d['kind']):
        num = 1
    return num

def one_hot(ss):
    dssp_ss = {"H":0, "B":1, "E":2, "G":3, "I":4, "T":5, "S":6, "-":7}
    dssp_num = dssp_ss[ss]

    one_hot_mat = np.array(F.one_hot(torch.arange(8), num_classes=8))
    one_hot_enc = one_hot_mat[dssp_num]
    return one_hot_enc

from graphein.protein.config import ProteinGraphConfig

config = ProteinGraphConfig()


## after 

from graphein.protein.subgraphs import extract_subgraph_from_point
from Bio.PDB.DSSP import dssp_dict_from_pdb_file, residue_max_acc
from graphein.ml.conversion import GraphFormatConvertor
from graphein.protein.utils import download_pdb

import torch
import numpy as np
import pandas as pd
import random
import numpy as np


aa_dict = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
         'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

def format_convert_surface(graphs_list, y, df, pssm, psaia, RSA):
    format_convertor = GraphFormatConvertor('nx', 'pyg',
                                            verbose = 'gnn',
                                            columns = None)
    # NBVAL_SKIP
    pyg_list = [format_convertor(graph) for graph in graphs_list]
    
    ## building amino acid composition features
    n_comp_list = []
    n_sg_len_list = []
    
    ## building dssp features (rsa, ss)
    n_asa_list = []
    n_ss_list = []
    n_phi_list = []
    n_psi_list = []
    for idx, g in enumerate(pyg_list): 
        aaindex1(graphs_list[idx], 'FAUJ880101') # Steric parameter (graph shape index)
        aaindex1(graphs_list[idx], 'CHAM820101') # Polarizability   
        aaindex1(graphs_list[idx], 'FAUJ880103') # Volume (normalized van der Waals volumn) 
        aaindex1(graphs_list[idx], 'ZIMJ680101') # Hydrophobicity
        aaindex1(graphs_list[idx], 'ZIMJ680104') # Isoelectric point
        aaindex1(graphs_list[idx], 'CHOP780201') # Helix probability
        aaindex1(graphs_list[idx], 'CHOP780202') # Sheet probability
        aaindex1(graphs_list[idx], 'CHOP780203') # Turn probability
        aaindex1(graphs_list[idx], 'ZIMJ680103') # Polarity
        
        esm_residue_embedding(graphs_list[idx]) # input of node embedding esm (1280 dim)
        
        ## building amino acid composition features
        
        aa_count_list = []
        sg_len_list = []
        for n, d in graphs_list[idx].nodes(data=True):
            radius = 8
            sg = extract_subgraph_from_point(graphs_list[idx], centre_point=tuple(d['coords']), radius=radius)
            sg_len_list.append(len(sg.nodes))
            
            aa_count = {'C': 0, 'D': 0, 'S': 0, 'Q': 0, 'K': 0,
                        'I': 0, 'P': 0, 'T': 0, 'F': 0, 'N': 0, 
                        'G': 0, 'H': 0, 'L': 0, 'R': 0, 'W': 0, 
                        'A': 0, 'V': 0, 'E': 0, 'Y': 0, 'M': 0}
            
            for n, d in sg.nodes(data=True):
                aa = aa_dict[d["residue_name"]]
                aa_count[aa] += 1
            aa_count_list.append(aa_count)
        n_comp_list.append(aa_count_list)
        n_sg_len_list.append(sg_len_list)        

        
        ## building dssp features (rsa, ss)
        
        pdb = download_pdb(config=config, pdb_code=graphs_list[idx].graph['pdb_id']) 
        dssp = dssp_dict_from_pdb_file(pdb)

        asa_list = []
        ss_list = []
        phi_list = []
        psi_list = []
        
        
        for n, d in graphs_list[idx].nodes(data=True):
            
            try:
                key = (d["chain_id"], (' ', d['residue_number'], ' '))
                ss_list.append(dssp[0][key][1])
                asa_list.append(dssp[0][key][2])
                phi_list.append(dssp[0][key][3])
                psi_list.append(dssp[0][key][4])
            except:
                ss_list.append("-")
                asa_list.append(0)
                phi_list.append(0)
                psi_list.append(0)
                print("Key Error... appending ss: -, asa: 0, phi: 0, psi: 0")
        n_asa_list.append(asa_list)
        n_ss_list.append(ss_list)
        n_phi_list.append(phi_list)
        n_psi_list.append(psi_list)

        
        # features   1. amino_acid type (one-hot)  (dim=20)
        #            2. Steric parameter (graph shape index) (max-min normalize)
        #            3. Polarizability     (max-min normalize)              
        #            4. Volume (normalized van der Waals volumn) (standardization) 
        #            5. Hydrophobicity (max-min normalize) 
        #            6. Isoelectric point (standardization)   
        #            7. Helix probability
        #            8. Sheet probability
        #            9. Turn probability
        #            10. Polarity (standardization) 
        #            11. B-factor
        #            12. RSA
        #            11. phi angle  
        #            12. psi angle  
        #            13. protrusion index  # thinking of adding. not yet!
        #            14. residue depth    # thinking of adding. not yet!
        #            15. pssm (max-min normalize) (dim=20) 
        #            16. a.a. composition (dim=20)
        #            17. secondary structure (ont-hot) (dim=8)

        ## re-scaling including esm embedding 

        g.node_attrs = torch.tensor(np.concatenate([np.array([d['amino_acid_one_hot'] for n, d in graphs_list[idx].nodes(data=True)]),
                                                      np.array([[normalize(d['aaindex1_FAUJ880101'], 4.19, 0), normalize(d['aaindex1_CHAM820101'], 0.409, 0), 
                                                      normalize(d['aaindex1_FAUJ880103'], 8.08, 0), normalize(d['aaindex1_ZIMJ680101'], 3.07, 0), 
                                                      normalize(d['aaindex1_ZIMJ680104'], 10.76 , 2.77), normalize(d['aaindex1_CHOP780201'], 1.51, 0.57),
                                                      normalize(d['aaindex1_CHOP780202'], 1.7, 0.37), normalize(d['aaindex1_CHOP780203'], 1.56, 0.47),
                                                      normalize(d['aaindex1_ZIMJ680103'], 52.0, 0), normalize(d['b_factor'], 459.9, 0),
                                                      normalize(n_asa_list[idx][index] / residue_max_acc["Sander"][d['residue_name']], 1.72, 0),
                                                      normalize(n_phi_list[idx][index], 180, -180), normalize(n_psi_list[idx][index], 180, -180),
                                                      normalize(psaia[idx][(psaia[idx]['chainid'] == d['chain_id']) & (psaia[idx]['resid'] == str(d['residue_number']))]["averageCX"], 10, 0),
                                                      normalize(psaia[idx][(psaia[idx]['chainid'] == d['chain_id']) & (psaia[idx]['resid'] == str(d['residue_number']))]["averageDPX"], 10, 0)] for index, (n, d) in enumerate(graphs_list[idx].nodes(data=True))]),
                                                      np.array(list([(np.array(pssm[idx].loc[index]).astype(np.float32) + 15) / 30 for index, (n, d) in enumerate(graphs_list[idx].nodes(data=True))])),
                                                      np.array(list([np.array(list(n_comp_list[idx][index].values())).astype(np.float32)  / n_sg_len_list[idx][index] for index, (n, d) in enumerate(graphs_list[idx].nodes(data=True))])),
                                                      np.array(list([one_hot(n_ss_list[idx][index]) for index, (n, d) in  enumerate(graphs_list[idx].nodes(data=True))])),
                                                      np.array([d["esm_embedding"] + 9.24 / 13.78 for n, d in graphs_list[idx].nodes(data=True)])], axis=1), dtype=torch.float32)

        
        g.y = torch.tensor([y[idx][n] for n, d in graphs_list[idx].nodes(data=True)])
        g.edge_attrs = torch.tensor([distance_bond(d) for u, v, d in graphs_list[idx].edges(data=True)], dtype=torch.float32)
        g.coords = torch.FloatTensor([d['coords'] for n, d in graphs_list[idx].nodes(data=True)])  
        
        ## added index that covers df
        epitope_list = df['Epitope'].values[idx].split(", ")
        
        non_epitope_list = []
        for index, (n, d) in enumerate(graphs_list[idx].nodes(data=True)):
            if (n_asa_list[idx][index] / residue_max_acc["Sander"][d['residue_name']] > RSA) & (n not in epitope_list):
                non_epitope_list.append(n)

        g.train_mask = torch.tensor([(n in epitope_list) | (n in non_epitope_list) for n, d in graphs_list[idx].nodes(data=True)])
        
    for i in pyg_list:
        if i.coords.shape[0] == len(i.node_id):
            pass
        else:
            print(i)
            pyg_list.remove(i)    
            
    return pyg_list


RSA = 0.15
example_pyg_list = format_convert_surface(example_graphs, y_example, df, example_pssm, example_psaia, RSA)

with open(f"./example_pyg_surface_{RSA}.pkl", "wb") as f:
    pickle.dump(example_pyg_list, f)