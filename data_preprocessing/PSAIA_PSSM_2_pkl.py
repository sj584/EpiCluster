import pandas as pd

df = pd.read_csv("example_dataset.csv")
df.head()

import pandas as pd

def PSAIA(pdb_list):
    ## append list
    serial = "202206291953" ## you need to change this serial according to your file name
    pdb_df_list = []
    psaia_pdb_list = []
    for pdb in pdb_list:
        with open(f"./example_psaia/{pdb}_{serial}_bound.tbl") as f:
            data = f.readlines()[9:]
        
        # make columns
        column_list = []
        for idx, i in enumerate(data):
            if idx == 0:
                column_list = i.rstrip("\n").replace(" ", "").split("|")
        column_list = column_list[:-1]
        
        # make contents
        data_list = []
        for i in data:
            data_list.append(i.rstrip("\n").split())
        data_list
        
        df = pd.DataFrame(data_list[1:], columns=column_list)
        pdb_df_list.append(df)
    
    return pdb_df_list

example_psaia_list = PSAIA(df["PDB"])

import pickle
with open("example_psaia.pkl", "wb") as f:
    pickle.dump(example_psaia_list, f)


from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph

config = ProteinGraphConfig()

with open("example_chain_dict.pkl", "rb") as f:
    example_chain_dict = pickle.load(f)

with open("example_nonsym_graphs_5A.pkl", "rb") as f:
    example_nonsym_graphs = pickle.load(f)

import pandas as pd

def pssm(chains_list, chain_dict):
    ## append list
    pssm_pdb_list = []
    for idx, g in enumerate(chains_list):
        pdb = g.graph["pdb_id"]
        pdb_lower = pdb.lower()
        
        ### for homo chain with one chain only
        df_stack = pd.DataFrame()
        if len(list(chain_dict.values())[idx]) == 1:
            
            assert list(g.nodes)[0][0] == list(g.nodes)[-1][0]
            chain = list(g.nodes)[0][0]
            

            try:
                with open(f"example_pssm/{pdb_lower}_{chain}.pssm") as f:
                    data = f.readlines()[2:-6]  ## remove title, K , Lambda scores

                data_list = []
                for i in data:
                    data_list.append(i.rstrip("\n"))

                matrix = []
                for i in data_list:
                    matrix.append(i[11:][:79].split()[:20]) ## take only pssm matrix. remove index

                df = pd.DataFrame(matrix[1:])
                df_stack = pd.concat([df_stack, df], ignore_index=True)

            except:
                ### when sequence is so short so that the pssm is not constructed in PSI-BLAST
                ### only put 0 in the matrix
                seq_length = len(g.graph[f"sequence_{chain}"])
                matrix = []
                matrix.append(["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"])
                for i in range(seq_length):
                    matrix.append([0]*20)
                df = pd.DataFrame(matrix[1:])
                df_stack = pd.concat([df_stack, df], ignore_index=True)
            df_stack.columns = matrix[0]
            pssm_pdb_list.append(df_stack)
        
        ### for hetero-chain/homo-chain more than 1 chain
        else:

            chains = g.graph["chain_ids"]
            print("chains :", chains)

            for chain in chains:
                try:
                    if chain.islower():
                        with open(f"example_pssm/{pdb_lower}_{chain}s.pssm") as f:
                            data = f.readlines()[2:-6]  ## remove title, K , Lambda scores

                        data_list = []
                        for i in data:
                            data_list.append(i.rstrip("\n"))

                        matrix = []
                        for i in data_list:
                            matrix.append(i[11:][:79].split()[:20]) ## take only pssm matrix. remove index

                        df = pd.DataFrame(matrix[1:])
                        df_stack = pd.concat([df_stack, df], ignore_index=True)
                    else:
                        with open(f"example_pssm/{pdb_lower}_{chain}.pssm") as f:
                            data = f.readlines()[2:-6]  ## remove title, K , Lambda scores

                        data_list = []
                        for i in data:
                            data_list.append(i.rstrip("\n"))

                        matrix = []
                        for i in data_list:
                            matrix.append(i[11:][:79].split()[:20]) ## take only pssm matrix. remove index

                        df = pd.DataFrame(matrix[1:])
                        df_stack = pd.concat([df_stack, df], ignore_index=True)
                except:
                    ### when sequence is so short so that the pssm is not constructed in PSI-BLAST
                    ### only put 0 in the matrix
                    seq_length = len(g.graph[f"sequence_{chain}"])
                    matrix = []
                    matrix.append(["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"])
                    for i in range(seq_length):
                        matrix.append([0]*20)
                    df = pd.DataFrame(matrix[1:])
                    df_stack = pd.concat([df_stack, df], ignore_index=True)
                    print("chains :", matrix[0])
            df_stack.columns = matrix[0]
            pssm_pdb_list.append(df_stack)
                
    return pssm_pdb_list

example_pssm = pssm(example_nonsym_graphs, example_chain_dict)