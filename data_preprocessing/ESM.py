from graphein.protein.features.sequence.utils import (
    compute_feature_over_chains,
    subset_by_node_feature_value,
)
import networkx as nx
import numpy as np
import torch

def esm_residue_embedding(
    G: nx.Graph,
    model_name: str = "esm2_t33_650M_UR50D",
    output_layer: int = 33,
) -> nx.Graph:                               # model_name: str = "esm1b_t33_650M_UR50S"
    

    
    ### important! even chain subgraph, G.graph shows all the chains -> for loop removal
    

    embedding_total = np.empty((0, 1280))
    for chain in G.graph["chain_ids"]:
        if len(G.graph[f"sequence_{chain}"]) > 1022:
            seq_len = len(G.graph[f"sequence_{chain}"])
            embedding = compute_esm_embedding(
                G.graph[f"sequence_{chain}"][:1022],
                representation="residue",
                model_name=model_name,
                output_layer=output_layer,
            )
        else:
            embedding = compute_esm_embedding(
            G.graph[f"sequence_{chain}"],
            representation="residue",
            model_name=model_name,
            output_layer=output_layer,
            )

        # remove start and end tokens from per-token residue embeddings
        embedding = embedding[0, 1:-1]
        #print("embedding type", type(embedding))
        print("embedding shape", embedding.shape)

        if len(G.graph[f"sequence_{chain}"]) > 1022:
            empty_len = len(G.graph[f"sequence_{chain}"]) - 1022
            embedding_ = np.concatenate((embedding, np.zeros((empty_len, 1280))), axis=0)
            embedding_total = np.concatenate((embedding_total, embedding_), axis=0)
            print("embedding_total.shape", embedding_total.shape)
        else:
            embedding = embedding.reshape(-1, 1280) ################
            embedding_total = np.concatenate((embedding_total, embedding), axis=0)
            print("embedding_total.shape", embedding_total.shape)
                


            ### for chain length longer than 1022, C terminal was trimmed. embedding 

    for emb, (n, d) in zip(embedding_total, G.nodes(data=True)):
        G.nodes[n]["esm_embedding"] = emb

        
    return G


import numpy as np

def compute_esm_embedding(
    sequence: str,
    representation: str,
    model_name: str = "esm2_t33_650M_UR50D",
    output_layer: int = 33,
) -> np.ndarray:
    """
    Computes sequence embedding using Pre-trained ESM model from FAIR

        Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences (2019)
        Rives, Alexander and Meier, Joshua and Sercu, Tom and Goyal, Siddharth and Lin, Zeming and Liu, Jason and Guo,
        Demi and Ott, Myle and Zitnick, C. Lawrence and Ma, Jerry and Fergus, Rob


        Transformer protein language models are unsupervised structure learners 2020
        Rao, Roshan M and Meier, Joshua and Sercu, Tom and Ovchinnikov, Sergey and Rives, Alexander

    Pre-trained models:

    Full Name layers params Dataset Embedding Dim Model URL
    ========= ====== ====== ======= ============= =========
    ESM-1b esm1b_t33_650M_UR50S 33 650M UR50/S 1280 https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt
    ESM1-main esm1_t34_670M_UR50S 34 670M UR50/S 1280 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR50S.pt
    esm1_t34_670M_UR50D 34 670M UR50/D 1280 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR50D.pt
    esm1_t34_670M_UR100 34 670M UR100 1280 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR100.pt
    esm1_t12_85M_UR50S 12 85M UR50/S 768 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t12_85M_UR50S.pt
    esm1_t6_43M_UR50S 6 43M UR50/S 768 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t6_43M_UR50S.pt

    :param sequence: Protein sequence to embed (str)
    :type sequence: str
    :param representation: Type of embedding to extract. ``"residue"`` or ``"sequence"``. Sequence-level embeddings are averaged residue embeddings
    :type representation: str
    :param model_name: Name of pre-trained model to use
    :type model_name: str
    :param output_layer: integer indicating which layer the output should be taken from
    :type output_layer: int
    :return: embedding (``np.ndarray``)
    :rtype: np.ndarray
    """
    model, alphabet = _load_esm_model(model_name)
    batch_converter = alphabet.get_batch_converter()

    data = [
        ("protein1", sequence),
    ]
    
    if len(sequence) <= 1022:
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(
                batch_tokens, repr_layers=[output_layer], return_contacts=True
            )
        token_representations = results["representations"][output_layer]
    
    ### important C-terminal ones longer than 1022
    else:
        data = [        
            ("protein1", sequence[:1022]),
        ]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(
                batch_tokens, repr_layers=[output_layer], return_contacts=True
            )
        token_representations = results["representations"][output_layer]

        
    if representation == "residue":
        return token_representations.numpy()
        
    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    elif representation == "sequence":
        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(
                token_representations[i, 1 : len(seq) + 1].mean(0)
            )
        return sequence_representations[0].numpy()


def _load_esm_model(model_name: str = "esm2_t33_650M_UR50D"):   # model_name: str = "esm1b_t33_650M_UR50S"
    """
    Loads pre-trained FAIR ESM model from torch hub.

        Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences (2019)
        Rives, Alexander and Meier, Joshua and Sercu, Tom and Goyal, Siddharth and Lin, Zeming and Liu, Jason and Guo,
        Demi and Ott, Myle and Zitnick, C. Lawrence and Ma, Jerry and Fergus, Rob


        Transformer protein language models are unsupervised structure learners 2020
        Rao, Roshan M and Meier, Joshua and Sercu, Tom and Ovchinnikov, Sergey and Rives, Alexander

    Pre-trained models:
    Full Name layers params Dataset Embedding Dim Model URL
    ========= ====== ====== ======= ============= =========
    ESM-1b   esm1b_t33_650M_UR50S 33 650M UR50/S 1280 https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt
    ESM1-main esm1_t34_670M_UR50S34 670M UR50/S 1280 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR50S.pt
    esm1_t34_670M_UR50D 34 670M UR50/D 1280 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR50D.pt
    esm1_t34_670M_UR100 34 670M UR100 1280 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR100.pt
    esm1_t12_85M_UR50S 12 85M UR50/S 768 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t12_85M_UR50S.pt
    esm1_t6_43M_UR50S 6 43M UR50/S 768 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t6_43M_UR50S.pt

    :param model_name: Name of pre-trained model to load
    :type model_name: str
    :return: loaded pre-trained model
    """

    return torch.hub.load("facebookresearch/esm", model_name)
