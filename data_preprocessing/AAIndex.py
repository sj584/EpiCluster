import logging
from typing import Dict, Tuple

import networkx as nx

from graphein.utils.utils import import_message, protein_letters_3to1_all_caps

log = logging.getLogger(__name__)


try:
    from pyaaisc import Aaindex
except ImportError:
    message = import_message(
        submodule="graphein.protein.features.nodes.aaindex",
        package="pyaaisc",
        pip_install=True,
    )
    log.warning(message)


# for error processing    
class Record:
    def __init__(self, title, index_data):
        self.title = title
        self.index_data = index_data

accession_dict = {"FAUJ880101": "Amino acid side chain parameters for correlation studies in biology and pharmacology",
                  "CHAM820101": "The structural dependence of amino acid hydrophobicity parameters",
                  "FAUJ880103": "Amino acid side chain parameters for correlation studies in biology and pharmacology",
                  "ZIMJ680101": "The characterization of amino acid sequences in proteins by statistical methods",
                  "ZIMJ680104": "The characterization of amino acid sequences in proteins by statistical methods",
                  "CHOP780201": "Prediction of the secondary structure of proteins from their amino acid sequence",
                  "CHOP780202": "Prediction of the secondary structure of proteins from their amino acid sequence",
                  "CHOP780203": "Prediction of the secondary structure of proteins from their amino acid sequence",
                  "ZIMJ680103": "The characterization of amino acid sequences in proteins by statistical methods"}

accession_index_data_dict = {"FAUJ880101": {'A': 1.28, 'L': 2.59, 'R': 2.34, 'K': 1.89, 'N': 1.6, 'M': 2.35, 'D': 1.6, 'F': 2.94, 'C': 1.77, 'P': 2.67, 'Q': 1.56, 'S': 1.31, 'E': 1.56, 'T': 3.03, 'G': 0.0, 'W': 3.21, 'H': 2.99, 'Y': 2.94, 'I': 4.19, 'V': 3.67},
                             "CHAM820101": {'A': 0.046, 'L': 0.186, 'R': 0.291, 'K': 0.219, 'N': 0.134, 'M': 0.221, 'D': 0.105, 'F': 0.29, 'C': 0.128, 'P': 0.131, 'Q': 0.18, 'S': 0.062, 'E': 0.151, 'T': 0.108, 'G': 0.0, 'W': 0.409, 'H': 0.23, 'Y': 0.298, 'I': 0.186, 'V': 0.14},
                             "FAUJ880103": {'A': 1.0, 'L': 4.0, 'R': 6.13, 'K': 4.77, 'N': 2.95, 'M': 4.43, 'D': 2.78, 'F': 5.89, 'C': 2.43, 'P': 2.72, 'Q': 3.95, 'S': 1.6, 'E': 3.78, 'T': 2.6, 'G': 0.0, 'W': 8.08, 'H': 4.66, 'Y': 6.47, 'I': 4.0, 'V': 3.0},
                             "ZIMJ680101": {'A': 0.83, 'L': 2.52, 'R': 0.83, 'K': 1.6, 'N': 0.09, 'M': 1.4, 'D': 0.64, 'F': 2.75, 'C': 1.48, 'P': 2.7, 'Q': 0.0, 'S': 0.14, 'E': 0.65, 'T': 0.54, 'G': 0.1, 'W': 0.31, 'H': 1.1, 'Y': 2.97, 'I': 3.07, 'V': 1.79},
                             "ZIMJ680104": {'A': 6.0, 'L': 5.98, 'R': 10.76, 'K': 9.74, 'N': 5.41, 'M': 5.74, 'D': 2.77, 'F': 5.48, 'C': 5.05, 'P': 6.3, 'Q': 5.65, 'S': 5.68, 'E': 3.22, 'T': 5.66, 'G': 5.97, 'W': 5.89, 'H': 7.59, 'Y': 5.66, 'I': 6.02, 'V': 5.96},
                             "CHOP780201": {'A': 1.42, 'L': 1.21, 'R': 0.98, 'K': 1.16, 'N': 0.67, 'M': 1.45, 'D': 1.01, 'F': 1.13, 'C': 0.7, 'P': 0.57, 'Q': 1.11, 'S': 0.77, 'E': 1.51, 'T': 0.83, 'G': 0.57, 'W': 1.08, 'H': 1.0, 'Y': 0.69, 'I': 1.08, 'V': 1.06},
                             "CHOP780202": {'A': 0.83, 'L': 1.3, 'R': 0.93, 'K': 0.74, 'N': 0.89, 'M': 1.05, 'D': 0.54, 'F': 1.38, 'C': 1.19, 'P': 0.55, 'Q': 1.1, 'S': 0.75, 'E': 0.37, 'T': 1.19, 'G': 0.75, 'W': 1.37, 'H': 0.87, 'Y': 1.47, 'I': 1.6, 'V': 1.7},
                             "CHOP780203": {'A': 0.74, 'L': 0.5, 'R': 1.01, 'K': 1.19, 'N': 1.46, 'M': 0.6, 'D': 1.52, 'F': 0.66, 'C': 0.96, 'P': 1.56, 'Q': 0.96, 'S': 1.43, 'E': 0.95, 'T': 0.98, 'G': 1.56, 'W': 0.6, 'H': 0.95, 'Y': 1.14, 'I': 0.47, 'V': 0.59},
                             "ZIMJ680103": {'A': 0.0, 'L': 0.13, 'R': 52.0, 'K': 49.5, 'N': 3.38, 'M': 1.43, 'D': 49.7, 'F': 0.35, 'C': 1.48, 'P': 1.58, 'Q': 3.53, 'S': 1.67, 'E': 49.9, 'T': 1.66, 'G': 0.0, 'W': 2.1, 'H': 51.6, 'Y': 1.61, 'I': 0.13, 'V': 0.13}}


def fetch_AAIndex(accession: str) -> Tuple[str, Dict[str, float]]:
    """
    Fetches AAindex1 dictionary from an accession code. The dictionary maps one-letter AA codes to float values

    :param accession: Aaindex1 accession code
    :type accession: str
    :return: tuple of record titel(str) and dictionary of AA:value mappings
    :rtype: Tuple[str, Dict[str, float]]
    """
    # Initialise AAindex object and get data
    
    # Turns out, I don't need these. It only hampers the data processing with error or slow down
    #aaindex = Aaindex()
    #record = aaindex.get(accession)
    
    # This is faster
    record = Record(accession_dict[accession], accession_index_data_dict[accession])
        
    return record.title, record.index_data

def aaindex1(G: nx.Graph, accession: str) -> nx.Graph:
    """Adds AAIndex1 datavalues for a given accession as node features.

    :param G: nx.Graph protein structure graphein to featurise
    :type G: nx.Graph
    :param accession: AAIndex1 accession code for values to use
    :type accession: str
    :return: Protein Structure graph with AAindex1 node features added
    :rtype: nx.Graph
    """

    title, index_data = fetch_AAIndex(accession)
    #print("accession", accession)
    #print("title", title)
    #print("title", type(title))
    #print("index_data", index_data)
    #print("index_data", type(index_data))
    # TODO: change to allow for a list of all accession numbers?
    G.graph["aaindex1"] = accession + ": " + title

    if G.graph["config"].granularity == "atom":
        raise NameError(
            "AAIndex features cannot be added to atom granularity graph"
        )

    for n in G.nodes:
        residue = n.split(":")[1]
        residue = protein_letters_3to1_all_caps(residue)

        aaindex = index_data[residue]

        G.nodes[n][f"aaindex1_{accession}"] = aaindex

    return G