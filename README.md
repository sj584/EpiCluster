# EpiCluster


EpiCluster is State-of-the-art model in conformational B cell epitope.
EpiCluster leverages structural features, evolutionary features, and clustering feature
of conformational B cell epitopes


Dependencies
=======================
python 3.8.12
pytorch 1.10.1
torch-geometric 2.0.4
biopython 1.79
graphein 1.4.0
numpy 1.21.2
pandas 1.3.4
scikit-learn 1.0.1
scipy 1.7.1


EpiCluster dataset 
=======================
1. Initial training set and independent set
"epitope3D: a machine learning method for conformational B-cell epitope prediction"
http://biosig.unimelb.edu.au/epitope3d/data

2. Benchmark set 1
"Critical review of conformational B-cell epitope prediction methods"
https://github.com/3BioCompBio/BCellEpitope

3. Benchmark set 2 (supplementary)
"Positive-unlabeled learning for the prediction of conformational B-cell epitopes"
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-16-S18-S12#additional-information


Acknowledgements
=======================
code heavily depends on the following repositories
1. E(n) Equivariant Neural Networks https://github.com/vgsatorras/egnn
2. Graphein - a Python Library for Geometric Deep Learning and Network Analysis on Protein Structures
   https://github.com/a-r-j/graphein
