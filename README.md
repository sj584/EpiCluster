# EpiCluster (Preprint, Under Revision)
EpiCluster: end-to-end deep learning model for B cell epitope prediction designed to capture epitope clustering property
https://www.researchsquare.com/article/rs-2709196/v1

EpiCluster is State-of-the-art model in conformational B cell epitope.<br />
EpiCluster leverages structural features, evolutionary features, <br /> and clustering feature
of conformational B cell epitopes.


Dependencies
=======================
python 3.8.12<br />
pytorch 1.10.1<br />
torch-geometric 2.0.4<br />
biopython 1.79<br />
graphein 1.4.0<br />
numpy 1.21.2<br />
pandas 1.3.4<br />
scikit-learn 1.0.1<br />
scipy 1.7.1<br />


EpiCluster dataset 
=======================
1. Initial training set and independent set<br />
"epitope3D: a machine learning method for conformational B-cell epitope prediction"
http://biosig.unimelb.edu.au/epitope3d/data

2. Benchmark set 1<br />
"Critical review of conformational B-cell epitope prediction methods"
https://github.com/3BioCompBio/BCellEpitope

3. Benchmark set 2 (supplementary)<br />
"Positive-unlabeled learning for the prediction of conformational B-cell epitopes"
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-16-S18-S12#additional-information

Data availability
======================= 
preprocessed epitope3D train/test/blind set files are provided from the link below.
https://drive.google.com/drive/folders/1erW8dht3YB6dAH6Z1YkbXHefgxfR_qjd?usp=share_link

blind set is provided under the name: "example_pyg_surface_0.15.pkl" 
and you can try out the steps of preprocessing the data in "data_preprocessing" folder using blind set.


Acknowledgements
=======================
code heavily depends on the following repositories
1. E(n) Equivariant Neural Networks https://github.com/vgsatorras/egnn
2. Graphein - a Python Library for Geometric Deep Learning and Network Analysis on Protein Structures
   https://github.com/a-r-j/graphein
