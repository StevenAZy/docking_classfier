##  Docking  Classfier

### Data processing
```
data_process
│
└───graph_protein
│   │   xxxx.pkl
│   │   xxxx.pkl
│   
└───graph_rna
│   │   xxxx.pkl
│   │   xxxx.pkl
│
└───pdb_protein
│   │   xxxx.pdb
│   │   xxxx.pdb
│
└───pdb_rna
    │   xxxx.pdb
    │   xxxx.pdb
```
`data_process/download_pdb.py` for downloading all pdb files about proteins and RNA.

`data_process/construct_graph.py` for constructing graphs about proteins and RNA.

`data.py` for loading train data.

`main.py` for start train/test.