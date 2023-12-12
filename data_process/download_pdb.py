'''
download pdb file for all protein and rna
'''
import wget
import pandas as pd


DOWNLOAD_URL = 'https://files.rcsb.org/download'

data_path = 'data/data.csv'
protein_pdb_save_path = 'data/pdb_protein'
rna_pdb_save_path = 'data/pdb_rna'

data = pd.read_csv(data_path)
protein_ids = set(data['protein_id'].values.tolist())
rna_ids = set(data['rna_id'].values.tolist())

for id in protein_ids:
    wget_url = f'{DOWNLOAD_URL}/{id}.pdb'
    try:
        wget.download(wget_url, out=f'{protein_pdb_save_path}/{id}.pdb')
    except:
        print(id)
        continue

for id in rna_ids:
    wget_url = f'{DOWNLOAD_URL}/{id}.pdb'
    try:
        wget.download(wget_url, out=f'{rna_pdb_save_path}/{id}.pdb')
    except:
        print(id)
        continue


