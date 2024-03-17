import os
import re
import time
import requests
import MDAnalysis as mda

from Bio import SeqIO
from tqdm import tqdm
from bs4 import BeautifulSoup
from MDAnalysis.analysis import rms

def cal_rmsd(pdb_1, pdb_2):
    u1 = mda.Universe(pdb_1)
    u2 = mda.Universe(pdb_2)

    rmsd_value = rms.rmsd(u1.select_atoms('protein'), u2.select_atoms('protein'))

    print(rmsd_value)


def RPISeq(protein, rna):
    form_data = {
        'p_input': protein,
        'r_input': rna,
        'submit': 'Submit'
    }

    response = requests.post('http://pridb.gdcb.iastate.edu/RPISeq/results.php', data=form_data)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')
    target_value = table.find_all('tr')[0].find_all('td')[1].text

    numbers = re.findall(r'\d+\.\d+', target_value)

    # return {'RF': numbers[0], 'SVM': numbers[1]}
    return numbers[0], numbers[1]


# if __name__ == '__main__':
#     protein_path = '/home/steven/code/docking_classfier/data_process/fasta_protein'
#     rna_path = '/home/steven/code/docking_classfier/data_process/fasta_rna'

#     ids = os.listdir(protein_path)

#     for id in tqdm(ids):
#         protein = os.path.join(protein_path, id)
#         rna = os.path.join(rna_path, id) 
#         try:
#             with open(protein, 'r') as handle:
#                 for record in SeqIO.parse(handle, 'fasta'):
#                     p_input = str(record.seq)

#             with open(rna, 'r') as handle:
#                 for record in SeqIO.parse(handle, 'fasta'):
#                     r_input = str(record.seq)
        
#             RF_p, SVM_p = RPISeq(p_input, r_input)
#         except:
#             with open('rpiseq_err.txt', 'a+') as f:
#                 f.write(f'{id}\n')
            
#             continue

#         with open('rpiseq.txt', 'a+') as f:
#             line_txt = f'{id}:RF-{RF_p}, SVM-{SVM_p}\n'
#             f.write(line_txt)
#         time.sleep(0.5)
        

if __name__ == '__main__':
    path = '/home/steven/code/docking_classfier/rpiseq.txt'

    with open(path, 'r') as f:
        lines = f.readlines()

    pred_cnt = 0

    for line in lines:
        # v = float(line.split('-')[-1])
        v = float(line.split(',')[0].split(':')[-1].split('-')[-1])
        if v > 0.5:
            pred_cnt = pred_cnt + 1

    acc = pred_cnt / 1428

    print(acc)
