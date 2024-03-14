import os
from Bio import SeqIO


def pdb2fa(pdb_file, save_path):
    pdb_id = os.path.basename(pdb_file).split('.')[0]
    save_path = os.path.join(save_path, f'{pdb_id}.fa')

    with open(pdb_file) as handle:
        sequence = next(SeqIO.parse(handle, "pdb-atom"))
        sequence.id = sequence.id.replace('????', pdb_id.upper())
        sequence.description = sequence.description.replace('????', pdb_id.upper())

    # 将PDB文件写入FA文件
    with open(save_path, "w") as output_handle:
        SeqIO.write(sequence, output_handle, "fasta")


from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
from Bio import SeqIO

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import RNAAlphabet
from Bio import SeqIO

def pdb_to_fasta(pdb_file, output_file):
    pdbid = os.path.basename(pdb_file).split('.')[0]
    # 初始化一个空的序列字符串
    sequence = ''

    # 打开PDB文件并逐行处理
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                # 提取碱基信息
                residue = line[17:20].strip()
                if residue in ['A', 'U', 'G', 'C']:
                    sequence += residue

    # 创建一个SeqRecord对象
    record = SeqRecord(Seq(sequence, RNAAlphabet()), id=pdbid, description='')

    # 将SeqRecord对象写入FASTA文件
    SeqIO.write(record, output_file, 'fasta')

# 指定输入的PDB文件和输出的FASTA文件
# input_pdb_file = 'input.pdb'
# output_fasta_file = 'output.fa'

# 调用函数进行转换
# pdb_to_fasta(input_pdb_file, output_fasta_file)




if __name__ == '__main__':
    pdb_path = '/home/steven/code/docking_classfier/data_process/pdb_rna'
    save_path = '/home/steven/code/docking_classfier/data_process/fasta_rna'

    pdbs = os.listdir(pdb_path)

    for pdb in pdbs:
        pdbfile = os.path.join(pdb_path, pdb)
        fa = pdb.replace('pdb', 'fa')
        output = os.path.join(save_path, fa)

        try:
            pdb_to_fasta(pdbfile, output)
        except:
            print('=' * 100)
            print(pdb)

        # try:
        #     pdb2fa(pdbfile, save_path)
        # except BaseException as error:
        #     print(error)
            # print(pdb)
