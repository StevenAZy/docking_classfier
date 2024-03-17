import os


path = '/home/steven/code/docking_classfier/data_process/fasta_rna'
save_file = 'db_rna.fa'

files = os.listdir(path)

for file in files:
    file_path = os.path.join(path, file)

    with open(save_file, 'a+') as sf:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            seq = lines[0]
            for line in lines[1:]:
                seq = seq + line.strip()
        if len(seq) < 7:
            continue
        sf.write(seq + '\n')

