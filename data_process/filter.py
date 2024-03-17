import os
import csv
import pickle

from tqdm import tqdm

path = '/home/steven/code/docking_classfier/data_process/graph_rna'
data_path = '/home/steven/code/docking_classfier/data_process/new_test_data.csv'


files = os.listdir(path)



with open(data_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)

    for _, row in enumerate(reader):
        f_p = os.path.join(path, f'{row[1]}.pkl')

        if not os.path.exists(f_p):
            print(row[1])
            continue

        with open(f_p, 'rb') as f:
            data = pickle.load(f)
            
            if isinstance(data, dict):
                print('d' * 100)
                print(row[1])
                continue
            
            # print(data.node_feat.size()[1])
            if data.node_feat.size()[1] != 120:
                print('9' * 100)
                print(row[1])

# cnt = 0
# err = 0

# for file in tqdm(files):
#     f_p = os.path.join(path, file)

#     with open(f_p, 'rb') as f:
#         data = pickle.load(f)
#         if isinstance(data, dict):
#             print(file)
    # try:
    #     if data.node_feat.size()[1] == 120:
    #         cnt = cnt + 1
    # except:
    #     print(file)
    #     err = err + 1

# train_data = []

# with open(data_path, 'r') as f:
#     reader = csv.reader(f)
#     next(reader)

#     for _, row in enumerate(reader):
#         if f'{row[1]}.pkl' in files:
#             train_data.append([row[0],row[1],row[2]])


# with open('val_data.csv', 'w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerows(train_data)