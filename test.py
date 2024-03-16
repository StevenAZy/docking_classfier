from cProfile import label
import torch
import numpy as np

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import P_RPairDataset, collate
from model import MyModel


sw = SummaryWriter('log')

train_dataset = P_RPairDataset(data_path='./data_process/full_train_data.csv')
train_dataloader  = DataLoader(dataset=train_dataset, batch_size=64, collate_fn=collate)

val_dataset = P_RPairDataset(data_path='./data_process/full_val_data.csv')
val_dataloader  = DataLoader(dataset=val_dataset, batch_size=64, collate_fn=collate)


model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(500):
    epoch_loss = []
    pbar = tqdm(train_dataloader, desc=f'epoch-{epoch}')
    for train_data in pbar:
    # for data in tqdm(dataloader, desc=f'epoch-{epoch}'):
        train_labels = torch.tensor([int(label) for label in train_data[2]], dtype=torch.long).cuda()
        out = model(train_data)
        loss = criterion(out, train_labels)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()
        pbar.set_description(f'epoch-{epoch},loss:{loss.detach().cpu().numpy()}', refresh=True)
        epoch_loss.append(loss.detach().cpu().numpy())
    
    sw.add_scalar('train_loss', np.mean(epoch_loss), epoch)

    model.eval()
    correct = 0
    for val_data in val_dataloader:
        preds = model(val_data)
        preds = preds.argmax(dim=1)
        val_labels = torch.tensor([int(label) for label in val_data[2]], dtype=torch.long).cuda()
        correct += int((preds == val_labels).sum())

    sw.add_scalar('val_acc', correct/len(val_dataloader), epoch)
        