import torch
import numpy as np

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import *
from model import MyModel
from metric import calculate_metrics
from data import P_RPairDataset, collate


sw = SummaryWriter('log')

def train():
    train_dataset = P_RPairDataset(data_path=TRAIN_DATA_PATH)
    train_dataloader  = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    val_dataset = P_RPairDataset(data_path=VAL_DATA_PATH)
    val_dataloader  = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    model = MyModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        epoch_loss = []
        train_pbar = tqdm(train_dataloader, desc=f'epoch-{epoch}')
        for train_data in train_pbar:
        # for train_data in tqdm(train_dataloader, desc=f'epoch-{epoch}'):
            train_labels = torch.tensor([int(label) for label in train_data[2]], dtype=torch.long).cuda()
            out = model(train_data)
            loss = criterion(out, train_labels)
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()
            train_pbar.set_description(f'epoch-{epoch},loss:{loss.detach().cpu().numpy()}', refresh=True)
            epoch_loss.append(loss.detach().cpu().numpy())
        
        sw.add_scalar('train_loss', np.mean(epoch_loss), epoch)

        model.eval()

        all_preds = []
        all_labels = []

        val_pbar = tqdm(val_dataloader, desc=f'val_acc:')
        for val_data in val_pbar:
            preds = model(val_data)
            preds = preds.argmax(dim=1)
            all_preds.extend(preds)

            val_labels = torch.tensor([int(label) for label in val_data[2]], dtype=torch.long).cuda()
            all_labels.extend(val_labels)

            batch_acc = int((preds == val_labels).sum()) / BATCH_SIZE
            val_pbar.set_description(f'val_acc:{batch_acc}', refresh=True)

        all_labels = [label.cpu().detach().numpy() for label in all_labels]
        all_preds = [pred.cpu().detach().numpy() for pred in all_preds]
        accuracy, precision, recall, f1, cm = calculate_metrics(all_labels, all_preds)
        sw.add_scalar('val_acc', accuracy, epoch)
        sw.add_scalar('val_pre', precision, epoch)
        sw.add_scalar('val_rec', recall, epoch)
        sw.add_scalar('val_f1', f1, epoch)
        sw.add_image('val_cm', torch.from_numpy(cm), dataformats='HW')

        if (epoch + 1) % SAVE_EPOCH == 0:
            torch.save(model.state_dict(), f'{MODEL_SAVE_PATH}/checkpoint_{epoch+1}.pth')


def test(save_epoch = 100):
    model = MyModel().cuda()
    model.load_state_dict(torch.load(f'{MODEL_SAVE_PATH}/checkpoint_{save_epoch}.pth'))
    model.eval()

    test_dataset = P_RPairDataset(data_path=TEST_DATA_PATH)
    test_dataloader  = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    test_pbar = tqdm(test_dataloader, desc=f'test_acc:')

    all_preds = []
    all_labels = []

    for test_data in test_pbar:
        preds = model(test_data)
        preds = preds.argmax(dim=1)
        all_preds.extend(preds)

        test_labels = torch.tensor([int(label) for label in test_data[2]], dtype=torch.long).cuda()
        all_labels.extend(test_labels)

        batch_acc = int((preds == test_labels).sum()) / BATCH_SIZE
        test_pbar.set_description(f'test_acc:{batch_acc}', refresh=True)

    all_labels = [label.cpu().detach().numpy() for label in all_labels]
    all_preds = [pred.cpu().detach().numpy() for pred in all_preds]
    accuracy, precision, recall, f1, cm = calculate_metrics(all_labels, all_preds)
    sw.add_scalar('test_acc', accuracy)
    sw.add_scalar('test_pre', precision)
    sw.add_scalar('test_rec', recall)
    sw.add_scalar('test_f1', f1)
    sw.add_image('test_cm', torch.from_numpy(cm), dataformats='HW')



if __name__ == '__main__':
    train()
    # test()
