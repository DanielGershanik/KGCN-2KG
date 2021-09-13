import pandas as pd
import numpy as np
import argparse
import random
from model import KGCN
from data_loader import DataLoader
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from metrics import Metrics

# prepare arguments (hyperparameters)
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='book_v2',
                    help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum',
                    help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=40,
                    help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int,
                    default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=16,
                    help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1,
                    help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-6,
                    help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--ratio', type=float, default=0.8,
                    help='size of training dataset')

args = parser.parse_args(['--l2_weight', '1e-6'])

# build dataset and knowledge graph
data_loader = DataLoader(args.dataset)

kg = data_loader.load_kg(mode='item')
user_kg = data_loader.load_kg(mode='user')
# user_kg = kg # SINGLE ONLY

df_dataset_item = data_loader.load_dataset(mode='item')
df_dataset_user = data_loader.load_dataset(mode='user')
# df_dataset_user = df_dataset_item  # SINGLE ONLY

# Dataset class


class KGCNDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_id = np.array(self.df.iloc[idx]['userID'])
        item_id = np.array(self.df.iloc[idx]['itemID'])
        label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
        return user_id, item_id, label


# train test split
x_train, x_test, y_train, y_test = train_test_split(df_dataset_item, df_dataset_item['label'], test_size=1 - args.ratio, shuffle=False, random_state=999)
train_dataset = KGCNDataset(x_train)
test_dataset = KGCNDataset(x_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

# User
x_train, x_test, y_train, y_test = train_test_split(df_dataset_user, df_dataset_user['label'], test_size=1 - args.ratio, shuffle=False, random_state=999)
train_dataset = KGCNDataset(x_train)
test_dataset = KGCNDataset(x_test)
train_loader_user = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
test_loader_user = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

# prepare network, loss function, optimizer
num_user, num_entity, num_relation = data_loader.get_num()
user_encoder, entity_encoder, relation_encoder = data_loader.get_encoders()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
# device = torch.device('cpu')
net = KGCN(num_user, num_entity, num_relation,
           user_kg, kg, args, device).to(device)
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)
print('device: ', device)

# train
loss_list = []
test_loss_list = []
auc_score_list = []

for epoch in range(args.n_epochs):
  # Item based KG
    running_loss = 0.0
    for i, (user_ids, item_ids, labels) in enumerate(train_loader):
        user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(user_ids, item_ids, mode='item')
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item() 
        

    # # User based KG
    # running_loss = 0.0
    # for i, (user_ids, item_ids, labels) in enumerate(train_loader_user):
    #     user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
    #     optimizer.zero_grad()
    #     outputs = net(user_ids, item_ids, mode='user')
    #     loss = criterion(outputs, labels)
    #     loss.backward()

    #     optimizer.step()

    #     running_loss += loss.item()

    # print train loss per every epoch
    print('[Epoch {}]train_loss: '.format(epoch+1), running_loss / len(train_loader))
    loss_list.append(running_loss / len(train_loader))

    # evaluate per every epoch
    with torch.no_grad():
        labels_list, outputs_list = [], []
        recommender_metrics = Metrics()
        test_loss = 0
        total_roc = 0
        for user_ids, item_ids, labels in test_loader:
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            outputs = net(user_ids, item_ids, mode='item')
            test_loss += criterion(outputs, labels).item()
            try:
                recommender_metrics.roc_accuracy(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                recommender_metrics.confusion_metrix(labels.cpu().detach().numpy(), np.round(outputs.cpu().detach().numpy()))
                labels_list.append(labels.cpu().detach().numpy()), outputs_list.append(outputs.cpu().detach().numpy())
            except:
                pass
        print('[Epoch {}]test_loss: '.format(epoch+1), test_loss / len(test_loader))
        print('[Epoch {}]auc: '.format(epoch+1),recommender_metrics.get_acc() / len(test_loader))
        print('[Epoch {0}]tpr: {1}, tnr: {2}'.format(epoch+1, recommender_metrics.get_conf()[0], recommender_metrics.get_conf()[1]))
        test_loss_list.append(test_loss / len(test_loader))
        auc_score_list.append(recommender_metrics.get_acc() / len(test_loader))

# plot losses / scores
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns
ax1.plot(loss_list)
ax1.plot(test_loss_list)
ax2.plot(auc_score_list)

plt.tight_layout()
plt.show()
