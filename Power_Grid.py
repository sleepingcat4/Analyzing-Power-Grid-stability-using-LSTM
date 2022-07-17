import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# df = gpd.read_file('electric-network-qatar.geojson')
# df = gpd.read_file('substations.geojson')
# type(df)

df = pd.read_csv('Data_for_UCI_named.csv')

# print(df.head())
# print(df.head())

# df = pd.read_csv('electric-network-qatar.geojson')
# print(df.head())

df1 = df.reset_index()['stab']
# print(df1)

stabf_plot = plt.plot(df1)
# print(stabf_plot)
# plt.show()

#noralzing the dataset value into 0 and 1
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler(feature_range=(0, 1))
df1 = scalar.fit_transform(np.array(df1).reshape(-1,1))

# Seeing the numpy array
# print(df1)

# Spilting the dataset into train and test
training_size = int(len(df1) * 0.8)
test_size = len(df1) - training_size
train_data, test_data = df1[0:training_size,:], df1[training_size:len(df1),:]

# Doing Pytroch thingy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class ElecDataset(Dataset):
    def __init__(self, data, seq_len = 180):
        self.data = data
        self.data = torch.from_numpy(data).float().view(-1)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len] , self.data[idx+self.seq_len]

train_dataset = ElecDataset(train_data)
test_dataset = ElecDataset(test_data)

batch_size = 10000
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

class lstm_model(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers):
        super(lstm_model, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_dim
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hn, cn):
        out, (hn, cn) = self.lstm(x, (hn, cn))
        final_out = self.fc(out[:, -1, :])
        return final_out, (hn, cn)
    
    def predict(self, x):
        out, hn, cn = self.init()
        final_out = self.fc(out[:, -1, :])
        return final_out
    
    
    def init(self):
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(device)
        return h0, c0

input_dim = 1
hidden_size = 128
num_layers = 1
model = lstm_model(input_dim, hidden_size, num_layers).to(device)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(dataloader):
    hn, cn = model.init()
    model.train()
    for batch, item in enumerate(dataloader):
        x, y = item
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out, (hn, cn) = model(x.reshape(100, batch_size), hn, cn)
        loss = loss_func(out.reshape(batch_size), y)
        loss.backward()
        hn = hn.detach()
        cn = cn.detach()
        optimizer.step()
        if batch == len(dataloader) - 1:
            print(loss.item())
            print(f'train loss: {loss}')


def test(dataloader):
    hn, cnn = model.init()
    model.eval()
    for batch, item in enumerate(dataloader):
        x, y = item
        x = x.to(device)
        y = y.to(device)
        out, (hn, cn) = model(x.reshape(100, batch_size, 1), hn, cn)
        loss = loss_func(out.reshape(batch_size), y)
        if batch == len(dataloader) - 1:
            loss = loss.item()
            print(f'test loss: {loss}')


epochs = 200
for epoch in range(epochs):
    print(f'Epoch {epoch}')
    train(train_data_loader)
    test(test_data_loader)

import math
from sklearn.metrics import mean_squared_error
def calculate_metrics(data_load):
    pred_arr = []
    y_arr = []
    with torch.no_grad():
        hn, cn = model.init()
        for batch, item in enumerate(data_load):
            x, y = item
            x = x.to(device)
            y = y.to(device)
            x = x.view(100, batch_size, 1)
            pred = model(x, hn, cn)[0]
            pred = scalar.inverse_transform(pred.detach().numpy().reshape(-1,1))
            y = scalar.inverse_transform(y.detach().numpy().reshape(-1,1))
            pred_arr = pred_arr + list(pred)
            y_arr = y_arr + list(y)
        #return math.sqrt(mean_squared_error(y_arr, pred_arr))


            # out, (hn, cn) = model(x.reshape(100, batch_size, 1), hn, cn)
            # pred_arr.append(out.reshape(batch_size))
            # y_arr.append(y)

# calculating final loss metrics
print(f'train MSE loss {calculate_metrics(train_data_loader)}')
print(f'test MSE loss {calculate_metrics(test_data_loader)}')