import math
import os
from itertools import product

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class two_lay_data(Dataset):
    def __init__(self,features,target):
        self.features = torch.FloatTensor(features)
        self.targets = torch.tensor(target, dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class reg_model(nn.Module):
    def __init__(self, input_features, hidden_layer1, hidden_layer2, hidden_layer3, hidden_layer4):
        super(reg_model,self).__init__()
        self.layer1 = nn.Linear(input_features, hidden_layer1)
        self.bn1 = nn.BatchNorm1d(hidden_layer1)

        self.layer2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.bn2 = nn.BatchNorm1d(hidden_layer2)

        self.layer3 = nn.Linear(hidden_layer2, hidden_layer3)
        self.bn3 = nn.BatchNorm1d(hidden_layer3)

        self.layer4 = nn.Linear(hidden_layer3, hidden_layer4)
        self.bn4 = nn.BatchNorm1d(hidden_layer4)

        self.layer5 = nn.Linear(hidden_layer4, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.layer5(x)
        return x

def string_to_ascii_list(text):
    if pd.isna(text):  # Handle NaN values
        return None
    return int("".join([ascii(char) for char in str(text).encode('ascii', errors='replace')]))

def test(input:list[int]):
    df = pd.read_csv("/Users/lucdenardi/Desktop/python/french_clear_speach/data/vowel_data_all_LabPhon.csv")

    features_all = df.drop(['target'], axis=1).values
    targets_all = df['target'].values

    X_train, X_val, y_train, y_val = train_test_split(
        features_all, targets_all, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_dataset = two_lay_data(X_train, y_train)
    val_dataset = two_lay_data(X_val, y_val)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=input[1],
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == 'cuda'
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=input[1],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == 'cuda'
    )

    model = reg_model(
        input_features=X_train.shape[1],
        hidden_layer1=input[2],
        hidden_layer2=input[3],
        hidden_layer3=input[4],
        hidden_layer4=input[4]
    )
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for Data Parallelism.")
        model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = input[0]
    for epochs in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs,batch_targets in train_dataloader:
            inputs, batch_targets = inputs.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f'Epoch {epochs+1}/{num_epochs}, Loss: {loss.item():.4f}')
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, batch_targets in val_dataloader:
            inputs, batch_targets = inputs.to(device), batch_targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, batch_targets)
            val_loss += loss.item()

    print(f'Epoch {epochs+1}/{num_epochs}, ',
            f'Train Loss: {train_loss / len(train_dataloader):.4f}, ',
            f'Val Loss: {val_loss / len(val_dataloader):.4f}')
    
    return train_loss / len(train_dataloader), val_loss / len(val_dataloader)

if __name__ == "__main__":
    epoch_range = 2500
    batches = 512
    hidden_layer_1_ranges = [64, 128]
    hidden_layer_2_ranges = [32, 64]
    hidden_layer_3_ranges = [16, 32]
    hidden_layer_range = list(product(hidden_layer_1_ranges, hidden_layer_2_ranges, hidden_layer_3_ranges))
    minimum_out = {"epoch": 2500, "batch": 512, "hidden layer": 128, "Train Loss": 0.0315, "Val Loss": 0.0585}
    best_hl = {"hidden layer 1": [], "hidden layer 2": [], "hidden layer 3": [], "Train Loss": math.inf,
               "Val Loss": math.inf}
    for hidden_layer in hidden_layer_range:
        parameter = [epoch_range, batches, hidden_layer[0], hidden_layer[1], hidden_layer[2]]
        print(f"epoch number: {parameter[0]} batch: {parameter[1]} hidden layer: {parameter[2:]}")
        train_loss, val_loss = test(parameter)
        if train_loss < best_hl["Train Loss"] and val_loss < best_hl["Val Loss"]:
            best_hl["hidden layer 1"] = hidden_layer[0]
            best_hl["hidden layer 2"] = hidden_layer[1]
            best_hl["hidden layer 3"] = hidden_layer[2]
            best_hl["Train Loss"] = train_loss
            best_hl["Val Loss"] = val_loss

    for element1, element2 in best_hl.items():
        print(f"{element1} : {element2}")
    for key, value in minimum_out.items():
        print(f"{key}: {value}")
