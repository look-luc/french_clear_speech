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
    def __init__(self, input_features, hidden_layer, dropout_rate=0.2):
        super(reg_model,self).__init__()
        self.layer1 = nn.Linear(input_features, hidden_layer)
        self.bn1 = nn.BatchNorm1d(hidden_layer)

        self.layer2 = nn.Linear(hidden_layer, hidden_layer//2)
        self.bn2 = nn.BatchNorm1d(hidden_layer//2)

        self.layer3 = nn.Linear(hidden_layer//2, hidden_layer//4)
        self.bn3 = nn.BatchNorm1d(hidden_layer//4)

        self.layer4 = nn.Linear(hidden_layer//4, hidden_layer//2)
        self.bn4 = nn.BatchNorm1d(hidden_layer//2)

        self.layer5 = nn.Linear(hidden_layer//2, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)

        x = F.relu(self.layer2(x))
        x = self.dropout(x)

        x = F.relu(self.layer3(x))
        x = self.dropout(x)

        x = F.relu(self.layer4(x))
        x = self.dropout(x)

        x = self.layer5(x)
        return x

def string_to_ascii_list(text):
    if pd.isna(text):  # Handle NaN values
        return None
    return int("".join([ascii(char) for char in str(text).encode('ascii', errors='replace')]))

def test(input:list[int]):
    nasal = pd.read_csv("/Users/lucdenardi/Desktop/python/french_clear_speach/data/nasal copy.csv")
    nasal['vowelSAMPA'] = nasal['vowelSAMPA'].apply(string_to_ascii_list)

    nasal_oral = pd.read_csv("/Users/lucdenardi/Desktop/python/french_clear_speach/data/nasal_normal copy.csv")
    nasal_oral['vowelSAMPA'] = nasal_oral['vowelSAMPA'].apply(string_to_ascii_list)

    df = pd.concat([
        nasal_oral,
        nasal
    ])

    features_all = df.drop(['target', 'vowelSAMPA'], axis=1).values
    targets_all = df['target'].values

    X_train_, dummyval, y_train, dummy_y_val = train_test_split(
        features_all, targets_all, test_size=0.2, random_state=42
    )

    df_eval = pd.concat([
        nasal_oral.sample(frac=0.8),
        nasal.sample(frac=0.2)
    ])

    eval_all = df_eval.drop(['target', 'vowelSAMPA'], axis=1).values
    targets_val_all = df_eval['target'].values

    dummytrain,X_val,dummy_y_train, y_val = train_test_split(
        eval_all, targets_val_all, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_)
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

    model = reg_model(input_features=X_train.shape[1], hidden_layer=input[2])
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
    epoch_range = 10500
    batches = 512
    hidden_layer_range = 128
    parameter = [epoch_range,batches,hidden_layer_range]
    minimum_out = {"epoch":2500, "batch":512, "hidden layer":128, "Train Loss":0.0315, "Val Loss":0.0585}
    print(f"epoch number: {parameter[0]} batch: {parameter[1]} hidden layer: {parameter[2]}")
    train_loss, val_loss = test(parameter)
    if train_loss < minimum_out["Train Loss"] and val_loss < minimum_out["Val Loss"]:
        minimum_out["epoch"] = parameter[0]
        minimum_out["batch"] = parameter[1]
        minimum_out["hidden layer"] = parameter[2]
        minimum_out["Train Loss"] = train_loss
        minimum_out["Val Loss"] = val_loss

    for key, value in minimum_out.items():
        print(f"{key}: {value}")
