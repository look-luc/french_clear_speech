import os

import numpy as np

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class data(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

class reg_model(nn.Module):
    def __init__(self, num_numerical_features, num_vowels, hidden_layer, embedding_dim=10):
        super(reg_model,self).__init__()
        self.vowel_embedding = nn.Embedding(num_embeddings=num_vowels, embedding_dim=embedding_dim)
        combined_input_size = num_numerical_features + embedding_dim

        self.regressor = nn.Sequential(
            nn.Linear(combined_input_size, hidden_layer),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_layer, hidden_layer // 2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_layer // 2, hidden_layer // 4),
            nn.ReLU(),
            nn.Linear(hidden_layer // 4, 1),
        )

    def forward(self, x_num, x_cat):
        vowel_embed = self.vowel_embedding(x_cat)
        combined_features = torch.cat([x_num, vowel_embed], dim=1)

        x = self.regressor(combined_features)
        return x

def test(input:list[int]):
    df = pd.read_csv("/Users/lucdenardi/Desktop/python/french_clear_speach/data/vowel_data_all_LabPhon.csv")

    unique_vowels = df["vowelSAMPA"].unique()
    vowel_to_index = {vowel: i for i, vowel in enumerate(unique_vowels)}
    df['vowel_index'] = df["vowelSAMPA"].map(vowel_to_index)
    num_vowels = len(unique_vowels)

    numerical_cols = [col for col in df.columns if col not in ["vowelSAMPA", 'vowel_index', "Target"]]

    features_num = df[numerical_cols].values.astype(np.float32)
    targets_num = df["Target"].values.astype(np.float32)

    scaler = StandardScaler()
    numerical_features_scaled = scaler.fit_transform(features_num)

    cat_indices = df['vowel_index'].values
    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
        numerical_features_scaled, cat_indices, targets_num, test_size=0.2, random_state=42
    )

    train_dataset = data(X_num_train, X_cat_train, y_train)
    val_dataset = data(X_num_test, X_cat_test, y_test)

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
        num_numerical_features=X_num_train.shape[1],
        num_vowels=num_vowels,
        hidden_layer=input[2],
        embedding_dim=10
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
        for inputs_num, inputs_cat, batch_target in train_dataloader:
            inputs_num, inputs_cat, batch_target = inputs_num.to(device), inputs_cat.to(device), batch_target.to(device)
            optimizer.zero_grad()
            outputs = model(inputs_num, inputs_cat)
            loss = criterion(outputs, batch_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f'Epoch {epochs + 1}/{num_epochs}, Train Batch Loss: {train_loss / len(train_dataloader):.4f}')
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs_num, val_inputs_cat, val_targets in val_dataloader:
            val_inputs_num, val_inputs_cat, val_targets = val_inputs_num.to(device), val_inputs_cat.to(
                device).long(), val_targets.to(device)
            val_outputs = model(val_inputs_num, val_inputs_cat)

            v_loss = criterion(val_outputs, val_targets)
            val_loss += v_loss.item()

    print(f'Epoch {epochs+1}/{num_epochs}, ',
            f'Train Loss: {train_loss / len(train_dataloader):.4f}, ',
            f'Val Loss: {val_loss / len(val_dataloader):.4f}')
    
    return train_loss / len(train_dataloader), val_loss / len(val_dataloader)

if __name__ == "__main__":
    epoch_range = 10200
    batches = 64
    hidden_layer = 128
    minimum_out = {"epoch": 4200, "batch": 64, "hidden layer": 512, "Train Loss": 0.2501, "Val Loss": 0.2501}
    parameter = [epoch_range, batches, hidden_layer]
    print(f"epoch number: {parameter[0]} batch: {parameter[1]} hidden layer: {parameter[2:]}")
    train_loss, val_loss = test(parameter)
    if train_loss < minimum_out["Train Loss"] and val_loss < minimum_out["Val Loss"]:
        minimum_out["epoch"] = parameter[0]
        minimum_out["batch"] = parameter[1]
        minimum_out["hidden layer"] = parameter[2]
        minimum_out["Train Loss"] = train_loss
        minimum_out["Val Loss"] = val_loss

    for key, value in minimum_out.items():
        print(f"{key}: {value}")
