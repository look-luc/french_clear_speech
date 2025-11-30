import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class reg_model(nn.Module):
    def __init__(
            self,
            num_numerical_features,
            num_vowels,
            hidden_layer,
            embedding_dim=5,
            dropout_rate=0.2
    ):
        super(reg_model,self).__init__()
        self.vowel_embedding = nn.Embedding(num_embeddings=num_vowels, embedding_dim=embedding_dim)
        combined_input_size = num_numerical_features + embedding_dim

        self.regressor = nn.Sequential(
            nn.Linear(combined_input_size, hidden_layer),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer, 1)
        )

    def forward(self, x_num, x_cat):
        vowel_embed = self.vowel_embedding(x_cat)
        if vowel_embed.dim() > 2:
            vowel_embed = vowel_embed.squeeze(1)

        combined_features = torch.cat([x_num, vowel_embed], dim=1)
        x = self.regressor(combined_features)
        return x


def run_training(df, params):
    # Unpack parameters
    num_epochs, batch_size, hidden_dim = params

    # --- Data Prep ---
    unique_vowels = sorted(df["vowelSAMPA"].unique())
    vowel_to_index = {vowel: i for i, vowel in enumerate(unique_vowels)}
    df['vowel_index'] = df["vowelSAMPA"].map(vowel_to_index)
    num_vowels = len(unique_vowels)

    numerical_cols = [col for col in df.columns if col not in ["vowelSAMPA", 'vowel_index', "Target"]]
    features_num = df[numerical_cols].values.astype(np.float32)
    cat_indices = df['vowel_index'].values.astype(np.int64)  # Ensure int64 for embeddings
    targets_num = df["Target"].values.astype(np.float32)

    # Split
    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
        features_num, cat_indices, targets_num, test_size=0.2, random_state=42, stratify=cat_indices
    )

    # Scale Inputs
    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled = scaler.transform(X_num_test)  # Use transform only!

    # Scale Targets
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()  # FIXED: Use transform only!

    # --- Speed Optimization: Move data to GPU *before* DataLoader ---
    # Tabular data is usually small enough to fit entirely in VRAM.
    # This prevents CPU-GPU transfer bottlenecks during the loop.

    t_X_num_train = torch.tensor(X_num_train_scaled, dtype=torch.float32).to(device)
    t_X_cat_train = torch.tensor(X_cat_train, dtype=torch.long).to(device)
    t_y_train = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1).to(device)

    t_X_num_test = torch.tensor(X_num_test_scaled, dtype=torch.float32).to(device)
    t_X_cat_test = torch.tensor(X_cat_test, dtype=torch.long).to(device)
    t_y_test = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1).to(device)

    # Use TensorDataset (Standard PyTorch class, no need for custom class for simple arrays)
    train_dataset = TensorDataset(t_X_num_train, t_X_cat_train, t_y_train)
    val_dataset = TensorDataset(t_X_num_test, t_X_cat_test, t_y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Model Setup ---
    model = reg_model(
        num_numerical_features=X_num_train_scaled.shape[1],
        num_vowels=num_vowels,
        hidden_layer=hidden_dim,
    )
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    # --- Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs_num, inputs_cat, batch_target in train_dataloader:
            # Data is already on device, so we skip the .to(device) step here
            optimizer.zero_grad()
            outputs = model(inputs_num, inputs_cat)
            loss = criterion(outputs, batch_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs_num, val_inputs_cat, val_targets in val_dataloader:
                val_outputs = model(val_inputs_num, val_inputs_cat)
                v_loss = criterion(val_outputs, val_targets)
                val_loss += v_loss.item()

        avg_val_loss = val_loss / len(val_dataloader)

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # FIXED: Actually save the weights
            best_model_wts = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Load best weights before returning
    model.load_state_dict(best_model_wts)

    return best_val_loss, avg_train_loss, avg_val_loss


if __name__ == "__main__":
    # Load data ONCE outside the loop
    try:
        df = pd.read_csv("../data/vowel_data_all_LabPhon.csv")

        # Configuration
        epoch_range = 200
        batches = 128
        hidden_layer = 128
        parameters = [epoch_range, batches, hidden_layer]

        print(f"Epochs: {parameters[0]} Batch: {parameters[1]} Hidden: {parameters[2]}")

        best_loss, final_train, final_val = run_training(df, parameters)

        print(f"Final Best Val Loss: {best_loss:.6f}")

    except FileNotFoundError:
        print("Error: CSV file not found. Please check the path.")
