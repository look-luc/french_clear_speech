import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import regression_model

def run_training(df, params, device):
    # Unpack parameters
    num_epochs, batch_size, hidden_dim = params

    # --- Data Prep ---
    numerical_cols = [col for col in df.columns if col not in ["vowelSAMPA", "Target"]]
    features_num = df[numerical_cols].values.astype(np.float32)
    targets_num = df["Target"].values.astype(np.float32)

    X_num_train, X_num_test, y_train, y_test = train_test_split(
        features_num, targets_num, test_size=0.2, random_state=42
    )

    # Scale Inputs
    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled = scaler.transform(X_num_test)  # Use transform only!

    # Scale Targets
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()  # FIXED: Use transform only!

    t_X_num_train = torch.tensor(X_num_train_scaled, dtype=torch.float32).to(device)
    t_y_train = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1).to(device)

    t_X_num_test = torch.tensor(X_num_test_scaled, dtype=torch.float32).to(device)
    t_y_test = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1).to(device)

    train_dataset = TensorDataset(t_X_num_train, t_y_train)
    val_dataset = TensorDataset(t_X_num_test, t_y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Model Setup ---
    model = regression_model.RegressionModel(
        num_numerical_features=X_num_train_scaled.shape[1],
        hidden_layer=hidden_dim,
    )
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    # --- Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs_num, batch_target in train_dataloader:
            # Data is already on device, so we skip the .to(device) step here
            optimizer.zero_grad()
            outputs = model(inputs_num)
            loss = criterion(outputs, batch_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs_num, val_targets in val_dataloader:
                val_outputs = model(val_inputs_num)
                v_loss = criterion(val_outputs, val_targets)
                val_loss += v_loss.item()

        avg_val_loss = val_loss / len(val_dataloader)

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Load best weights before returning
    model.load_state_dict(best_model_wts)

    return best_val_loss, avg_train_loss, avg_val_loss