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
    X_train_scaled = scaler.fit_transform(X_num_train)
    X_test_scaled = scaler.transform(X_num_test)

    # Scale Targets
    t_X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    t_y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)  # No target scaling

    t_X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    t_y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)  # No target scaling

    train_dataset = TensorDataset(t_X_train, t_y_train)
    val_dataset = TensorDataset(t_X_test, t_y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Model Setup ---
    model = regression_model.RegressionModel(
        num_numerical_features=X_train_scaled.shape[1],
        hidden_layer=hidden_dim,
    )
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=1, eta_min=1e-6
    )

    best_val_loss = 0.0
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for val_inputs, val_targets in val_dataloader:
                val_outputs = model(val_inputs)
                v_loss = criterion(val_outputs, val_targets)
                val_loss += v_loss.item()

                # Calculate Accuracy
                # Apply sigmoid to turn logits into probabilities (0 to 1)
                probs = torch.sigmoid(val_outputs)
                preds = (probs > 0.5).float()
                correct += (preds == val_targets).sum().item()
                total += val_targets.size(0)

        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = correct / total

        scheduler.step(avg_val_loss)

        # Save Best Model
        if val_accuracy >= best_val_loss:
            best_val_loss = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Load best weights before returning
    model.load_state_dict(best_model_wts)

    return best_val_loss, avg_train_loss, avg_val_loss
