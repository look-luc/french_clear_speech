import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

import regression_model


def run_training(df, params, device):
    num_epochs, batch_size, hidden_dim = params

    # --- Data Prep ---
    numerical_cols = [col for col in df.columns if col not in ["vowelSAMPA", "Target"]]

    # making the list from the data compatable with the model
    # turning the normal list into a list of a specific type of float
    features_num = df[numerical_cols].values.astype(np.float32)

    # getting the targets for latter, for testing
    targets_num = df["Target"].values.astype(np.float32)

    #splitting the data 80 20 (80% training 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        features_num, targets_num, test_size=0.2, random_state=42
    )

    #having the scaler here to better standardize the range of the features of the data
    scaler = StandardScaler()

    '''
    we fit the data so that it can calculate the variance
    the transform is just taking the variance and manipulating the data to best represent the variance
    '''
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    '''
    we don't scale these values because we want the purest version of the feature data for the model
     to learn and best see the difference
    '''

    #training sets
    t_X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    t_y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    #testing set
    t_X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    t_y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    #putting both pairs of sets into their own dataset to use for training/validation
    train_dataset = TensorDataset(t_X_train, t_y_train)
    val_dataset = TensorDataset(t_X_test, t_y_test)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True  # shuffling so that the model doesn't memorize
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False  # no need to shuffle bc this is the 'test'
    )

    #pasing in one of the training sets into the model
    model = regression_model.RegressionModel(
        num_numerical_features=X_train_scaled.shape[1],
        hidden_layer=hidden_dim,
    )
    model.to(device)  #just putting into the right CPU/GPU

    criterion = nn.BCEWithLogitsLoss()  #calculator for loss with the Sigmoid function

    optimizer = torch.optim.AdamW(  # to check how the learning is going and updating the weights
        model.parameters(),
        lr=1e-3,
        weight_decay=5e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #updates the learning rate for each parameter group
        optimizer, T_0=100, T_mult=1, eta_min=1e-6
    )

    best_val_acc = 0.0  # for later use to check for the best weights and all
    best_model_wts = copy.deepcopy(model.state_dict())  #keeping the best version

    # --- Training Loop ---
    for epoch in range(num_epochs):
        model.train()  #set the model in learning mode
        train_loss = 0.0

        for inputs, targets in train_dataloader:  #looking at the dataset and getting the loss and optimizing
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  #clips the gradients
            optimizer.step()
            train_loss += loss.item()  #getting how much loss is there

        avg_train_loss = train_loss / len(train_dataloader)  #calculating the average loss

        # Validation
        model.eval()  #setting to testing mode
        val_loss = 0.0
        correct = 0  #how much it got right
        total = 0

        with torch.no_grad():
            for val_inputs, val_targets in val_dataloader:  #looking at the validation dataset
                val_outputs = model(val_inputs)
                v_loss = criterion(val_outputs, val_targets)
                val_loss += v_loss.item()

                probs = torch.sigmoid(val_outputs)
                preds = (probs > 0.5).float()

                correct += (preds == val_targets).sum().item()
                total += val_targets.size(0)

        avg_val_loss = val_loss / len(val_dataloader)  # calculating the average validation loss
        val_accuracy = correct / total  #calculate how accurate the model is at this stage

        scheduler.step(avg_val_loss)  #going to the next step

        # Save Best Model based on Accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 50 == 0:  #every 50th epoch, it will spit out the losses and accuracy
            print(
                f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

    # Load best weights
    model.load_state_dict(best_model_wts)  #loads best weights

    # Return best accuracy
    return best_val_acc, avg_train_loss, avg_val_loss
