import pandas as pd

from regression_model import train_model

if __name__ == "__main__":
    try:
        df = pd.read_csv("../../data/vowel_data_nasality.csv")

        # Configuration
        epoch_range = 250
        batches = 128
        hidden_layer = 256
        parameters = [epoch_range, batches, hidden_layer]

        print(f"Epochs: {parameters[0]} Batch: {parameters[1]} Hidden: {parameters[2]}")

        best_loss, final_train, final_val = train_model(df, parameters)

        print(f"Final Best Val Loss: {best_loss:.6f}")

    except FileNotFoundError:
        print("Error: CSV file not found. Please check the path.")