import pandas as pd

from regression_model import train_model

if __name__ == "__main__":
    try:
        df = pd.read_csv("../../data/vowel_data_all_LabPhon.csv")  # getting the data

        # Configuration
        epoch_range = 1500
        batches = 64
        hidden_layer = 128
        parameters = [epoch_range, batches, hidden_layer]

        print(f"Epochs: {parameters[0]} Batch: {parameters[1]} Hidden: {parameters[2]}")

        best_loss, final_train, final_val = train_model(df, parameters)

        print(f"Final Best Val Loss: {best_loss:.6f}")

    except FileNotFoundError:
        print("Error: CSV file not found. Please check the path.")