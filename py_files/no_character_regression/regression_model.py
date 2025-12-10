import torch
from torch import nn

from training import run_training

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))

class RegressionModel(nn.Module):
    def __init__(
            self,
            num_numerical_features,
            hidden_layer,
            dropout_rate=0.2
    ):
        super().__init__()

        self.initial_projection = nn.Sequential(
            nn.Linear(num_numerical_features, hidden_layer),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        self.residual_layers = nn.Sequential(
            ResidualBlock(hidden_layer, dropout_rate),
            ResidualBlock(hidden_layer, dropout_rate),
        )

        self.final_regressor = nn.Sequential(
            nn.Linear(hidden_layer, hidden_layer // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer // 2, 1)  # Output
        )

    def forward(self, x_num):
        x = self.initial_projection(x_num)
        x = self.residual_layers(x)
        x = self.final_regressor(x)
        return x

def train_model(df, params):
    return run_training(df, params, device=device)