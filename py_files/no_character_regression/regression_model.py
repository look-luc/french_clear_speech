from sympy.printing.pytorch import torch
from torch import nn
from training import run_training

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class RegressionModel(nn.Module):
    def __init__(
            self,
            num_numerical_features,
            hidden_layer,
            dropout_rate=0.2
    ):
        super(RegressionModel,self).__init__()

        self.regressor = nn.Sequential(
            nn.Linear(num_numerical_features, hidden_layer),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer, 1)
        )

    def forward(self, x_num):
        x = self.regressor(x_num)
        return x

def train_model(df, params):
    return run_training(df, params, device=device)