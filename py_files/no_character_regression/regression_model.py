import torch
from torch import nn

from training import run_training

'''
Setting the available GPU/CPU device
'''
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
            nn.LayerNorm(dim),  # normalizing the size coming in
            nn.GELU(),  # activation
            nn.Linear(dim, dim),  # making sure that it will be the right size coming out
            nn.Dropout(dropout_rate),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)  #this is just so that if it has seen something similar, it can skip or take a shortcut

class RegressionModel(nn.Module):
    #setting out the linear regression classification model
    def __init__(
            self,
            num_numerical_features,
            hidden_layer,
            dropout_rate=0.3  #getting rid of a neuron that is classifying similarly to another at a certain rate
    ):
        super().__init__()

        #the process of how the model will learn
        self.initial_projection = nn.Sequential(
            nn.Linear(num_numerical_features, hidden_layer),
            # making sure it is the right size, [|input| by |hidden layer defined|
            nn.LayerNorm(hidden_layer),  # normalizing the data to best learn
            nn.GELU(),  #activation
            nn.Dropout(dropout_rate)
        )

        self.residual_layers = nn.Sequential(
            ResidualBlock(hidden_layer, dropout_rate),
        )

        self.final_regressor = nn.Sequential(
            nn.Linear(hidden_layer, hidden_layer // 2),  # making the size of the hidden layer by the half of that layer
            nn.LayerNorm(hidden_layer // 2),  #normalizing that layer
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer // 2, 1)  # Output
        )

    def forward(self, x_num):  #going to the next step/next data point
        x = self.initial_projection(x_num)
        x = self.residual_layers(x)
        x = self.final_regressor(x)
        return x

def train_model(df, params):
    return run_training(df, params, device=device)