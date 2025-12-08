from training import *

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class regression_model(nn.Module):
    def __init__(
            self,
            num_numerical_features,
            # num_vowels,
            hidden_layer,
            # embedding_dim=5,
            dropout_rate=0.2
    ):
        super(regression_model,self).__init__()
        # self.vowel_embedding = nn.Embedding(num_embeddings=num_vowels, embedding_dim=embedding_dim)
        # combined_input_size = num_numerical_features + embedding_dim

        self.regressor = nn.Sequential(
            nn.Linear(num_numerical_features, hidden_layer),
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

def train_model(df, params):
    return run_training(df, params, device=device)