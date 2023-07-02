import torch
from .attn import AttentionModule
from .mlp import MLP

device = 'cpu'


class EncoderTransformerLayer(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.attention = AttentionModule(hidden_dim, num_heads)
        self.mlp = MLP(hidden_dim)

        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attn_output = self.dropout(self.norm(self.attention(value, key, query, mask)))
        mlp_output = self.dropout(self.norm(self.mlp(attn_output)))
        return mlp_output


# !g1.1
class Encoder(torch.nn.Module):
    def __init__(self, de_dictionary_size: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float = 0.1,
                 max_seq_len: int = 256):
        super().__init__()

        self.word_embedding = torch.nn.Embedding(de_dictionary_size, hidden_dim)
        self.pos_embedding = torch.nn.Embedding(max_seq_len, hidden_dim)
        self.layers = torch.nn.ModuleList(
            [
                EncoderTransformerLayer(
                    hidden_dim,
                    num_heads,
                    dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs, mask):
        batch_size, seq_len = inputs.shape
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(device)
        hidden_dim = self.dropout(self.word_embedding(inputs) + self.pos_embedding(positions))

        for layer in self.layers:
            hidden_dim = layer(hidden_dim, hidden_dim, hidden_dim, mask)

        return hidden_dim