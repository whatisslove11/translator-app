import torch
from .encoder import (
    EncoderTransformerLayer,
    AttentionModule,
    device
)


class DecoderTransformerLayer(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.self_attention = AttentionModule(hidden_dim, num_heads)  # Аттенш на то, что происходит в переводе
        self.out_attention = EncoderTransformerLayer(hidden_dim, num_heads)  # Аттенш на то, что происходит в оригинале

        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, hidden_state, encoder_layer_output, src_mask, trg_mask):
        self_attn_output = self.dropout(
            self.norm(self.self_attention(hidden_state, hidden_state, hidden_state, trg_mask)))
        output = self.out_attention(encoder_layer_output, encoder_layer_output, self_attn_output, src_mask)
        return output


# !g1.1
class Decoder(torch.nn.Module):
    def __init__(self, en_dictionary_size: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float = 0.1,
                 max_seq_len: int = 256):
        super().__init__()

        self.word_embedding = torch.nn.Embedding(en_dictionary_size, hidden_dim)
        self.pos_embedding = torch.nn.Embedding(max_seq_len, hidden_dim)
        self.layers = torch.nn.ModuleList(
            [
                DecoderTransformerLayer(hidden_dim, num_heads)
                for _ in range(num_layers)
            ]
        )

        self.lm_head = torch.nn.Linear(hidden_dim, en_dictionary_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs, encoder_output, src_mask, trg_mask):
        batch_size, seq_len = inputs.shape
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(device)
        inputs = self.dropout(self.word_embedding(inputs) + self.pos_embedding(positions))

        for layer in self.layers:
            inputs = layer(inputs, encoder_output, src_mask, trg_mask)

        return self.lm_head(inputs)