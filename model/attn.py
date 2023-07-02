import torch
import math

def attention(K, V, Q, num_heads, mask=None):
    batch_size, hidden_dim = Q.size(0), Q.size(2)
    key_len, value_len, query_len = K.size(1), V.size(1), Q.size(1)

    K = K.reshape(batch_size, key_len, num_heads, -1)
    V = V.reshape(batch_size, value_len, num_heads, -1)
    Q = Q.reshape(batch_size, query_len, num_heads, -1)

    energy = torch.einsum('bqhd,bkhd->bhqk', [Q, K])

    if mask is not None:
        energy = energy.masked_fill(mask == 0, -1e12)

    attention = torch.softmax(energy / math.sqrt(hidden_dim // num_heads), dim=3)
    result_headed = torch.einsum('bhql,blhd->bqhd', [attention, V])
    return result_headed.reshape(batch_size, query_len, hidden_dim)


class AttentionModule(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.k_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.q_linear = torch.nn.Linear(hidden_dim, hidden_dim)

        self.out_linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, keys, values, query, mask):
        K = self.k_linear(keys)
        V = self.v_linear(values)
        Q = self.q_linear(query)
        attention_output = attention(K, V, Q, self.num_heads, mask)
        return self.out_linear(attention_output) + query