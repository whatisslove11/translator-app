import torch
import math


def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class MLP(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.linear_0 = torch.nn.Linear(hidden_dim, 4 * hidden_dim)
        self.linear_1 = torch.nn.Linear(4 * hidden_dim, hidden_dim)

        mean = 0
        std = (2 / (5 * hidden_dim)) ** 0.5
        torch.nn.init.normal_(self.linear_0.weight, mean=mean, std=std)
        torch.nn.init.normal_(self.linear_1.weight, mean=mean, std=std)

    def forward(self, hidden_state):
        return self.linear_1(new_gelu(self.linear_0(hidden_state))) + hidden_state
