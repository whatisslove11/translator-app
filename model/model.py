import torch
from .encoder import (
    Encoder,
    device
)
from .decoder import Decoder

PAD_token = 0
BOS_token = 1
EOS_token = 2
UNK_token = 3


class TranslationModel(torch.nn.Module):
    def __init__(self, de_dictionary_size: int, en_dictionary_size: int, hidden_dim: int = 512, num_layers: int = 6,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.encoder = Encoder(de_dictionary_size, hidden_dim, num_layers, num_heads, dropout)
        self.decoder = Decoder(en_dictionary_size, hidden_dim, num_layers, num_heads, dropout)

    def make_src_mask(self, src):
        src_pad_mask = (src != PAD_token).unsqueeze(1).unsqueeze(2)
        return src_pad_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != PAD_token).unsqueeze(1).unsqueeze(2)

        batch_size, trg_len = trg.shape
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        ).bool().to(device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, inputs):
        src_ids, trg_ids = inputs
        src_mask = self.make_src_mask(src_ids).to(device)
        trg_mask = self.make_trg_mask(trg_ids).to(device)
        encoder_output = self.encoder(src_ids, src_mask)
        decoder_output = self.decoder(trg_ids, encoder_output, src_mask, trg_mask)
        return decoder_output