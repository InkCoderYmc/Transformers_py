import torch
from torch import nn
from PositionEncoding import PositionalEncoding
from EncoderLayer import EncoderLayer
from utils import get_attn_pad_mask

class Encoder(nn.Module):
    def __init__(self, source_len, d_model=512, num_encoder_layers=6):
        super(Encoder, self).__init__()
        self.input_embedding = nn.Embedding(source_len, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for layer in range(num_encoder_layers)])

    def forward(self, encoder_itputs):
        # input: [batch_size, source_len]
        # input_embedding: [batch_size, source_len, d_model]
        encoder_outputs = self.input_embedding(encoder_itputs)
        # input_positional_encoding: [batch_size, source_len, d_model]
        encoder_outputs = self.positional_encoding(encoder_outputs.transpose(0, 1)).transpose(0, 1)

        # 输入为单出的mutil-head attention,只需要增加pad掩码即可
        encode_self_attn_mask = get_attn_pad_mask(encoder_itputs, encoder_itputs)
        encoder_self_attns = []
        for layer in self.layers:
            encoder_outputs, encoder_self_attn = layer(encoder_outputs, encode_self_attn_mask)
            encoder_self_attns.append(encoder_self_attn)

        return encoder_outputs, encoder_self_attns