import torch
from torch import nn
from PositionEncoding import PositionalEncoding
from DecoderLayer import DecoderLayer
from utils import get_attn_pad_mask, get_attn_subsequent_mask

class Decoder(nn.Module):
    def __init__(self, target_len, d_model=512, num_decoder_layers=6):
        super(Decoder, self).__init__()
        self.target_embedding = nn.Embedding(target_len, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for layer in range(num_decoder_layers)])

    def forward(self, decoder_inputs, encoder_inputs, encoder_outputs):
        # input: [batch_size, source_len]
        # input_embedding: [batch_size, source_len, d_model]
        decoder_outputs = self.target_embedding(decoder_inputs)
        # input_positional_encoding: [batch_size, source_len, d_model]
        decoder_outputs = self.positional_encoding(decoder_outputs.transpose(0, 1)).transpose(0, 1)

        # pad掩码 [batch_size, target_len, target_len]
        decoder_pad_mask = get_attn_pad_mask(decoder_inputs, decoder_inputs)
        # subseq掩码 [batch_size, target_len, target_len]
        decoder_subseq_mask = get_attn_subsequent_mask(decoder_inputs)
        # decoder整体掩码
        decoder_mask = torch.gt((decoder_pad_mask + decoder_subseq_mask), 0)

        # decoder-encoder的pad掩码
        decoder_encoder_pad_mask = get_attn_pad_mask(decoder_inputs, encoder_inputs)

        decoder_self_attns = []
        decoder_encoder_attns = []
        for layer in self.layers:
            decoder_outputs, decoder_self_attn, decoder_encoder_attn = layer(decoder_outputs, encoder_outputs, decoder_mask, decoder_encoder_pad_mask)
            decoder_self_attns.append(decoder_self_attn)
            decoder_encoder_attns.append(decoder_encoder_attn)

        return decoder_outputs, decoder_self_attns, decoder_encoder_attns