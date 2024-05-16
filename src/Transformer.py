import torch
from torch import nn
from Encoder import Encoder
from Decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, d_model, source_len, target_len, vocab_size):
        super(Transformer, self).__init__()
        self.encoder = Encoder(source_len=source_len, d_model=d_model)
        self.decoder = Decoder(target_len=target_len, d_model=d_model)
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, encoder_inputs, decoder_inputs):
        encoder_outputs, encoder_self_attns = self.encoder(encoder_inputs)
        decoder_outputs, decoder_self_attns, decoder_encoder_attns = self.decoder(decoder_inputs, encoder_inputs, encoder_outputs)
        deocder_logits = self.projection(decoder_outputs)
        return deocder_logits.view(-1, deocder_logits.size(-1)), encoder_self_attns, decoder_self_attns, decoder_encoder_attns
