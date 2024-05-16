import torch
from torch import nn
from MultiHeadAttention import MultiHeadAttention
from FeedForwardNetwork import FeedForwardNetwork

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention()
        self.encoder_attention = MultiHeadAttention()
        self.feed_forward_network = FeedForwardNetwork()

    def forward(self, decode_input, encode_output, decode_self_attn_mask, decode_attn_mask):
        # self attention
        self_attn_output, self_attn = self.self_attention(decode_input, decode_input, decode_input, decode_self_attn_mask)
        # encoder attention
        encode_attn_output, encode_attn = self.encoder_attention(self_attn_output, encode_output, encode_output, decode_attn_mask)
        # feed forward network
        output = self.feed_forward_network(encode_attn_output)
        return output, self_attn, encode_attn