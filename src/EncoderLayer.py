import torch
from torch import nn
from FeedForwardNetwork import FeedForwardNetwork
from MultiHeadAttention import MultiHeadAttention

class EncoderLayer(nn.module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention()
        self.feed_forward_network = FeedForwardNetwork()

    def forward(self, encode_input, encode_self_attn_mask):
        encode_output, attn = self.multi_head_attention(encode_input, encode_input, encode_input, encode_self_attn_mask)
        encode_output = self.feed_forward_network(encode_output)
        return encode_output, attn