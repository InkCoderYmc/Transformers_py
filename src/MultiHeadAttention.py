import torch
from torch import nn
from SelfAttention import SelfAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_k * num_heads, bias=False)
        self.W_k = nn.Linear(d_model, d_k * num_heads, bias=False)
        self.W_v = nn.Linear(d_model, d_v * num_heads, bias=False)
        self.fc = nn.Linear(d_v * num_heads, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        """
        Args:
            Q (_type_): [batch_size, len_q, d_model]
            K (_type_): [batch_size, len_k, d_model]
            V (_type_): [batch_size, len_v, d_model]
            attn_mask (_type_): [batch_size, n_heads, seq_len, seq_len]
        """
        residual, batch_size = Q, Q.size(0)
        # Q: [batch_size, num_heads, len_q, d_k]
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # K: [batch_size, num_heads, len_k, d_k]
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # V: [batch_size, num_heads, len_v, d_v]
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        # attn_mask: [batch_size, num_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # prob: [batch_size, num_heads, len_q, d_v]
        # attn: [batch_size, num_heads, len_q, len_k]
        prob, attn = SelfAttention()(Q ,K ,V, attn_mask)

        # Concat: [batch_size, len_q, num_heads * d_v]
        concat = prob.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_v)
        output = self.fc(concat)
        return self.layer_norm(output + residual), attn

