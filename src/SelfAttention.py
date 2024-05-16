import torch
from torch import nn
from numpy import np

class SelfAttention(nn.Module):
    """
    自注意力机制
    Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V
    """
    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(Q,K,V,attn_mask):
        """
        Args:
            Q (_type_): [batch_size, n_heads, len_q, d_k]
            K (_type_): [batch_size, n_heads, len_q, d_k]
            V (_type_): [batch_size, n_heads, len_q, d_v]
            attn_mask (_type_): [batch_size, n_heads, seq_len, seq_len]
        """
        attn_score = torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(K.size(-1))
        attn_score.mask_fill_(attn_mask,-1e9)

        # 对最后一个维度做softmax
        attn = nn.Softmax(dim=-1)(attn_score)
        # output: [batch_size, n_heads, len_q, d_v]
        output = torch.matmul(attn,V)

        return output, attn