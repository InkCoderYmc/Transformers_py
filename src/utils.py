import torch
from torch import nn
import numpy as np

def get_attn_pad_mask(seq_q, seq_k):
    """
    key padding mask
    Args:
        seq_q: [batch_size, seq_len]
        seq_k: [batch_size, seq_len]

    由于在 Encoder 和 Decoder 中都需要进行 mask 操作，因此就无法确定这个函数的参数中 seq_len 的值，如果是在 Encoder 中调用的,seq_len 就等于 src_len;如果是在 Decoder 中调用的,seq_len 就有可能等于 src_len,也有可能等于 tgt_len(因为 Decoder 有两次 mask)

    这个函数最核心的一句代码是 seq_k.data.eq(0)，这句的作用是返回一个大小和 seq_k 一样的 tensor,只不过里面的值只有 True 和 False。如果 seq_k 某个位置的值等于 0,那么对应位置就是 True,否则即为 False。举个例子,输入为 seq_data = [1, 2, 3, 4, 0],seq_data.data.eq(0) 就会返回 [False, False, False, False, True]
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # [batch_size, 1, len_k], False is masked
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # [batch_size, len_q, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def get_attn_subsequent_mask(seq):
    """
    key padding mask
    Args:
        seq: [batch_size, target_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # [batch_size, target_len, target_len]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask