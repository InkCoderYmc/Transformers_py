import torch
from torch import nn

class PositionalEncoding(nn.Module):
    """
    正弦位置编码,通过三角函数构建位置编码
    PE(pos,2_i)=sin(pos/10000^(2i/d_model))
    PE(pos,2_i+1)=cos(pos/10000^(2i/d_model))
    """
    def __init__(self, d_model: int, dropout=0.1, max_len=5000):
        """
        Args:
            d_model (int): 输入维度
            dropout (float, optional): drop率. Defaults to 0.1.
            max_len (int, optional): embedding向量长度. Defaults to 5000.
        """
        super(PositionalEncoding).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 初始化位置编码矩阵
        # torch.Size([max_len, d_model])
        positional_encoding = torch.zeros(max_len, d_model)
        # torch.Size([max_len, 1])
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # torch.Size(max_len/2)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=float) * (-torch.log(torch.tensor(10000.0)) / d_model))

        # 根据公式计算位置编码的值,偶数使用sin函数,奇数使用cos函数
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        # 添加一个维度，使得位置编码矩阵可以与输入矩阵相加
        # [max_len, d_model] -> [1, max_len, d_model] -> [max_len, 1, d_model]
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        """
        Args:
            x (_type_): 输入数据，[batch_size, seq_len, d_model]
        Returns:
            output (_type_): 输出数据，[batch_size, seq_len, d_model]
        """
        x = x + self.positional_encoding[:x.size(0), :]
        output = self.dropout(x)
        return output