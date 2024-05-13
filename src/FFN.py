import torch
from torch import nn

class FeedForwardLayer(nn.Module):
    # 根据论文中解释，本质为两层的MLP
    # FFN(x) = ReLU(xW1^T+b1)W2+b2
    def __init__(self, d_model: int, d_ff: int, dropout=0.1):
        """
        Args:
            d_model (int): FFN的输入维度
            d_ff (int): FFN的隐藏层维度
            dropout (float, optional): drop率. Defaults to 0.1.
        """
        super(FeedForwardLayer, self).__init__()
        # Linear实现了xW1^T+b1
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.rule = nn.ReLU()
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        """
        Args:
            x (_type_): 输入数据，[batch_size, seq_len, d_model]
        Returns:
            output (_type_): 输出数据，[batch_size, seq_len, d_model]
        """
        # 先定义残差计算的x
        residual = x
        inter = self.dropout_1(self.rule(self.w_1(x)))
        ouput = self.dropout_2(self.w_2(inter))
        return self.norm(ouput + residual)