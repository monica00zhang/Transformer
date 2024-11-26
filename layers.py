import torch
import torch.nn as nn
import math

class ProjectionLayer(nn.Module):
    """ 将模型输出的每个 token 的表示转换为该 token 在词汇表中的概率分布
        如果是分类任务，vocab size会是 num classes"""

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class ResidualConnection(nn.Module):
    def __init__(self, features, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNormalization(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return (x + self.dropout(sublayer(self.norm(x))))



class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        """ (batch size, seq_len, d_model) -- > (batch, seq_len, d_ff)映射到更高的维度，捕捉更多信息 """
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) # relu激活函数，非线性变换，帮助模型学习复杂的特征


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h" # 为什么要整除

        self.d_k = d_model // h  # Dimension of vector seen by each head

        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq --> (batch, seq_len, d_model)
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):

        d_k = query.shape[-1] # d_k 是向量长度
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)

        """
        query @ key.transpose(-2, -1)：这是矩阵乘法操作，
        其中 key.transpose(-2, -1) 表示将 key 的最后两个维度转置。
        通过这种操作，计算每个查询和所有键之间的相似度，生成一个注意力得分矩阵。
        """
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        """
        attention_scores.masked_fill_(mask == 0, -1e9)：
        将 mask 中为 0 的位置对应的 attention_scores 值设为一个非常小的值（例如 -1e9），
        这相当于在 softmax 中将这些位置设为 -∞，确保它们在 softmax 后的概率接近于零。
        """
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """ q12 表示 1表示序列的第一个元素，2表示第二个头"""
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        """
        view 从新调整为 张量
        query.view(...) 将 query 的形状从 (batch, seq_len, d_model) 变为 (batch, seq_len, h, d_k)，其中：
        h 是注意力头的数量（self.h）。
        d_k 是每个注意力头的维度，通常是 d_model / h。
        不变换写 矩阵计算效率低
        """
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        """
        x.transpose(1, 2) 将 x 的形状从 (batch, h, seq_len, d_k) 变为 (batch, seq_len, h, d_k)，以便后续合并多个注意力头的结果。
        .contiguous().view(...) 将形状重塑为 (batch, seq_len, d_model)，其中 d_model = h * d_k，将所有注意力头的结果拼接起来。
        """
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)
