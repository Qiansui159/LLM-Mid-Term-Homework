import math
import torch
import torch.nn as nn
import torch.optim as optim

# Step 3: Transformer 模型构建

# 定义多头注意力机制
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale = d_k ** -0.5  # 缩放因子

    def forward(self, Q, K, V, mask=None):
        """
        :param Q: [batch_size, heads, seq_len, d_k]
        :param K: [batch_size, heads, seq_len, d_k]
        :param V: [batch_size, heads, seq_len, d_v]
        :param mask: [batch_size, 1, 1, seq_len] 或 [batch_size, 1, seq_len, seq_len]
        :return: 注意力输出和注意力权重
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch_size, heads, seq_len, seq_len]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)  # [batch_size, heads, seq_len, d_v]
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_k = self.d_v = d_model // num_heads
        self.num_heads = num_heads

        # 定义线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k)
        self.dropout = nn.Dropout(0.1)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 线性变换并分头
        Q = self.w_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        K = self.w_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        V = self.w_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)        # 计算注意力
        if mask is not None:
            mask = mask.repeat(1, self.num_heads, 1, 1)  # 扩展维度以匹配多头
        output, attn = self.attention(Q, K, V, mask=mask)# 拼接多头的输出
        output = output.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.d_v)
        output = self.fc(output)
        return output

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10 ** -6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features))  # bias is a learnable parameter
    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# 定义前馈神经网络
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

# 定义位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, pos_state=True):
        super().__init__()
        # 创建一个 [max_len, d_model] 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # 奇数和偶数位置分别使用 sin 和 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        pe = pe.unsqueeze(0)  # 增加 batch 维度
        self.register_buffer('pe', pe)
        self.pos_state = pos_state

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        if self.pos_state:
            x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

# 定义编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力子层
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        # 前馈神经网络子层
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x

# 定义解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, trg_mask=None):
        # 掩码多头自注意力子层
        self_attn_output = self.self_attn(x, x, x, trg_mask)
        x = x + self.dropout(self_attn_output)
        x = self.norm1(x)
        # 编码器-解码器注意力子层
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = x + self.dropout(cross_attn_output)
        x = self.norm2(x)
        # 前馈神经网络子层
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm3(x)
        return x

# 定义编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers, dropout, pos_state=True):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, pos_state=pos_state)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src: [batch_size, src_len]
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

# 定义解码器
class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, num_heads, d_ff, num_layers, dropout, pos_state=True):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(output_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, pos_state=pos_state)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, trg, enc_output, src_mask=None, trg_mask=None):
        # trg: [batch_size, trg_len]
        x = self.embedding(trg) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, trg_mask)
        output = self.fc_out(x)
        return output

# 定义 Transformer 模型
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, en_vocab, zh_vocab):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.en_vocab = en_vocab
        self.zh_vocab = zh_vocab

    def make_src_mask(self, src):
        # 生成源序列的掩码，屏蔽填充位置
        src_mask = (src != self.en_vocab['<pad>']).unsqueeze(1).unsqueeze(2)
        return src_mask  # [batch_size, 1, 1, src_len]

    def make_trg_mask(self, trg):
        # 生成目标序列的掩码，包含填充位置和未来信息
        trg_pad_mask = (trg != self.zh_vocab['<pad>']).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, trg_len]
        trg_len = trg.size(1)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()  # [trg_len, trg_len]
        trg_mask = trg_pad_mask & trg_sub_mask  # [batch_size, 1, trg_len, trg_len]
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_output = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_output, src_mask, trg_mask)
        return output


def build_transformer_model(en_vocab, zh_vocab, d_model=512, num_heads=8, d_ff=2048, num_layers=6, dropout=0.1, pos_state=True):
    input_dim = len(en_vocab)
    output_dim = len(zh_vocab)
    encoder = Encoder(input_dim, d_model, num_heads, d_ff, num_layers, dropout, pos_state=pos_state)
    decoder = Decoder(output_dim, d_model, num_heads, d_ff, num_layers, dropout, pos_state=pos_state)
    model = Transformer(encoder, decoder, en_vocab, zh_vocab)
    return model