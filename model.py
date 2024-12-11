import torch
import torch.nn as nn
from Embedding import InputEmbeddings, PositionalEncoding
from layers import ResidualConnection, LayerNormalization, ProjectionLayer, MultiHeadAttentionBlock,FeedForwardBlock



import logging
from tqdm import tqdm


def build_transformer(device, src_vocab_size: int, tgt_vocab_size: int,
                      src_seq_len: int, tgt_seq_len: int,
                      d_model: int =512,
                      N: int =6,
                      h: int =8,
                      dropout: float =0.1,
                      d_ff: int =2048):
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size).to(device)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size).to(device)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout).to(device)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout).to(device)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout).to(device)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout).to(device)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout).to(device)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout).to(device)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout).to(device)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks)).to(device)
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks)).to(device)

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size).to(device)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer).to(device)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) # Initialize the parameters

    return transformer.to(device)




class EncoderBlock(nn.Module):

    def __init__(self, features: int,
                 self_attention_block,
                 feed_forward_block,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        """ self.feed_forward_block 是一个前馈神经网络模块，
           它实际上包含了两个全连接层，尽管这些层在这段代码中没有直接展示。 """
        self.feed_forward_block = feed_forward_block

        """ nn.ModuleList 用于将多个模块组织在一起，可以像列表一样访问这些模块 """
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """ """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x



""" 累加多个Encoder Block 增加模型深度和表达能力"""
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)





class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block,
                 cross_attention_block, feed_forward_block,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                 src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):

    def __init__(self, encoder, decoder,
                      src_embed, tgt_embed,
                      src_pos, tgt_pos,
                     projection_layer: ProjectionLayer) -> None:
        """ src_embed - input x from dataset, like "我喜欢猫"
           tgt-embed - input y from dataset, like "I like cat" """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)


    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        定义前向传播逻辑
        :param src: 源语言输入
        :param tgt: 目标语言输入
        :param src_mask: 源语言掩码
        :param tgt_mask: 目标语言掩码
        :return: 预测输出 (batch, seq_len, vocab_size)
        """
        # 编码器输出
        encoder_output = self.encode(src, src_mask)
        # 解码器输出
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        # 投影到词汇表大小
        output = self.project(decoder_output)
        return output

