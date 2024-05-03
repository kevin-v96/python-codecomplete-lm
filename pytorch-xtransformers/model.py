import torch 
import torch.nn as nn
import torch.nn.functional as F 
import math 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_layer, dropout) -> None:
        """
        d_model: The dimension of the input vector.
        num_heads: The number of heads in the multihead attention mechanism.
        ff_hidden_layer: The dimension of the feedforward hidden layer.
        dropout: The dropout rate.
        """
        super(DecoderBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, ff_hidden_layer)
        self.linear2 = nn.Linear(ff_hidden_layer, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, target_mask):
        """
        x: The input tensor.
        target_mask: Masks to prevent the attention to certain positions.
        """
        attn_output, _ = self.self_attention(x, x, x, attn_mask = target_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x

def generate_square_subsequent_mask(sz):
    """Generate a mask to prevent attention to future positions."""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, ff_hidden_layer, dropout, num_layers):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_blocks = nn.ModuleList([DecoderBlock(d_model, num_heads, ff_hidden_layer, dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        target_mask = generate_square_subsequent_mask(x.size(0))
        for block in self.transformer_blocks:
            x = block(x, target_mask)
        output = self.linear(x)
        output = self.softmax(output)
        return output
