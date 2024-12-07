import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils

ops = {
    'ffn': lambda hidden_size, filter_size, dropout_rate, is_encoder: FFNModule(hidden_size, filter_size, dropout_rate),
    'mha': lambda hidden_size, filter_size, dropout_rate, is_encoder: MHAModule(hidden_size, dropout_rate),
    'cmha': lambda hidden_size, filter_size, dropout_rate, is_encoder: CMHAModule(hidden_size, dropout_rate),
    'identity': lambda hidden_size, filter_size, dropout_rate, is_encoder: Identity(),
    'cnn_1': lambda hidden_size, filter_size, dropout_rate, is_encoder: CNNModule(hidden_size, dropout_rate, k=1, is_encoder=is_encoder),
    'cnn_3': lambda hidden_size, filter_size, dropout_rate, is_encoder: CNNModule(hidden_size, dropout_rate, k=3, is_encoder=is_encoder),
    'dep_sep_cnn_3': lambda hidden_size, filter_size, dropout_rate, is_encoder: DepSepCNNModule(hidden_size, dropout_rate, k=3, is_encoder=is_encoder),
    'dep_sep_cnn_5': lambda hidden_size, filter_size, dropout_rate, is_encoder: DepSepCNNModule(hidden_size, dropout_rate, k=5, is_encoder=is_encoder),
    'dep_sep_cnn_7': lambda hidden_size, filter_size, dropout_rate, is_encoder: DepSepCNNModule(hidden_size, dropout_rate, k=7, is_encoder=is_encoder),
    'dep_sep_cnn_9': lambda hidden_size, filter_size, dropout_rate, is_encoder: DepSepCNNModule(hidden_size, dropout_rate, k=9, is_encoder=is_encoder),
    'dep_sep_cnn_11': lambda hidden_size, filter_size, dropout_rate, is_encoder: DepSepCNNModule(hidden_size, dropout_rate, k=11, is_encoder=is_encoder),
    'dyn_cnn_3': lambda hidden_size, filter_size, dropout_rate, is_encoder: DynCNNModule(hidden_size, dropout_rate, k=3, is_encoder=is_encoder),
    'dyn_cnn_7': lambda hidden_size, filter_size, dropout_rate, is_encoder: DynCNNModule(hidden_size, dropout_rate, k=7, is_encoder=is_encoder),
    'dyn_cnn_11': lambda hidden_size, filter_size, dropout_rate, is_encoder: DynCNNModule(hidden_size, dropout_rate, k=11, is_encoder=is_encoder),
    'dyn_cnn_15': lambda hidden_size, filter_size, dropout_rate, is_encoder: DynCNNModule(hidden_size, dropout_rate, k=15, is_encoder=is_encoder),
    'glu': lambda hidden_size, filter_size, dropout_rate, is_encoder: GLUModule(hidden_size, dropout_rate)
}

pool_ops = {
    'maxpool': lambda out_features: nn.AdaptiveMaxPool1d(out_features),
    'avgpool': lambda out_features: nn.AdaptiveAvgPool1d(out_features)
}

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class FFNModule(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FFNModule, self).__init__()
        # FFN with layernorm, dropout, and residual connection
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, enc_output, self_mask, i_mask):
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
        
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size=8):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias=False)
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size,
                                      bias=False)
        initialize_weight(self.output_layer)

    def forward(self, q, k, v, mask, cache=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        if cache is not None and 'encdec_k' in cache:
            k, v = cache['encdec_k'], cache['encdec_v']
        else:
            k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
            v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

            if cache is not None:
                cache['encdec_k'], cache['encdec_v'] = k, v

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q.mul_(self.scale)
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        x.masked_fill_(mask.unsqueeze(1).bool(), -1e9)
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x

class MHAModule(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(MHAModule, self).__init__()
        # MHA with layernorm, dropout, and residual connection
        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, enc_output, self_mask, i_mask):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, self_mask)
        y = self.self_attention_dropout(y)
        x = x + y
        return x

class CMHAModule(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(CMHAModule, self).__init__()
        # CMHA with layernorm, dropout, and residual connection
        self.enc_dec_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.enc_dec_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.enc_dec_attention_dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, enc_output, self_mask, i_mask):
        y = self.enc_dec_attention_norm(x)
        y = self.enc_dec_attention(y, enc_output, enc_output, i_mask)
        y = self.enc_dec_attention_dropout(y)
        x = x + y
        return x
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x, enc_output, self_mask, i_mask):
        return x


class DepthwiseSeparableCNN(nn.Module):
    def __init__(self, hidden_size, k):
        super(DepthwiseSeparableCNN, self).__init__()
        
        self.depthwise = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=k, padding=k//2, groups=hidden_size, bias = False)
        self.pointwise = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, padding = 0, bias = True)
        self.relu = nn.ReLU()
        nn.init.kaiming_normal_(self.pointwise.weight, nonlinearity="relu")
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = x.transpose(1, 2)
        x = self.relu(x)
        return x

class DepSepCNNModule(nn.Module):
    def __init__(self, hidden_size, dropout_rate, k, is_encoder):
        super(DepSepCNNModule, self).__init__()
        # DepSepCNNModule with layernorm, dropout, and residual connection
        self.cnn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cnn = DepthwiseSeparableCNN(hidden_size, k=k)
        self.cnn_dropout = nn.Dropout(dropout_rate)
        self.is_encoder = is_encoder
        assert (k - 1) % 2 == 0
        self.shift = int((k - 1) / 2)
    
    def forward(self, x, enc_output, self_mask, i_mask):
        y = x
        if not self.is_encoder and self.shift != 0:
            y = y[:, :-self.shift]
            y = F.pad(y, (0, 0, self.shift, 0))
        y = self.cnn_norm(y)
        y = self.cnn(y)
        y = self.cnn_dropout(y)
        x = x + y
        return x
        
class CNNModule(nn.Module):
    def __init__(self, hidden_size, dropout_rate, k, is_encoder):
        super(CNNModule, self).__init__()
        # CNNModule with layernorm, dropout, and residual connection
        self.cnn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size=k, padding=k//2)
        self.cnn_dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        nn.init.kaiming_normal_(self.cnn.weight, nonlinearity="relu")
        self.is_encoder = is_encoder
        assert (k - 1) % 2 == 0
        self.shift = int((k - 1) / 2)
    
    def forward(self, x, enc_output, self_mask, i_mask):
        y = x
        if not self.is_encoder and self.shift != 0:
            y = y[:, :-self.shift]
            y = F.pad(y, (0, 0, self.shift, 0))
        y = self.cnn_norm(y)
        y = y.transpose(1, 2)
        y = self.cnn(y)
        y = y.transpose(1, 2)
        y = self.relu(y)
        y = self.cnn_dropout(y)
        x = x + y
        return x
        
class GLUModule(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(GLUModule, self).__init__()
        # GLUModule with layernorm, dropout, and residual connection
        self.glu_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.glu_dropout = nn.Dropout(dropout_rate)
        
        initialize_weight(self.layer1)
        initialize_weight(self.layer2)
    
    def forward(self, x, enc_output, self_mask, i_mask):
        y = self.glu_norm(x)
        values = self.layer1(y)
        gates = self.sigmoid(self.layer2(y))
        y = values * gates
        y = self.glu_dropout(y)
        x = x + y
        return x

class FactorizedEmbedding(nn.Module):
    def __init__(self, num_embedding, hidden_size, latent_size):
        super(FactorizedEmbedding, self).__init__()
        
        self.emb1 = nn.Embedding(num_embedding, latent_size)
        self.emb2 = nn.Linear(latent_size, hidden_size, bias=False)
    
    def forward(self, x):
        x = self.emb1(x)
        x = self.emb2(x)
        return x

# borrow from fairseq
def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

def unfold1d(x, kernel_size, padding_l, pad_value=0):
    '''unfold T x B x C to T x B x C x K'''
    if kernel_size > 1:
        T, B, C = x.size()
        x = F.pad(x, (0, 0, 0, 0, padding_l, kernel_size - 1 - padding_l), value=pad_value)
        x = x.as_strided((T, B, C, kernel_size), (B*C, C, 1, B*C))
    else:
        x = x.unsqueeze(3)
    return x
    
class DynamicConv1d(nn.Module):
    '''Dynamic lightweight convolution taking T x B x C inputs
    Args:
        input_size: # of channels of the input
        kernel_size: convolution channels
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout: the drop rate of the DropConnect to drop the weight
    Shape:
        Input: TxBxC, i.e. (timesteps, batch_size, input_size)
        Output: TxBxC, i.e. (timesteps, batch_size, input_size)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(input_size)`
    '''
    def __init__(self, input_size, kernel_size=1, num_heads=1,
                 weight_dropout=0.):
        super(DynamicConv1d, self).__init__()
        self.input_size = input_size
        self.query_size = input_size
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.weight_dropout= weight_dropout

        self.weight_linear = Linear(self.query_size, num_heads * kernel_size * 1, bias=False)
        self.conv_bias = nn.Parameter(torch.Tensor(input_size))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight_linear.reset_parameters()
        nn.init.constant_(self.conv_bias, 0.)

    def forward(self, x):
        '''Assuming the input, x, of the shape T x B x C and producing an output in the shape T x B x C
        args:
            x: Input of shape T x B x C, i.e. (timesteps, batch_size, input_size)
            incremental_state: A dict to keep the state
            unfold: unfold the input or not. If not, we use the matrix trick instead
            query: use the specified query to predict the conv filters
        '''

        query = x
        output = self._forward_unfolded(x, query)
        output = output + self.conv_bias.view(1, 1, -1)
        return output

    def _forward_unfolded(self, x, query):
        '''The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right.'''
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size

        weight = self.weight_linear(query).view(T*B*H, -1)
    
        # unfold the input: T x B x C --> T' x B x C x K
        padding_l = K - 1
        if K > T and padding_l == K-1:
            weight = weight.narrow(1, K-T, T)
            K, padding_l = T, T-1
        x_unfold = unfold1d(x, K, padding_l, 0)
        x_unfold = x_unfold.view(T*B*H, R, K)
        
        weight = weight.narrow(1, 0, K)
        
        weight = F.softmax(weight, dim=1)
        weight = F.dropout(weight, p=self.weight_dropout)
        
        output = torch.bmm(x_unfold, weight.unsqueeze(2))  # T*B*H x R x 1
        output = output.view(T, B, C)
        return output

class DynCNNModule(nn.Module):
    def __init__(self, hidden_size, dropout_rate, k, is_encoder):
        super(DynCNNModule, self).__init__()
        # DynCNNModule with layernorm, dropout, and residual connection
        self.cnn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cnn = DynamicConv1d(hidden_size, kernel_size=k, weight_dropout=dropout_rate)
        self.is_encoder = is_encoder
        assert (k - 1) % 2 == 0
        self.shift = int((k - 1) / 2)
    
    def forward(self, x, enc_output, self_mask, i_mask):
        y = x
        if not self.is_encoder and self.shift != 0:
            y = y[:, :-self.shift]
            y = F.pad(y, (0, 0, self.shift, 0))
        y = self.cnn_norm(y)
        y = y.transpose(0, 1)
        y = self.cnn(y)
        y = y.transpose(0, 1)
        x = x + y
        return x
