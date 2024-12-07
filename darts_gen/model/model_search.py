import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import math

from model.operations import ops
from model.genotypes import encoder_primitives, decoder_primitives, Genotype
from utils import utils

class MixedOp(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, is_encoder):
        super(MixedOp, self).__init__()
        
        self.ops = nn.ModuleList()
        primitives = encoder_primitives if is_encoder else decoder_primitives
        for primitive in primitives:
          op = ops[primitive](hidden_size, filter_size, dropout_rate, is_encoder)
          self.ops.append(op)

    def forward(self, x, weights, enc_output, self_mask, i_mask):
        return sum(w * op(x, enc_output, self_mask, i_mask) for w, op in zip(weights, self.ops))

class SearchLayer(nn.Module):
    def __init__(self, steps, hidden_size, filter_size, dropout_rate, is_encoder, num_layer):
        super(SearchLayer, self).__init__()
        
        self.layers = nn.ModuleList()
        self.preprocess = nn.ModuleList()
        self.is_encoder = is_encoder
        self.steps = steps
        
        self.preprocess.append(ops['mha'](hidden_size, filter_size, dropout_rate, is_encoder))
        if not is_encoder:
            self.preprocess.append(ops['cmha'](hidden_size, filter_size, dropout_rate, is_encoder))
        for i in range(steps * num_layer):
            self.layers.append(MixedOp(hidden_size, filter_size, dropout_rate, is_encoder))
        
    def forward(self, x, weights, enc_output, self_mask, i_mask):
        output = x
        for pre_layer in self.preprocess:
            output = pre_layer(output, enc_output, self_mask, i_mask)
        for i, layer in enumerate(self.layers):
            output = layer(output, weights[i], enc_output, self_mask, i_mask)
        return output
    
class Transformer(nn.Module):
    def __init__(self, i_vocab_size, t_vocab_size,
                 n_layers=6,
                 hidden_size=512,
                 filter_size=2048,
                 dropout_rate=0.1,
                 share_target_embedding=True,
                 has_inputs=True,
                 src_pad_idx=None,
                 trg_pad_idx=None,
                 n_steps=2,
                 n_search_layers=1
                 ):
        super(Transformer, self).__init__()

        self.hidden_size = hidden_size
        self.emb_scale = hidden_size ** 0.5
        self.has_inputs = has_inputs
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.encoder_steps = n_steps
        self.decoder_steps = n_steps
        self.n_search_layers = n_search_layers
        assert n_layers % n_search_layers == 0
        
        self.t_vocab_embedding = nn.Embedding(t_vocab_size, hidden_size)
        nn.init.normal_(self.t_vocab_embedding.weight, mean=0,
                        std=hidden_size**-0.5)
        self.t_emb_dropout = nn.Dropout(dropout_rate)
        decoders = [SearchLayer(self.decoder_steps, hidden_size, filter_size, dropout_rate, False, n_search_layers) for _ in range(n_layers // n_search_layers)]
        self.decoder = nn.ModuleList(decoders)
        self.decoder_last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        if not share_target_embedding:
            self.i_vocab_embedding = nn.Embedding(i_vocab_size,
                                                  hidden_size)
            nn.init.normal_(self.i_vocab_embedding.weight, mean=0,
                            std=hidden_size**-0.5)
        else:
            self.i_vocab_embedding = self.t_vocab_embedding
        
        self.i_perturb = nn.Embedding(i_vocab_size,
                                        hidden_size)
        nn.init.normal_(self.i_perturb.weight, mean=0,
                        std=hidden_size**-0.5)
        self.i_emb_dropout = nn.Dropout(dropout_rate)
        encoders = [SearchLayer(self.encoder_steps, hidden_size, filter_size, dropout_rate, True, n_search_layers) for _ in range(n_layers // n_search_layers)]
        self.encoder = nn.ModuleList(encoders)
        self.encoder_last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        # For positional encoding
        num_timescales = self.hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            max(num_timescales - 1, 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)
        
        self.initialize_alphas()
        
    def initialize_alphas(self):
        k_encoder = self.encoder_steps * self.n_search_layers
        k_decoder = self.decoder_steps * self.n_search_layers

        self.alphas_encoder = nn.Parameter(1e-3*torch.randn(k_encoder, len(encoder_primitives)))
        self.alphas_decoder = nn.Parameter(1e-3*torch.randn(k_decoder, len(decoder_primitives)))
        self.arch_parameters = [
          self.alphas_encoder,
          self.alphas_decoder,
        ]
    
    def arch_params(self):
        return self.arch_parameters
        
    def weight_params(self):
        weight_params = iter(v for k, v in self.named_parameters() if "alphas_encoder" not in k and "alphas_decoder" not in k and "i_perturb" not in k)
        return weight_params
    
    def perturb_params(self):
        return iter(v for k, v in self.named_parameters() if "i_perturb" in k)
    
    def perturb_data(self, inputs):
        input_embedded = self.i_vocab_embedding(inputs)
        input_perturbed = input_embedded + self.i_perturb(inputs)
        return input_embedded, input_perturbed
        
    def forward(self, inputs, targets, perturb=False):
        enc_output, i_mask = None, None
        if self.has_inputs:
            i_mask = utils.create_pad_mask(inputs, self.src_pad_idx)
            enc_output = self.encode(inputs, i_mask, perturb=perturb)

        t_mask = utils.create_pad_mask(targets, self.trg_pad_idx)
        target_size = targets.size()[1]
        t_self_mask = utils.create_trg_self_mask(target_size,
                                                 device=targets.device)
        return self.decode(targets, enc_output, i_mask, t_self_mask, t_mask)
    
    def encode(self, inputs, i_mask, perturb):
        # Input embedding
        input_embedded = self.i_vocab_embedding(inputs)
        if perturb:
            input_embedded += self.i_perturb(inputs)
        input_embedded.masked_fill_(i_mask.squeeze(1).unsqueeze(-1), 0)
        input_embedded *= self.emb_scale
        input_embedded += self.get_position_encoding(inputs)
        input_embedded = self.i_emb_dropout(input_embedded)
        
        encoder_output = input_embedded
        for enc_layer in self.encoder:
            weights = F.softmax(self.alphas_encoder, dim = -1)
            encoder_output = enc_layer(encoder_output, weights, None, i_mask, None)
        encoder_output = self.encoder_last_norm(encoder_output)
        return encoder_output
        
    def decode(self, targets, enc_output, i_mask, t_self_mask, t_mask):
        # target embedding
        target_embedded = self.t_vocab_embedding(targets)
        target_embedded.masked_fill_(t_mask.squeeze(1).unsqueeze(-1), 0)

        # Shifting
        target_embedded = target_embedded[:, :-1]
        target_embedded = F.pad(target_embedded, (0, 0, 1, 0))

        target_embedded *= self.emb_scale
        target_embedded += self.get_position_encoding(targets)
        target_embedded = self.t_emb_dropout(target_embedded)

        # decoder
        decoder_output = target_embedded
        for dec_layer in self.decoder:
            weights = F.softmax(self.alphas_decoder, dim = -1)
            decoder_output = dec_layer(decoder_output, weights, enc_output, t_self_mask, i_mask)
        decoder_output = self.decoder_last_norm(decoder_output)
        
        # linear
        output = torch.matmul(decoder_output,
                              self.t_vocab_embedding.weight.transpose(0, 1))

        return output

    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32,
                                device=x.device)
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)
        signal = F.pad(signal, (0, 0, 0, self.hidden_size % 2))
        signal = signal.view(1, max_length, self.hidden_size)
        return signal

    def genotype(self):
        def parse(weights, primitives):
            gene = []
            n = 0
            for i in range(weights.shape[0]):
                best_op = weights[i].argmax()
                gene.append(primitives[best_op])
            return gene
    
        gene_encoder = parse(F.softmax(self.alphas_encoder, dim=-1).detach().cpu().numpy(), encoder_primitives)
        gene_decoder = parse(F.softmax(self.alphas_decoder, dim=-1).detach().cpu().numpy(), decoder_primitives)
        
        genotype = Genotype(
            encoder = gene_encoder,
            decoder = gene_decoder,
            n_search_layers = self.n_search_layers,
            encoder_n_steps = self.encoder_steps,
            decoder_n_steps = self.decoder_steps
        )
        return genotype