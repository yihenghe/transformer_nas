from collections import namedtuple

Genotype = namedtuple('Genotype', 'encoder decoder n_search_layers encoder_n_steps decoder_n_steps')

encoder_primitives = [
    'ffn',
    'mha', # multi-head attention
    'identity', # return itse
    'cnn_1', # cnn1d with kernel_size 1
    'cnn_3', # cnn1d with kernel_size 3
    'dep_sep_cnn_3', # depthwise separable cnn1d with kernel_size 3
    'dep_sep_cnn_5', # depthwise separable cnn1d with kernel_size 5
    'dep_sep_cnn_7', # depthwise separable cnn1d with kernel_size 7
    'dep_sep_cnn_9', # depthwise separable cnn1d with kernel_size 9
    'dep_sep_cnn_11', # depthwise separable cnn1d with kernel_size 11
    'dyn_cnn_3', # dynamic cnn1d with kernel_size 3
    'dyn_cnn_7', # dynamic cnn1d with kernel_size 7
    'dyn_cnn_11', # dynamic cnn1d with kernel_size 11
    'dyn_cnn_15', # dynamic cnn1d with kernel_size 15
    'glu' # gated linear unit
]

decoder_primitives = encoder_primitives[:] + ['cmha'] # cross multi-head attention (only for decoder)
