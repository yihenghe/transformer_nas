from collections import Counter, OrderedDict
import glob
import io
import os
import pickle
import re
import math

import torch
import spacy
from torchtext import data
from tqdm import tqdm

from dataset import common

# pylint: disable=arguments-differ


url = re.compile('(<url>.*</url>)')
spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')
spacy_fr = spacy.load('fr_core_news_sm')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

def tokenize_fr(text):
    return [tok.text for tok in spacy_fr.tokenizer(url.sub('@URL@', text))]

def tokenize_cs(text):
    return [tok.text for tok in spacy_fr.tokenizer(url.sub('@URL@', text))]
    
tokenizer = {
    'de': tokenize_de,
    'en': tokenize_en,
    'fr': tokenize_fr,
    'cs': tokenize_cs,
    'ig': tokenize_en,
    'ha': tokenize_en
}

def read_examples(paths, exts, fields, data_dir, mode, filter_pred, num_shard):
    data_path_fmt = data_dir + '/examples-' + mode + '-{}.pt'
    data_paths = [data_path_fmt.format(i) for i in range(num_shard)]
    writers = [open(data_path, 'wb') for data_path in data_paths]
    shard = 0

    for path in paths:
        print("Preprocessing {}".format(path))
        src_path, trg_path = tuple(path + x for x in exts)

        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
            for src_line, trg_line in tqdm(zip(src_file, trg_file),
                                           ascii=True):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line == '' or trg_line == '':
                    continue

                example = data.Example.fromlist(
                    [src_line, trg_line], fields)
                if not filter_pred(example):
                    continue

                pickle.dump(example, writers[shard])
                shard = (shard + 1) % num_shard

    for writer in writers:
        writer.close()

    # Reload pickled objects, and save them again as a list.
    common.pickles_to_torch(data_paths)

    examples = torch.load(data_paths[0])
    return examples, data_paths


class WMT32k(data.Dataset):
    urls = ['http://data.statmt.org/wmt18/translation-task/'
            'training-parallel-nc-v13.tgz',
            'http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz',
            'http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz',
            'http://data.statmt.org/wmt17/translation-task/dev.tgz',
            'https://storage.googleapis.com/cloud-tpu-test-datasets/'
            'transformer_data/newstest2014.tgz']
#    urls = ['http://data.statmt.org/wmt17/translation-task/'
#            'training-parallel-nc-v12.tgz',
#            'http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz',
#            'http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz',
#            'http://www.statmt.org/wmt13/training-parallel-un.tgz',
#            'http://www.statmt.org/wmt10/training-giga-fren.tar',
#            'http://data.statmt.org/wmt17/translation-task/dev.tgz']
#   # process test dataset via process_fr_test.sh
    name = 'wmt32k'
    dirname = ''

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    @classmethod
    def splits(cls, exts, fields, data_dir, root='.data', test_flag=False, bpe_dir=None, produce_test_set=False, **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        filter_pred = kwargs['filter_pred']
        if bpe_dir is None:
            expected_dir = os.path.join(root, cls.name)
            path = (expected_dir if os.path.exists(expected_dir)
                    else cls.download(root))

            train_files = ['training-parallel-nc-v13/news-commentary-v13.de-en',
                           'commoncrawl.de-en',
                           'training/europarl-v7.de-en']
    #        train_files = ['training/news-commentary-v12.fr-en',
    #               'commoncrawl.fr-en',
    #               'training/europarl-v7.fr-en',
    #               'un/undoc.2000.fr-en',
    #               'giga-fren.release2.fixed']
    #        #consult https://github.com/facebookresearch/fairseq/blob/main/examples/translation/prepare-wmt14en2fr.sh
            train_files = map(lambda x: os.path.join(path, x), train_files)
        else:
            train_files = [os.path.join(bpe_dir, 'train')]
        num_files = 100
        ig_or_ha = bpe_dir is not None and (bpe_dir.endswith('ig') or bpe_dir.endswith('ha'))
        if ig_or_ha:
            num_files = 1 if bpe_dir.endswith('ig') else 2
        train_examples, data_paths = \
            read_examples(train_files, exts, fields, data_dir, 'train',
                          filter_pred, num_files)
        if test_flag or produce_test_set:
            if ig_or_ha:
                test_files = [os.path.join(bpe_dir, 'gnome_test'), os.path.join(bpe_dir, 'ubuntu_test')]
                for test_file in test_files:
                    test_name = test_file[len(bpe_dir)+1:]
                    test_examples, _ = read_examples([test_file], exts, fields, data_dir,
                                            test_name, filter_pred, 1)
            else:
                test_files = [os.path.join(path, 'newstest2014')] if bpe_dir is None else [os.path.join(bpe_dir, 'test')]
                test_examples, _ = read_examples(test_files, exts, fields, data_dir,
                                            'test', filter_pred, 1)
        if not test_flag:
            val_files = [os.path.join(path, 'dev/newstest2013')] if bpe_dir is None else [os.path.join(bpe_dir, 'valid')]
            val_examples, _ = read_examples(val_files, exts, fields, data_dir,
                                            'val', filter_pred, 1)

        train_data = cls(train_examples, fields, **kwargs)
        if test_flag:
            test_data = cls(test_examples, fields, **kwargs)
            return (train_data, test_data, data_paths)
        else:
            val_data = cls(val_examples, fields, **kwargs)
            return (train_data, val_data, data_paths)


def len_of_example(example):
    return max(len(example.src) + 1, len(example.trg) + 1)


def build_vocabs(src_field, trg_field, data_paths, share_target_embedding):
    src_counter = Counter()
    trg_counter = src_counter if share_target_embedding else Counter()
    for data_path in tqdm(data_paths, ascii=True):
        examples = torch.load(data_path)
        for x in examples:
            src_counter.update(x.src)
            trg_counter.update(x.trg)

    specials = list(OrderedDict.fromkeys(
        tok for tok in [src_field.unk_token,
                        src_field.pad_token,
                        src_field.init_token,
                        src_field.eos_token]
        if tok is not None))
    src_field.vocab = src_field.vocab_cls(src_counter, specials=specials,
                                          min_freq=50)
    trg_field.vocab = trg_field.vocab_cls(trg_counter, specials=specials,
                                          min_freq=50)


def prepare(max_length, batch_size, device, opt, data_dir, test_flag=False):
    pad = '<pad>'
    src_path = data_dir + '/source_share.pt' if opt.share_target_embedding else data_dir + '/source.pt'
    trg_path = data_dir + '/target_share.pt' if opt.share_target_embedding else data_dir + '/target.pt'
    load_preprocessed = os.path.exists(src_path)

    def filter_pred(x):
        return len(x.src) < max_length and len(x.trg) < max_length

    if load_preprocessed:
        print("Loading preprocessed data...")
        src_field = torch.load(src_path)['field']
        trg_field = torch.load(trg_path)['field']

        data_paths = glob.glob(data_dir + '/examples-train-*.pt')
        examples_train = torch.load(data_paths[0])
        if test_flag:
            examples_dev = torch.load(data_dir + '/examples-test-0.pt')
        else:
            examples_dev = torch.load(data_dir + '/examples-val-0.pt')

        fields = [('src', src_field), ('trg', trg_field)]
        train = WMT32k(examples_train, fields, filter_pred=filter_pred)
        dev = WMT32k(examples_dev, fields, filter_pred=filter_pred)
    else:
        src_field = data.Field(tokenize=tokenizer['en'], batch_first=True,
                               pad_token=pad, lower=True, eos_token='<eos>')
        trg_field = data.Field(tokenize=tokenizer[opt.lan], batch_first=True,
                               pad_token=pad, lower=True, eos_token='<eos>')

        print("Loading data... (this may take a while)")
        train, dev, data_paths = \
            WMT32k.splits(exts=('.en', f'.{opt.lan}'),
                          fields=(src_field, trg_field),
                          data_dir=data_dir,
                          filter_pred=filter_pred,
                          test_flag=test_flag)

        print("Building vocabs... (this may take a while)")
        build_vocabs(src_field, trg_field, data_paths, opt.share_target_embedding)

    print("Creating iterators...")
    train_iter, dev_iter = common.BucketByLengthIterator.splits(
        (train, dev),
        data_paths=data_paths,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
        example_length_fn=len_of_example)

    opt.src_vocab_size = len(src_field.vocab)
    opt.trg_vocab_size = len(trg_field.vocab)
    opt.src_pad_idx = src_field.vocab.stoi[pad]
    opt.trg_pad_idx = trg_field.vocab.stoi[pad]

    if not load_preprocessed:
        
        torch.save({'pad_idx': opt.src_pad_idx, 'field': src_field},
                       src_path)
        torch.save({'pad_idx': opt.trg_pad_idx, 'field': trg_field},
                       trg_path)

    return train_iter, dev_iter, opt




# bpe
def prepare_bpe(max_length, batch_size, device, opt, data_dir, test_flag=False):
    pad = '<pad>'
    src_path = data_dir + '/source_share.pt'
    trg_path = data_dir + '/target_share.pt'
    load_preprocessed = os.path.exists(src_path)
    
    def filter_pred(x):
            return len(x.src) < max_length and len(x.trg) < max_length
            
    if load_preprocessed:
        print("Loading preprocessed data...")
        src_field = torch.load(src_path)['field']
        trg_field = torch.load(trg_path)['field']
        
        data_paths = glob.glob(data_dir + '/examples-train-*.pt')
        examples_train = torch.load(data_paths[0])
        if test_flag:
            test_name = f'/examples-{opt.test_set}_test-0.pt' if opt.lan in ['ig', 'ha'] and opt.test_set is not None else '/examples-test-0.pt'
            examples_dev = torch.load(data_dir + test_name)
        else:
            examples_dev = torch.load(data_dir + '/examples-val-0.pt')

        fields = [('src', src_field), ('trg', trg_field)]
        train = WMT32k(examples_train, fields, filter_pred=filter_pred)
        dev = WMT32k(examples_dev, fields, filter_pred=filter_pred)
    else:
        dataset_dir = os.path.join(os.path.dirname(data_dir), 'dataset')
#        os.system(f'bash {dataset_dir}/process_{opt.lan}.sh')
        
        if opt.lan in ['ig', 'ha']:
            lan_dir = os.path.join(dataset_dir, f'en_{opt.lan}')
        else:
            lan_dir = os.path.join(dataset_dir, f'wmt14_en_{opt.lan}')
        if opt.lan in tokenizer:
            src_field = data.Field(tokenize=tokenizer['en'], batch_first=True,
                                   pad_token=pad, lower=True, eos_token='<eos>')
            trg_field = data.Field(tokenize=tokenizer[opt.lan], batch_first=True,
                                   pad_token=pad, lower=True, eos_token='<eos>')
        else:
            src_field = data.Field(tokenize=str.split, batch_first=True,
                                   pad_token=pad, lower=True, eos_token='<eos>')
            trg_field = data.Field(tokenize=str.split, batch_first=True,
                                   pad_token=pad, lower=True, eos_token='<eos>')
                            
        print("Loading data... (this may take a while)")
        train, dev, data_paths = \
            WMT32k.splits(exts=('.en', f'.{opt.lan}'),
                          fields=(src_field, trg_field),
                          data_dir=data_dir,
                          filter_pred=filter_pred,
                          test_flag=test_flag,
                          bpe_dir=lan_dir,
                          produce_test_set=opt.produce_test_set if hasattr(opt, "produce_test_set") else False)
        
        print("Building vocabs... (this may take a while)")
        build_vocabs(src_field, trg_field, data_paths, True)
            
    print("Creating iterators...")
    train_iter, dev_iter = common.BucketByLengthIterator.splits(
        (train, dev),
        data_paths=data_paths,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
        example_length_fn=len_of_example)

    opt.src_vocab_size = len(src_field.vocab)
    opt.trg_vocab_size = len(trg_field.vocab)
    opt.src_pad_idx = src_field.vocab.stoi[pad]
    opt.trg_pad_idx = trg_field.vocab.stoi[pad]

    if not load_preprocessed:
        
        torch.save({'pad_idx': opt.src_pad_idx, 'field': src_field},
                       src_path)
        torch.save({'pad_idx': opt.trg_pad_idx, 'field': trg_field},
                       trg_path)
                       
    return train_iter, dev_iter, opt
