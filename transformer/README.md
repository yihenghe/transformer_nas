
# Transformer

This is a pytorch implementation of the
[Transformer](https://arxiv.org/abs/1706.03762) model.

## Prerequisite

```
$ conda create -n nmt python=3.8 --yes
$ source activate nmt
$ pip3 install -r requirements.txt
$ pip3 install torch==1.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
$ python -m spacy download en_core_web_sm
$ python -m spacy download de_core_news_sm
$ python -m spacy download fr_core_news_sm
```

## Usage

1. Processs dataset. We use En-De as an example
```
$ cd dataset && bash process_de.sh && cd ..
```

2. Train the model
```
$ python train.py --problem wmt32k --batch_size 4096 --output_dir ./output --data_dir ./wmt32k_data_de --share_target_embedding --average_checkpoints --lan de --use_bpe --seed 1 // add --produce_test_set when run the model for the first time
```

3. Test the trained model
```
$ python test.py --problem wmt32k --model_dir ./output/last/models --data_dir ./wmt32k_data_de --output_dir ./output --share_target_embedding --average_checkpoints --lan de --use_bpe // add --test_set for ig or ha
```
