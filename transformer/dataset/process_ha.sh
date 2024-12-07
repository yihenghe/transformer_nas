#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000

URLS=(
    "https://object.pouta.csc.fi/OPUS-CCMatrix/v1/moses/en-ha.txt.zip"
    "https://object.pouta.csc.fi/OPUS-GNOME/v1/moses/en-ha.txt.zip"
    "https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/moses/en-ha.txt.zip"
)
FILES=(
    "en-ha.txt.zip"
    "en-ha.txt.zip"
    "en-ha.txt.zip"
)
TRAINING_CORPORA=(
    "CCMatrix.en-ha"
)

TEST_CORPORA=(
    "GNOME.en-ha"
    "Ubuntu.en-ha"
)

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=ha
lang=en-ha
prep=en_ha
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

cd $orig

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        elif [ ${file: -4} == ".zip" ]; then
            unzip $file
            rm README && rm LICENSE && rm $file
        fi
    fi
done
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    for f in "${TRAINING_CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
    done
done

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    for f in "${TEST_CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
        if [ ${f:0:5} == "GNOME" ]; then
            mv $tmp/test.$l $tmp/gnome_test.$l
        elif [ ${f:0:6} == "Ubuntu" ]; then
            mv $tmp/test.$l $tmp/ubuntu_test.$l
        fi
    done
    echo ""
done

echo "splitting train and valid..."
for l in $src $tgt; do
    awk '{if (NR%100 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
    awk '{if (NR%100 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
done

TRAIN=$tmp/train.ig-ha
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L gnome_test.$L ubuntu_test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done

perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

for L in $src $tgt; do
    cp $tmp/bpe.gnome_test.$L $prep/gnome_test.$L
    cp $tmp/bpe.ubuntu_test.$L $prep/ubuntu_test.$L
done
