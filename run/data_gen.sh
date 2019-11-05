#!/usr/bin/env bash

HOME_DIR=~/soft_prototype/

FS_DIR=$HOME_DIR/codes/fairseq_proto
export PYTHONPATH=$FS_DIR:${PYTHONPATH}

DATA_HOME_DIR=$HOME_DIR/data_fs/wmt_ende
WMT_ENDE_DATA_DIR=$DATA_HOME_DIR/lib/raw

GEN_WMT_DATA ()
{
    SRC=${1:-"en"}
    TRG=${2:-"de"}

    RAW_DATA_DIR=$DATA_HOME_DIR/lib/raw
    DATA_DIR=$DATA_HOME_DIR/data
    ORIG_DATA_DIR=$DATA_HOME_DIR/data

    mkdir -p $DATA_DIR
    cp $WMT_ENDE_DATA_DIR/dict.$SRC.txt $DATA_DIR
    cp $WMT_ENDE_DATA_DIR/dict.$TRG.txt $DATA_DIR

    python $FS_DIR/preprocess_proto.py \
        --source-lang "$SRC" \
        --target-lang "$TRG" \
        --trainpref "train" \
        --validpref "valid" \
        --testpref "test" \
        --srcdir $RAW_DATA_DIR \
        --destdir $DATA_DIR \
        --srcdict $DATA_DIR/dict.$SRC.txt \
        --tgtdict $DATA_DIR/dict.$TRG.txt \
        --output-format "binary"

    ls -lh $DATA_DIR/*
}

GEN_WMT_DICT_DATA ()
{
    SETTING=$1
    SRC=${2:-"en"}
    TRG=${3:-"de"}

    RAW_DATA_DIR=$DATA_HOME_DIR/lib/raw_${SETTING}
    DATA_DIR=$DATA_HOME_DIR/data_${SETTING}
    ORIG_DATA_DIR=$DATA_HOME_DIR/data

    mkdir -p $DATA_DIR
    cp $WMT_ENDE_DATA_DIR/dict.$SRC.txt $DATA_DIR
    cp $WMT_ENDE_DATA_DIR/dict.$TRG.txt $DATA_DIR

    python $FS_DIR/preprocess_proto.py \
        --source-lang "$SRC" \
        --target-lang "$TRG" \
        --source2-suffix "${SRC}2$TRG.dict" \
        --source2-lang "target" \
        --trainpref "train" \
        --validpref "valid" \
        --testpref "test" \
        --srcdir $RAW_DATA_DIR \
        --destdir $DATA_DIR \
        --srcdict $DATA_DIR/dict.$SRC.txt \
        --tgtdict $DATA_DIR/dict.$TRG.txt \
        --output-format "binary" \
        --skip-make-source \
        --skip-make-target
    ls -lh $DATA_DIR/*
}


# first build raw (x, y) data
# vocab can be pre-build with fairseq_unsup_proto/preprocess_build_dict.py
GEN_WMT_DATA "en" "de"

# build y_hat with the same vocab
# source2-lang: specifies whether use source/target vocab
GEN_WMT_DICT_DATA "soft_proto" "en" "de"