#!/usr/bin/env bash

TASK=translation_proto
ARCH=transformer_proto_big_v1
EXTRA_SETTING_STR=id000_ende_protobigv1_warm

HOME_DIR=~/soft_proto
TMP_DIR=/tmp

FS_DIR=$HOME_DIR/codes/fairseq_proto
export PYTHONPATH=$FS_DIR:$PYTHONPATH
F_MOSESDECODER=$HOME_DIR/mosesdecoder.tar.gz

DATA_DIR=$HOME_DIR/data_fs/wmt_ende/data
DATA_DICT_PATH=$HOME_DIR/data_fs/wmt_ende/data_proto
TRAIN_DIR=$HOME_DIR/model_fs/$EXTRA_SETTING_STR
LOG_DIR=$HOME_DIR/model_fs/logs
BLEU_DIR=$HOME_DIR/model_fs/ALL_BLEU

BEAM_SIZE=4
LPEN=0.6
BATCH_SIZE=64

TEST_LOG_DIR=$TRAIN_DIR/test-beam$BEAM_SIZE-lpen$LPEN
CKPT=checkpoint_best.pt
TEST_MODEL_DIR=$TMP_DIR/test/$CKPT-beam$BEAM_SIZE-lpen$LPEN

echo "======================= GPU & CUDA Version Checks ========================"
nvidia-smi
cat /usr/local/cuda/version.txt
nvcc -V

echo "===================== Python & PyTorch Version Checks ===================="
python3 -V
python3 -c 'import torch; print(torch.__version__)'

echo "============================= FairSeq Main ==============================="

mkdir -p $TEST_MODEL_DIR
cp $TRAIN_DIR/$CKPT $TEST_MODEL_DIR

python3 $FS_DIR/generate.py $DATA_DIR \
        --path $TEST_MODEL_DIR/$CKPT \
        --batch-size $BATCH_SIZE \
        --beam $BEAM_SIZE \
        --lenpen $LPEN \
        --task $TASK \
        --source2-path $DATA_DICT_PATH \
        --source2-suffix en2de.dict \
        --source2-domain target \
        --quiet \
        --remove-bpe \
        --no-progress-bar \
        2>&1 \
        | tee $TEST_LOG_DIR/test.$CKPT.txt



echo "$TRAIN_DIR/$CKPT: $(cut -d":" -f2 <<<$(grep "BLEU4" $TEST_LOG_DIR/test.$CKPT.txt))"
rm -r $TEST_MODEL_DIR