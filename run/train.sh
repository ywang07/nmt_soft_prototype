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

echo "======================= GPU & CUDA Version Checks ========================"
nvidia-smi
cat /usr/local/cuda/version.txt
nvcc -V

echo "===================== Python & PyTorch Version Checks ===================="
python3 -V
python3 -c 'import torch; print(torch.__version__)'

echo "============================= FairSeq Main ==============================="

mkdir -p $TRAIN_DIR $LOG_DIR $BLEU_DIR

TRAIN_LOG_NAME=train.$EXTRA_SETTING_STR
echo "logging to $TRAIN_LOG_NAME.log"

if [ ! -f $TRAIN_DIR/init/checkpoint_last.pt ];
then
    echo -e "\n--------------------------------------------"
    echo "init params"
    mkdir -p $TRAIN_DIR/init
    python3 $FS_DIR/train.py $DATA_DIR \
        --save-dir $TRAIN_DIR/init \
        --arch $ARCH \
        --task $TASK \
        --share-all-embeddings \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --clip-norm 0.0 \
        --lr-scheduler inverse_sqrt \
        --warmup-init-lr 1e-07 \
        --warmup-updates 4000 \
        --lr 0.0005 \
        --min-lr 1e-09  \
        --weight-decay 0.0 \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --max-tokens 4096 \
        --save-interval-updates 500 \
        --warmup-updates 4000 \
        --lr 0.0005 \
        --proto-layers "all" \
        --encoder2-pos-emb "timing" \
        --source2-path $DATA_DICT_PATH \
        --source2-suffix en2de.dict \
        --source2-domain target \
        --update-freq 1 \
        --max-update 1 \
        --no-progress-bar \
        2>&1 \
        | tee -a $LOG_DIR/$TRAIN_LOG_NAME.log
fi

echo "ls -lh TRAIN_DIR/init"
ls -lh $TRAIN_DIR/init

if [ ! -f $TRAIN_DIR/checkpoint_last.pt ];
then
    echo -e "\n--------------------------------------------"
    echo "merge params"
    WARM_START_CHECKPOINT_1=$HOME_DIR/data_fs/wmt_ende_teacher/checkpoint_warm.pt
    WARM_START_CHECKPOINT_2=None
    python3 $FS_DIR/tools/merge_params.py \
        --checkpoint_warm   $WARM_START_CHECKPOINT_1 \
        --checkpoint_warm_2 $WARM_START_CHECKPOINT_2 \
        --checkpoint_init   $TRAIN_DIR/init/checkpoint_last.pt \
        --checkpoint_merged $TRAIN_DIR/checkpoint_start.pt \
        2>&1 \
        | tee -a $LOG_DIR/$TRAIN_LOG_NAME.log
    cp $TRAIN_DIR/checkpoint_start.pt $TRAIN_DIR/checkpoint_last.pt
    echo -e "--------------------------------------------\n"
fi

echo "ls -lh TRAIN_DIR"
ls -lh $TRAIN_DIR

python3 $FS_DIR/train.py $DATA_DIR \
        --save-dir $TRAIN_DIR \
        --arch $ARCH \
        --task $TASK \
        --share-all-embeddings \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --clip-norm 0.0 \
        --lr-scheduler inverse_sqrt \
        --warmup-init-lr 1e-07 \
        --min-lr 1e-09  \
        --weight-decay 0.0 \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --no-progress-bar \
        --update-freq 16 \
        --max-tokens 4096 \
        --save-interval-updates 2000 \
        --warmup-updates 4000 \
        --lr 0.0005 \
        --proto-layers "all" \
        --encoder2-pos-emb "timing" \
        --source2-path $DATA_DICT_PATH \
        --source2-suffix en2de.dict \
        --source2-domain target \
        --not-reload-opt \
        2>&1 \
        | tee -a $LOG_DIR/$TRAIN_LOG_NAME.log
