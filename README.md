# Neural Machine Translation with Soft Prototype

This repository is the implementation for paper of NeurIPS 2019: Neural Machine Translation with Soft Prototype.

The project is based on the [fairseq (version 0.5.0)](https://github.com/pytorch/fairseq/tree/v0.5.0).

### Requirements
* A [PyTorch installation (0.4.0)](http://pytorch.org/)
and install fairseq-0.5.0 with:
```
pip install -r ./requirements.txt
python ./setup.py build develop
```

### Training and Inference
The training and inference procedure is:

* Prepare data and train model with [standard procedure](https://github.com/pytorch/fairseq/tree/v0.6.0/examples/translation)

```
$ python train.py data-bin/wmt_ende \
        --arch transformer_big_v1 \
        --task translation \
        --share-all-embeddings \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --clip-norm 0.0 \
        --lr-scheduler inverse_sqrt \
        --warmup-init-lr 1e-07 \
        --warmup-updates 4000 \
        --lr 0.0005 \
        --min-lr 1e-09 \
        --weight-decay 0.0 \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --max-tokens 4096 \
        --update-freq 16 \
        --no-progress-bar
```

* Build prototype dictionary and training data
```
run/data_gen.sh
```

* Train model with prototype
```
run/train.sh
```

* Inference
```
run/test.sh
```