# coding=utf-8
"""Script to merge values of variables in a list of checkpoint files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
from collections import OrderedDict

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_warm", default=None, metavar='PATH')
parser.add_argument("--checkpoint_warm_2", default=None, metavar='PATH')
parser.add_argument("--checkpoint_init", default=None, metavar='PATH')
parser.add_argument("--checkpoint_merged", default=None, metavar='PATH')
args = parser.parse_args()


def main():
    if not os.path.exists(args.checkpoint_warm):
        raise IOError("Model file not found: {}".format(args.checkpoint_warm))
    if not os.path.exists(args.checkpoint_init):
        raise IOError("Model file not found: {}".format(args.checkpoint_init))

    print("| loading checkpoint_warm from {}".format(args.checkpoint_warm))
    state_warm = torch.load(args.checkpoint_warm)
    state_warm_2 = None
    if args.checkpoint_warm_2 is not None and os.path.exists(args.checkpoint_warm_2):
        print("| loading checkpoint_warm_2 from {}".format(args.checkpoint_warm_2))
        state_warm_2 = torch.load(args.checkpoint_warm_2)
    print("| loading checkpoint_init from {}".format(args.checkpoint_init))
    state_init = torch.load(args.checkpoint_init)

    state_new = dict()
    for kk, vv in state_warm.items():
        if kk not in ["model", "args"]:
            state_new[kk] = vv
            continue
    state_new['args'] = state_init['args']

    model_warm = state_warm['model']
    model_warm_2 = state_warm_2['model'] if state_warm_2 is not None else None
    model_init = state_init['model']
    model_new = OrderedDict()
    for pname, param in model_init.items():

        # encoder params
        if pname.startswith("encoder.encoder_1."):
            look_name = "encoder." + pname[len("encoder.encoder_1."):]
            try:
                model_new[pname] = model_warm[look_name]
            except:
                print("| [WARNING] {} not in checkpoint_warm, copy {} from checkpoint_init".format(look_name, pname))
                model_new[pname] = param

        # encoder_attn
        elif re.match("decoder.layers.[0-9]*.encoder_attn_1.[a-z\_]*", pname) is not None:
            look_name = pname.replace("encoder_attn_1", "encoder_attn")
            try:
                model_new[pname] = model_warm[look_name]
            except:
                print("| [WARNING] {} not in checkpoint_warm, copy {} from checkpoint_init".format(look_name, pname))
                model_new[pname] = param

        # encoder_2 params
        elif model_warm_2 is not None and pname.startswith("encoder.encoder_2."):
            try:
                model_new[pname] = model_warm_2[pname]
            except:
                try:
                    look_name = pname.replace("encoder.encoder_2.", "encoder_2.")
                    model_new[pname] = model_warm_2[look_name]
                except:
                    print("| [WARNING] {} not in checkpoint_warm_2, copy from checkpoint_init".format(pname))
                    model_new[pname] = param

        # encoder_attn_2
        elif model_warm_2 is not None and re.match("decoder.layers.[0-9]*.encoder_attn_2.[a-z\_]*", pname) is not None:
            try:
                model_new[pname] = model_warm_2[pname]
            except:
                print("| [WARNING] {} not in checkpoint_warm_2, copy from checkpoint_init".format(pname))
                model_new[pname] = param

        elif pname in model_warm:
            try:
                model_new[pname] = model_warm[pname]
            except:
                print("| [WARNING] {} not in checkpoint_warm, copy from checkpoint_init".format(pname))
                model_new[pname] = param

        else:
            print("| [COPY] {}".format(pname))
            model_new[pname] = param
    state_new['model'] = model_new

    for pname, _ in state_new['model'].items():
        print(pname)

    torch.save(state_new, args.checkpoint_merged)
    print("| Model saved to file {}".format(args.checkpoint_merged))


if __name__ == "__main__":
    main()