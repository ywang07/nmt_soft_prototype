# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter
import re

import torch


SPACE_NORMALIZER = re.compile("\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


class Tokenizer:

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize):
        with open(filename, 'r') as f:
            for line in f:
                for word in tokenize(line):
                    dict.add_symbol(word)
                dict.add_symbol(dict.eos_word)

    @staticmethod
    def binarize(filename, dict, consumer, tokenize=tokenize_line,
                 append_eos=True, reverse_order=False):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, 'r') as f:
            for line in f:
                ids = Tokenizer.tokenize(
                    line=line,
                    dict=dict,
                    tokenize=tokenize,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )
                nseq += 1

                consumer(ids)
                ntok += len(ids)
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': len(replaced)}

    @staticmethod
    def tokenize(line, dict, tokenize=tokenize_line, add_if_not_exist=True,
                 consumer=None, append_eos=True, reverse_order=False):
        words = tokenize(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = dict.add_symbol(word)
            else:
                idx = dict.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = dict.eos_index
        return ids


def tokenize_prob_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


class TokenizerProb:

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize):
        with open(filename, 'r') as f:
            for line in f:
                for word in tokenize(line):
                    dict.add_symbol(word)
                dict.add_symbol(dict.eos_word)

    @staticmethod
    def binarize(filename, dict, consumer, consumer_weights, soft_proto_dict, proto_k,
                 tokenize=tokenize_line, append_eos=True, reverse_order=False):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, 'r') as f:
            for line in f:
                ids, weights = TokenizerProb.tokenize_prob(
                    line=line,
                    dict=dict,
                    soft_proto_dict=soft_proto_dict,
                    proto_k=proto_k,
                    tokenize=tokenize,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )
                nseq += 1

                consumer(ids)
                consumer_weights(weights)

                ntok += ids.size(0)

        return {
            'nseq': nseq,
            'nunk': sum(replaced.values()),
            'ntok': ntok,
            'replaced': len(replaced)
        }

    @staticmethod
    def tokenize_prob(line, dict, soft_proto_dict, proto_k, tokenize=tokenize_line, add_if_not_exist=True,
                      consumer=None, append_eos=True, reverse_order=False):
        words = tokenize(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords, proto_k).fill_(dict.pad_index)  # T x k
        weights = torch.FloatTensor(nwords + 1 if append_eos else nwords, proto_k).zero_()

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = dict.add_symbol(word)
            else:
                idx = dict.index(word)
            if consumer is not None:
                consumer(word, idx)
            trans, probs = soft_proto_dict[word] if word in soft_proto_dict else (None, None)
            if trans is None:
                ids[i][0] = idx
                weights[i][0] = 1.
            else:
                trans = [dict.index(ww) for ww in trans]
                ids[i][:len(trans)] = torch.IntTensor(trans)
                weights[i][:len(trans)] = torch.Tensor(probs)
        if append_eos:
            ids[nwords][0] = dict.eos_index
            weights[nwords][0] = 1.
        ids = ids.view(-1, 1).squeeze()
        weights = weights.view(-1, 1).squeeze()
        return ids, weights

