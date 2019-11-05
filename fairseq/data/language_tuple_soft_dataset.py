# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from . import data_utils, FairseqDataset


def collate_tokens_probs(values, scores, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    """Convert a list of 2d tensors into a padded 3d tensor."""
    size = max(v.size(0) for v in values)
    proto_k = values[0].size(1)
    res = values[0].new(len(values), size, proto_k).fill_(pad_idx)
    sco = scores[0].new(len(values), size, proto_k).fill_(0)

    def copy_token(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    def copy_score(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            dst[0] = 1. / proto_k
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_token(v, res[i][size - v.size(0):] if left_pad else res[i][:v.size(0)])
    for i, s in enumerate(scores):
        copy_score(s, sco[i][size - s.size(0):] if left_pad else sco[i][:s.size(0)])
    return res, sco


class LanguageTupleProbDataset(FairseqDataset):
    """A pair of torch.utils.data.Datasets."""

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        src2=None, src2_sizes=None, src2_dict=None,
        src2_score=None, proto_k=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src2 = src2
        self.src2_score = src2_score
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src2_sizes = np.array(src2_sizes) if src2_sizes is not None else None
        self.proto_k = proto_k
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.src2_dict = src2_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle

    def __getitem__(self, index):
        return {
            'id': index,
            'source': self.src[index],  # T
            'source2': self.src2[index].view(-1, self.proto_k) if self.src2 is not None else None,  # T * k
            'source2_score': self.src2_score[index].view(-1, self.proto_k) if self.src2_score is not None else None,  # T * k
            'target': self.tgt[index] if self.tgt is not None else None,  # T
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        return self.collate(samples)

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        max_source_positions, max_target_positions = self._get_max_positions(max_positions)
        src_len, tgt_len = min(src_len, max_source_positions), min(tgt_len, max_target_positions)
        bsz = num_tokens // max(src_len, tgt_len)
        return self.collater([
            {
                'id': i,
                'source': self.src_dict.dummy_sentence(src_len),
                'source2': self.src2_dict.dummy_sentence(src_len*self.proto_k).view(
                    src_len, self.proto_k) if self.src2_dict is not None else None,
                'source2_score': torch.Tensor(src_len, self.proto_k).fill_(1./self.proto_k),
                'target': self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_dict is not None else None,
            }
            for i in range(bsz)
        ])

    def collate(self, samples):
        if len(samples) == 0:
            return {}

        def merge(key, left_pad, move_eos_to_beginning=False):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                self.src_dict.pad(), self.src_dict.eos(),
                left_pad, move_eos_to_beginning,
            )

        # src: T
        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens = merge('source', left_pad=self.left_pad_source)
        # sort by descending source length
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

        # src2: T * k
        src_tokens_2, src_scores_2, src_lengths_2 = None, None, None
        if samples[0].get('source2', None) is not None:
            src_tokens_2, src_scores_2 = collate_tokens_probs(
                [s['source2'] for s in samples],
                [s['source2_score'] for s in samples],
                self.src_dict.pad(), self.src_dict.eos(), self.left_pad_source)
            src_lengths_2 = torch.LongTensor([s['source2'].size(0) for s in samples])
            src_tokens_2 = src_tokens_2.index_select(0, sort_order)
            src_scores_2 = src_scores_2.index_select(0, sort_order)
            src_lengths_2 = src_lengths_2.index_select(0, sort_order)

        # tgt
        prev_output_tokens = None
        target = None
        if samples[0].get('target', None) is not None:
            target = merge('target', left_pad=self.left_pad_target)
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=self.left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
            target = target.index_select(0, sort_order)
            ntokens = sum(len(s['target']) for s in samples)
        else:
            ntokens = sum(len(s['source']) for s in samples)

        # self._print(src_tokens, 'src')
        # self._print(src_tokens_2, 'src2')
        # self._print(src_scores_2, 'scores', False)
        return {
            'id': id,
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
                'src_tokens_2': src_tokens_2,
                'src_scores_2': src_scores_2,
                'src_lengths_2': src_lengths_2,
                'prev_output_tokens': prev_output_tokens,
            },
            'target': target,
        }

    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        return max(self.src_sizes[index],
                   self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
                   self.src2_sizes[index] if self.src2_sizes is not None else 0)

    def ordered_indices(self):
        """Ordered indices for batching."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    def valid_size(self, index, max_positions):
        """Check if an example's size is valid according to max_positions."""
        max_source_positions, max_target_positions = self._get_max_positions(max_positions)
        return (
            self.src_sizes[index] <= max_source_positions
            and (self.tgt_sizes is None or self.tgt_sizes[index] <= max_target_positions)
            and (self.src2_sizes is None or self.src2_sizes[index] <= max_target_positions)
        )

    def _get_max_positions(self, max_positions):
        if max_positions is None:
            return self.max_source_positions, self.max_target_positions
        assert len(max_positions) == 2
        max_src_pos, max_tgt_pos = max_positions
        return min(self.max_source_positions, max_src_pos), min(self.max_target_positions, max_tgt_pos)

    def _print(self, x, name, translate=True):
        print('\n' + '=' * 80)
        print(name)
        print(x)
        print('-' * 80)
        vocab = dict()
        for ww, ii in self.src_dict.indices.items():
            vocab[ii] = ww

        if not translate:
            return
        if x.dim() == 2:
            for i in range(x.size(0)):
                words = [vocab[widx] for widx in x[i].tolist() if widx != self.src_dict.pad()]
                words = " ".join(words)
                print("\t#{:03d}:\t{}".format(i, words))
        elif x.dim() == 3:
            for i in range(x.size(0)):
                words = ["(" + " ".join([vocab[widx] for widx in xx.tolist() if widx != self.src_dict.pad()]) + ")"
                         for xx in x[i]]
                words = " ".join(words)
                print("\t#{:03d}:\t{}".format(i, words))