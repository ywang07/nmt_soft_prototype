# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch
import six

from . import data_utils
from . import LanguageTupleDataset


class LanguageTupleNoiseDataset(LanguageTupleDataset):
    """A pair of torch.utils.data.Datasets."""

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        src2=None, src2_sizes=None, src2_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True,
        shuffle_word=0, drop_word=0, add_word=0,
        is_training=False
    ):
        super().__init__(
            src, src_sizes, src_dict,
            tgt, tgt_sizes, tgt_dict,
            src2, src2_sizes, src2_dict,
            left_pad_source, left_pad_target,
            max_source_positions, max_target_positions,
            shuffle
        )

        self.is_training = is_training
        self.shuffle_word = shuffle_word
        self.drop_word = drop_word
        self.add_word = add_word

        self.src_bpe_end = None
        self.tgt_bpe_end = None
        self.index_bpe()

    def index_bpe(self):
        self.src_bpe_end = np.array(
            [not ww.endswith('@@') for ww, ii in six.iteritems(self.src_dict.indices)])
        self.tgt_bpe_end = np.array(
            [not ww.endswith('@@') for ww, ii in six.iteritems(self.tgt_dict.indices)])

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        return self.collate(samples)

    def collate(self, samples):
        if len(samples) == 0:
            return {}

        def merge(key, left_pad, move_eos_to_beginning=False):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                self.src_dict.pad(), self.src_dict.eos(), left_pad, move_eos_to_beginning,
            )

        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens = merge('source', left_pad=self.left_pad_source)
        # sort by descending source length
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

        sigma_y, len_y = None, None
        if samples[0].get('source2', None) is not None:
            src_tokens_2 = merge('source2', left_pad=self.left_pad_source)
            src_lengths_2 = torch.LongTensor([s['source2'].numel() for s in samples])
            src_tokens_2 = src_tokens_2.index_select(0, sort_order)
            src_lengths_2 = src_lengths_2.index_select(0, sort_order)

            assert samples[0].get('target', None) is not None
            target_left_pad = merge('target', left_pad=True, move_eos_to_beginning=False)
            target_lengths = torch.LongTensor([s['target'].numel() for s in samples])
            target_left_pad = target_left_pad.index_select(0, sort_order)
            target_lengths = target_lengths.index_select(0, sort_order)
            sigma_y, len_y = self._get_sigma_y(
                y=target_left_pad,
                x_hat=src_tokens_2,
                len_y=target_lengths,
                len_x_hat=src_lengths_2,
            )
            sigma_y = data_utils.collate_tokens(
                sigma_y,
                pad_idx=self.src_dict.pad(),
                eos_idx=self.src_dict.eos(),
                left_pad=self.left_pad_source,
                move_eos_to_beginning=False
            )

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

        # print("="*80)
        # self._print_tokens(src_tokens, name='x')
        # self._print_tokens(prev_output_tokens, name='y')
        # self._print_tokens(sigma_y, name='sigma_y')

        return {
            'id': id,
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
                'src_tokens_2': sigma_y,
                'src_lengths_2': len_y,
                'prev_output_tokens': prev_output_tokens,
            },
            'target': target,
        }

    def _get_sigma_y(self, y, x_hat, len_y, len_x_hat):
        """
        get sigma(y) as input to the second encoder
        :param y: [B, T], must be left padded with eos at the end, from true y
        :param x_hat: [B, T], must be left padded with eos at the end, from dictionary
        :param len_y: tensor, lengths of y
        :param len_x_hat: tensor, lengths of x_hat
        :return: sigma_y: list of tensors, len_y: tensor
        """
        # print("="*80)
        # self._print_tokens(x_hat, name='dict')
        # self._print_tokens(y, name='target')
        if self.is_training:
            if self.drop_word == 1 and self.add_word == 1:
                y, len_y = x_hat, len_x_hat
            else:
                y, len_y = self.word_drop(y, len_y, lang='tgt')
                # self._print_tokens(y, name='dropped')
                y, len_y = self.word_add(y, x_hat, len_y, len_x_hat, lang='tgt')
                # self._print_tokens(y, name='added')
            y, len_y = self.word_shuffle(y, len_y, lang='tgt')
            # self._print_tokens(y, name='shuffled')
        else:
            y, len_y = x_hat, len_x_hat

        sigma_y = []
        max_len = y.size(1)
        for i in range(len_y.size(0)):
            sigma_y.append(y[i, max_len - len_y[i]:])

        return sigma_y, len_y

    def word_shuffle(self, x, len_x, lang='tgt'):
        """
        shuffle words in x
        :param x: [B, T], must be left padded, eos at the end
        :param len_x: [B]
        :param lang: 'src' or 'tgt'
        :return: x2: [B, T] left padded, len_x: a tensor of lengths
        """
        if self.shuffle_word == 0:
            return x, len_x

        # define noise word scores [B, T-1]
        noise = np.random.uniform(0, self.shuffle_word, size=(x.size(0), x.size(1)-1))
        # noise[:, 0] = -1  # do not move start sentence symbol if any

        # word index: be sure to shuffle the entire word
        bpe_end = self.src_bpe_end[x] if lang == 'src' else self.tgt_bpe_end[x]
        word_idx = bpe_end[:, ::-1].cumsum(1)[:, ::-1]
        word_idx = word_idx.max(1)[:, None] - word_idx
        max_len = x.size(1)

        assert self.shuffle_word > 1
        x2 = x.clone()
        for i in range(len_x.size(0)):
            # generate a random permutation for each sentence
            scores = word_idx[i, max_len-len_x[i]:-1] + noise[i, word_idx[i, max_len-len_x[i]:-1]]
            scores += 1e-6 * np.arange(len_x[i] - 1)  # ensure no reordering inside a word
            permutation = scores.argsort()
            # shuffle sentence
            x2[i, max_len-len_x[i]:-1].copy_(x2[i, max_len-len_x[i]:-1][torch.from_numpy(permutation)])
        return x2, len_x

    def word_drop(self, x, len_x, lang='tgt'):
        """
        drop word in x
        :param x: [B, T], must be left padded, eos at the end
        :param len_x: [B]
        :param lang: 'src' or 'tgt'
        :return: x2: [B, T] left padded, len_x: a tensor of lengths
        """
        if self.drop_word == 0:
            return x, len_x

        # define words to drop
        keep = np.random.rand(x.size(0), x.size(1)-1) >= self.drop_word
        # keep[:, 0] = 1  # do not move start sentence symbol if any

        # word index: be sure to drop the entire word
        bpe_end = self.src_bpe_end[x] if lang == 'src' else self.tgt_bpe_end[x]
        word_idx = bpe_end[:, ::-1].cumsum(1)[:, ::-1]
        word_idx = word_idx.max(1)[:, None] - word_idx
        max_len = x.size(1)

        sentences = []
        lengths = []
        for i in range(len_x.size(0)):
            words = x[i, max_len-len_x[i]:-1].tolist()
            if len(words) > 2:
                # randomly drop words from the input
                new_s = [w for wid, w in enumerate(words) if keep[i, word_idx[i, wid]]]
                # at least one word other than eos left
                if len(new_s) == 0:
                    new_s.append(words[np.random.randint(0, len(words))])
            else:
                new_s = words
            new_s.append(self.src_dict.eos())  # add eos
            sentences.append(torch.LongTensor(new_s))
            lengths.append(len(new_s))

        x2 = data_utils.collate_tokens(
            sentences,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.src_dict.eos(),
            left_pad=True,
            move_eos_to_beginning=False,
        )
        len_x2 = torch.LongTensor(lengths)

        return x2, len_x2

    def word_add(self, x, x_hat, len_x, len_x_hat, lang='tgt'):
        """
        add word from x_hat to x
        :param x: [B, T], must be left padded, eos at the end
        :param x_hat: [B, T], must be left padded, eos at the end
        :param len_x: [B]
        :param len_x_hat: [B]
        :param lang: 'src' or 'tgt'
        :return: x2: [B, T] left padded, len_x: a tensor of lengths
        """
        if self.add_word == 0:
            return x, len_x

        # define words to add
        add = np.random.rand(x_hat.size(0), x_hat.size(1) - 1) <= self.add_word
        add[:, 0] = 1  # do not add start sentence symbol

        # word index: be sure to replace the entire word
        bpe_end_hat = self.src_bpe_end[x_hat] if lang == 'src' else self.tgt_bpe_end[x_hat]
        word_idx_hat = bpe_end_hat[:, ::-1].cumsum(1)[:, ::-1]
        word_idx_hat = word_idx_hat.max(1)[:, None] - word_idx_hat

        max_len = x.size(1)
        max_len_hat = x_hat.size(1)

        sentences = []
        lengths = []
        for i in range(len_x.size(0)):
            words = x[i, max_len-len_x[i]:-1].tolist()
            words_hat = x_hat[i, max_len_hat-len_x_hat[i]:-1].tolist()
            # randomly add words from x_hat
            words += [w for wid, w in enumerate(words_hat) if add[i, word_idx_hat[i, wid]]]
            words.append(self.src_dict.eos())  # add eos
            sentences.append(torch.LongTensor(words))
            lengths.append(len(words))

        x2 = data_utils.collate_tokens(
            sentences,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.src_dict.eos(),
            left_pad=True,
            move_eos_to_beginning=False,
        )
        len_x2 = torch.LongTensor(lengths)

        return x2, len_x2

    def _print_tokens(self, x, lang='tgt', name="y"):
        """
        for debugging
        """
        vocab = dict()
        for ww, ii in six.iteritems(self.tgt_dict.indices if lang == 'tgt' else self.src_dict.indices):
            vocab[ii] = ww

        print("-"*60)
        print(name)
        for i in range(x.size(0)):
            words = [vocab[widx] for widx in x[i].tolist() if widx != self.src_dict.pad()]
            words = " ".join(words)
            print("\t#{:03d}:\t{}".format(i, words))
