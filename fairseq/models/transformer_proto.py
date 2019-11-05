# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils

from fairseq.modules import (
    LearnedPositionalEmbedding, MultiheadAttention,
    SinusoidalPositionalEmbedding,
)

from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqDecoder, FairseqModel,
    register_model, register_model_architecture,
)

from fairseq.models.transformer import TransformerEncoderLayer, TransformerDecoderLayer


@register_model('transformer_proto')
class TransformerProtoModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, FairseqDecoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, src_tokens_2=None, src_lengths_2=None):
        encoder_out = self.encoder(src_tokens, src_lengths, src_tokens_2, src_lengths_2)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder2-layers', type=int, metavar='N',
                            help='num encoder2 layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', default=False, action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', default=False, action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--encoder1-pos-emb', default="timing", type=str, metavar='STR',
                            help='whether to use positional embeddings in encoder_1 (not used if none)')
        parser.add_argument('--encoder2-pos-emb', default="none", type=str, metavar='STR',
                            help='whether to use positional embeddings in encoder_2 (not used if none)')
        parser.add_argument('--share-encoder-params', default=False, action='store_true',
                            help='share encoder and enc-dec attn parameters')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--proto-layers', type=str, default='all', metavar='STR',
                            help='all, or layer ids to add prototype, semicolon separated, e.g., 0;1;2;')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', default=False, action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', default=False, action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', default=False, action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', default=False, action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise RuntimeError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise RuntimeError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        add_pos_emb_1 = args.encoder1_pos_emb != "none"
        add_pos_emb_2 = args.encoder2_pos_emb != "none"
        encoder = TransformerProtoEncoder(
            args, src_dict, encoder_embed_tokens,
            add_pos_emb_1=add_pos_emb_1,
            add_pos_emd_2=add_pos_emb_2)
        decoder = TransformerProtoDecoder(args, tgt_dict, decoder_embed_tokens)
        return TransformerProtoModel(encoder, decoder)


class TransformerProtoEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens,
                 left_pad=True, add_pos_emb_1=True, add_pos_emd_2=False):
        super().__init__(dictionary)
        self.share_encoder_params = args.share_encoder_params

        print("| [model] building prototype encoder, shared = {}".format(self.share_encoder_params))
        self.encoder_1 = TransformerProtoSingleEncoder(
            args, dictionary, embed_tokens, left_pad, add_pos_emb_1)
        args2 = copy(args)
        args2.encoder_layers = args.encoder2_layers
        self.encoder_2 = TransformerProtoSingleEncoder(
            args2, dictionary, embed_tokens, left_pad, add_pos_emd_2) \
            if not self.share_encoder_params else None

    def forward(self, src_tokens, src_lengths, src_tokens_2=None, src_lengths_2=None):
        encoder_out_1 = self.encoder_1(src_tokens, src_lengths)
        encoder_out_2 = self.encoder_2(src_tokens_2, src_lengths_2) \
            if not self.share_encoder_params else self.encoder_1(src_tokens_2, src_lengths_2)
        return {
            'encoder_out_1': encoder_out_1['encoder_out'],
            'encoder_padding_mask_1': encoder_out_1['encoder_padding_mask'],
            'encoder_out_2': encoder_out_2['encoder_out'],
            'encoder_padding_mask_2': encoder_out_2['encoder_padding_mask'],
        }

    def reorder_encoder_out(self, encoder_out_dict, new_order):
        if encoder_out_dict['encoder_out_1'] is not None:
            encoder_out_dict['encoder_out_1'] = \
                encoder_out_dict['encoder_out_1'].index_select(1, new_order)
        if encoder_out_dict['encoder_out_2'] is not None:
            encoder_out_dict['encoder_out_2'] = \
                encoder_out_dict['encoder_out_2'].index_select(1, new_order)
        if encoder_out_dict['encoder_padding_mask_1'] is not None:
            encoder_out_dict['encoder_padding_mask_1'] = \
                encoder_out_dict['encoder_padding_mask_1'].index_select(0, new_order)
        if encoder_out_dict['encoder_padding_mask_2'] is not None:
            encoder_out_dict['encoder_padding_mask_2'] = \
                encoder_out_dict['encoder_padding_mask_2'].index_select(0, new_order)
        return encoder_out_dict

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.encoder_1.max_positions()

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.encoder_1.embed_positions, SinusoidalPositionalEmbedding):
            if 'encoder.encoder_1.embed_positions.weights' in state_dict:
                del state_dict['encoder.encoder_1.embed_positions.weights']
            if 'encoder.encoder_1.embed_positions._float_tensor' not in state_dict:
                state_dict['encoder.encoder_1.embed_positions._float_tensor'] = torch.FloatTensor()
        if not self.share_encoder_params and isinstance(
            self.encoder_2.embed_positions, SinusoidalPositionalEmbedding):
            if 'encoder.encoder_2.embed_positions.weights' in state_dict:
                del state_dict['encoder.encoder_2.embed_positions.weights']
            if 'encoder.encoder_2.embed_positions._float_tensor' not in state_dict:
                state_dict['encoder.encoder_2.embed_positions._float_tensor'] = torch.FloatTensor()
        return state_dict


class TransformerProtoSingleEncoder(FairseqEncoder):
    """Transformer encoder."""

    def __init__(self, args, dictionary, embed_tokens, left_pad=True, add_pos_emb=True):
        super().__init__(dictionary)
        print("| ---- [encoder] building encoder, layers = {}, add_pos_emb = {}".format(
            args.encoder_layers, add_pos_emb))

        self.dropout = args.dropout
        self.add_pos_emb = add_pos_emb

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            1024, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

    def forward(self, src_tokens, src_lengths):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.add_pos_emb:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out_dict, new_order):
        if encoder_out_dict.get('encoder_out') is not None:
            encoder_out_dict['encoder_out'] = \
                encoder_out_dict['encoder_out'].index_select(1, new_order)
        if encoder_out_dict.get('encoder_padding_mask') is not None:
            encoder_out_dict['encoder_padding_mask'] = \
                encoder_out_dict['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out_dict

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        return state_dict


class TransformerProtoDecoder(FairseqIncrementalDecoder):
    """Transformer decoder."""

    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        print("| [model] building prototype decoder, layers = {}, shared = {}".format(
            args.decoder_layers, args.share_encoder_params))

        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.share_encoder_params = args.share_encoder_params

        embed_dim = embed_tokens.embedding_dim
        padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            1024, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        )

        self.proto_layers = [int(i) for i in args.proto_layers.split(";")] \
            if args.proto_layers != "all" else [i for i in range(args.decoder_layers)]
        print("| ---- [decoder] adding prototype to layer: {}".format(args.proto_layers))

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerProtoDecoderLayer(args) if i in self.proto_layers
            else TransformerDecoderLayer(args)
            for i in range(args.decoder_layers)
        ])

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=embed_dim ** -0.5)

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        incremental_state_single = incremental_state
        if incremental_state is not None and self.share_encoder_params:
            incremental_state_single = incremental_state[0]

        # decoder layers
        for layer_id, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                (encoder_out['encoder_out_1'], encoder_out['encoder_out_2']),
                (encoder_out['encoder_padding_mask_1'], encoder_out['encoder_padding_mask_2']),
                incremental_state,
            ) if layer_id in self.proto_layers else layer(
                x,
                encoder_out['encoder_out_1'],
                encoder_out['encoder_padding_mask_1'],
                incremental_state_single,
            )

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # project back to size of vocabulary
        if self.share_input_output_embed:
            x = F.linear(x, self.embed_tokens.weight)
        else:
            x = F.linear(x, self.embed_out)

        return x, attn

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'decoder.embed_positions.weights' in state_dict:
                del state_dict['decoder.embed_positions.weights']
            if 'decoder.embed_positions._float_tensor' not in state_dict:
                state_dict['decoder.embed_positions._float_tensor'] = torch.FloatTensor()
        return state_dict

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder incremental state.

        This should be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        def apply_reorder_incremental_state(module):
            if module != self and hasattr(module, 'reorder_incremental_state'):
                if self.share_encoder_params and len(incremental_state) == 2:
                    module.reorder_incremental_state(
                        incremental_state[0],
                        new_order,
                    )
                    module.reorder_incremental_state(
                        incremental_state[1],
                        new_order,
                    )
                else:
                    module.reorder_incremental_state(
                        incremental_state,
                        new_order,
                    )
        self.apply(apply_reorder_incremental_state)


class TransformerProtoDecoderLayer(nn.Module):
    """Decoder layer block."""

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.share_encoder_params = args.share_encoder_params

        self.self_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before
        self.encoder_attn_1 = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.encoder_attn_2 = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout
        ) if not self.share_encoder_params else None
        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(3)])

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state):
        incremental_state_2 = None
        if incremental_state is not None and self.share_encoder_params:
            incremental_state, incremental_state_2 = incremental_state

        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask_future_timesteps=True,
            incremental_state=incremental_state,
            need_weights=False,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x1, attn1 = self.encoder_attn_1(
            query=x,
            key=encoder_out[0],
            value=encoder_out[0],
            key_padding_mask=encoder_padding_mask[0],
            incremental_state=incremental_state,
            static_kv=True,
        )
        x2, attn2 = self.encoder_attn_2(
            query=x,
            key=encoder_out[1],
            value=encoder_out[1],
            key_padding_mask=encoder_padding_mask[1],
            incremental_state=incremental_state,
            static_kv=True,
        ) if not self.share_encoder_params else self.encoder_attn_1(
            query=x,
            key=encoder_out[1],
            value=encoder_out[1],
            key_padding_mask=encoder_padding_mask[1],
            incremental_state=incremental_state_2,
            static_kv=True,
        )
        x = x1 + x2
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)

        residual = x
        x = self.maybe_layer_norm(2, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(2, x, after=True)
        return x, attn1

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings)
    return m


@register_model_architecture('transformer_proto', 'transformer_proto')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder2_layers = getattr(args, 'encoder2_layers', args.encoder_layers)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.proto_layers = getattr(args, 'proto_layers', 'all')
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder1_pos_emb = getattr(args, 'encoder1_pos_emb', "timing")  # add positional embedding
    args.encoder2_pos_emb = getattr(args, 'encoder2_pos_emb', "none")    # no positional embedding


@register_model_architecture('transformer_proto', 'transformer_proto_base_v1')
def transformer_proto_base_v1(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.dropout = getattr(args, 'dropout', 0.1)
    base_architecture(args)


@register_model_architecture('transformer_proto', 'transformer_proto_base_v2')
def transformer_proto_base_v2(args):
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.encoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    transformer_proto_big_v1(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani, et al, 2017)
@register_model_architecture('transformer_proto', 'transformer_proto_big_v1')
def transformer_proto_big_v1(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.3)
    base_architecture(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture('transformer_proto', 'transformer_proto_big_v2')
def transformer_proto_big_v2(args):
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.encoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    transformer_proto_big_v1(args)

