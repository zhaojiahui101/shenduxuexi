#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder self-attention layer definition."""

import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class EncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(
        self,
        size,
        self_attn,
        self_attn2,
        feed_forward,
        feed_forward2,
        feed_forward3,
        feed_forward_macaron,
        conv_module,
        conv_module2,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
        dual_ffn=False,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size)  # for the FNN module
        self.norm_mha = LayerNorm(size)  # for the MHA module
        self.dual_ffn = dual_ffn
        if self.dual_ffn:
            self.feed_forward_cn = feed_forward
            self.feed_forward_en = feed_forward2
            self.feed_forward_mix = feed_forward3

            # self.cn_linear = torch.nn.Linear(size, 2)
            # self.en_linear = torch.nn.Linear(size, 2)

            # self.linear_down = nn.Linear(size + size, size)
            # self.adapter = torch.nn.Sequential(
            #     torch.nn.Linear(size, 32),
            #     torch.nn.GELU(),
            #     torch.nn.Dropout(p=0.3),
            #     torch.nn.Linear(32, size),
            #     torch.nn.LayerNorm(size, eps=1e-5),
            #     torch.nn.Dropout(p=0.3),
            # )

            # self.linear_dense = nn.Linear(size + size, size)
            
            # self.self_attn2 = self_attn2
            # self.norm_mha2 = LayerNorm(size)
            # self.conv_module2 = conv_module2
            self.norm_ff2 = LayerNorm(size)
        else:
            self.feed_forward = feed_forward
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            if self.dual_ffn:
                self.norm_final_cn = LayerNorm(size)
                self.norm_final_en = LayerNorm(size)
                self.norm_final_mix = LayerNorm(size)
                # self.norm_conv2 = LayerNorm(size)
            else:
                self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
            if self.dual_ffn:
                self.concat_linear2 = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x_input, mask, ffn_out_input=None, cache=None):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, 1, time).

        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        ffn_out = ffn_out_input
        
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)
        
        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            if pos_emb is not None:
                return (x, pos_emb), mask, ffn_out
            return x, mask, ffn_out

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                self.feed_forward_macaron(x)
            )
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        # if self.dual_ffn:
        #     if self.normalize_before:
        #         x_cn = self.norm_mha(x)
        #         x_en = self.norm_mha2(x)
            
        #     if cache is None:
        #         x_q_cn = x_cn
        #         x_q_en = x_en
        #     else:
        #         assert cache.shape == (x_cn.shape[0], x_cn.shape[1] - 1, self.size)
        #         assert cache.shape == (x_en.shape[0], x_en.shape[1] - 1, self.size)
        #         x_q_cn = x_cn[:, -1:, :]
        #         x_q_en = x_en[:, -1:, :]
        #         residual = residual[:, -1:, :]
        #         mask = None if mask is None else mask[:, -1:, :]

        #     if pos_emb is not None:
        #         x_att_cn = self.self_attn(x_q_cn, x_cn, x_cn, pos_emb, mask)
        #         x_att_en = self.self_attn2(x_q_en, x_en, x_en, pos_emb, mask)
        #     else:
        #         x_att_cn = self.self_attn(x_q_cn, x_cn, x_cn, mask)
        #         x_att_en = self.self_attn2(x_q_en, x_en, x_en, mask)

        #     if self.concat_after:
        #         x_concat_cn = torch.cat((x_cn, x_att_cn), dim=-1)
        #         x_cn = residual + stoch_layer_coeff * self.concat_linear(x_concat_cn)
        #         x_concat_en = torch.cat((x_en, x_att_en), dim=-1)
        #         x_en = residual + stoch_layer_coeff * self.concat_linear2(x_concat_en)
        #     else:
        #         x_cn = residual + stoch_layer_coeff * self.dropout(x_att_cn)
        #         x_en = residual + stoch_layer_coeff * self.dropout(x_att_en)
        #     if not self.normalize_before:
        #         x_cn = self.norm_mha(x_cn)
        #         x_en = self.norm_mha2(x_en)
        #     ffn_out = (x_cn, x_en)
        # else:
        if self.normalize_before:
            x = self.norm_mha(x)
                    
        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)

        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        if self.conv_module is not None:
            # if self.dual_ffn:
            #     residual_cn = x_cn
            #     residual_en = x_en
            #     if self.normalize_before:
            #         x_cn = self.norm_conv(x_cn)
            #         x_en = self.norm_conv2(x_en)
            #     x_cn = residual_cn + stoch_layer_coeff * self.dropout(self.conv_module(x_cn))
            #     x_en = residual_en + stoch_layer_coeff * self.dropout(self.conv_module2(x_en))
            #     if not self.normalize_before:
            #         x_cn = self.norm_conv(x_cn)
            #         x_en = self.norm_conv2(x_en)
            # else:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + stoch_layer_coeff * self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        if self.dual_ffn:
            # residual_cn = x_cn
            # residual_en = x_en
            residual = x
            if self.normalize_before:
                x_cn = self.norm_ff(x)
                x_en = self.norm_ff2(x)
            x_cn = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                self.feed_forward_cn(x_cn)
            )
            x_en = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                self.feed_forward_en(x_en)
            )
            if not self.normalize_before:
                x_cn = self.norm_ff(x_cn)
                x_en = self.norm_ff2(x_en)
        else:
            residual = x
            if self.normalize_before:
                x = self.norm_ff(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                self.feed_forward(x)
            )
            if not self.normalize_before:
                x = self.norm_ff(x)

        if self.conv_module is not None:
            if self.dual_ffn:
                x_cn = self.norm_final_cn(x_cn)
                x_en = self.norm_final_en(x_en)
                #----------ffn-dis----------
                x = torch.cat((x_cn, x_en), dim=-1)
                x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(self.feed_forward_mix(x))
                x = self.norm_final_mix(x)
                ffn_out = (x_cn, x_en, x)
                # #----------expert----------
                # alpha = self.cn_linear(x_cn) + self.en_linear(x_en)
                # alpha = torch.nn.functional.softmax(alpha,dim=-1)
                # alpha_cn, alpha_en = torch.split(alpha,1,dim=-1)
                # x = alpha_cn * x_cn + alpha_en * x_en
                # #----------adapter----------
                # x = torch.cat((x_cn, x_en), dim=-1)
                # x = self.linear_down(x)
                # x = self.adapter(x)
                # #----------dense----------
                # x = torch.cat((x_cn, x_en), dim=-1)
                # x = self.linear_dense(x) + residual
            else:
                x = self.norm_final(x)
                # ffn_out = x

        if cache is not None:
            x = torch.cat([cache, x], dim=1)
        
        if pos_emb is not None:
            return (x, pos_emb), mask
        return x, mask
