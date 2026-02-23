# Copyright 2026 the Tencent and HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from typing import Any, Callable, Optional, Tuple, Union
import numpy as np
import PIL.Image
import decord
import os
import mimetypes
import ffmpeg
import time

from ... import initialization as init
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import PreTrainedModel
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPooling
from ...generation import GenerationMixin
from ...processing_utils import ProcessorMixin, Unpack
from ...image_processing_utils import BaseImageProcessor
from ...video_processing_utils import BaseVideoProcessor
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...configuration_utils import PreTrainedConfig
from ...utils import logging
from ..youtu.configuration_youtu import YoutuConfig
from ..siglip2.configuration_siglip2 import Siglip2VisionConfig
# from ..siglip2.modeling_siglip2 import Siglip2VisionModel, Siglip2VisionTransformer
from ..youtu.modeling_youtu import (
    YoutuAttention,
    YoutuMLP,
    YoutuDecoderLayer,
    YoutuForCausalLM,
    YoutuModel,
    YoutuPreTrainedModel,
    YoutuRMSNorm,
    YoutuRotaryEmbedding,
)
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs, AudioKwargs, ImagesKwargs
from ...tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput
from ...utils import is_flash_attn_2_available
from ...video_utils import VideoInput

is_aiter_available = False

# if is_flash_attn_2_available():
#     try:
#         from aiter import flash_attn_varlen_func
#         is_aiter_available = True
#     except ImportError:
#         from flash_attn import flash_attn_varlen_func
# else:
#     flash_attn_varlen_func = None
from flash_attn import flash_attn_varlen_func
from flash_attn.layers.rotary import apply_rotary_emb


logger = logging.get_logger(__name__)



class YoutuVITAAudioConfig(PreTrainedConfig):
    r"""
    YoutuVITAAudioConfig
    """
    model_type = "youtu_vita_audio"
    base_config_key = "audio_config"

    def __init__(
        self,
        output_size=512,
        attention_heads=4,
        linear_units=2048,
        num_blocks=50,
        tp_blocks=20,
        dropout_rate=0.0,
        positional_dropout_rate=0.0,
        attention_dropout_rate=0.0,
        normalize_before=True,
        kernel_size=11,
        sanm_shfit=0,
        input_size=560,
        # vocab_size=25055,
        spatial_merge_size=1,
        out_hidden_size=4608,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.attention_heads = attention_heads
        self.linear_units = linear_units
        self.num_blocks = num_blocks
        self.tp_blocks = tp_blocks
        self.dropout_rate = dropout_rate
        self.positional_dropout_rate = positional_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.normalize_before = normalize_before
        self.kernel_size = kernel_size
        self.sanm_shfit = sanm_shfit

        self.hidden_size = output_size
        self.spatial_merge_size = spatial_merge_size
        self.out_hidden_size = out_hidden_size


class YoutuVITAVisionConfig(PreTrainedConfig):
    r"""
    YoutuVITAVisionConfig
    """
    model_type = "youtu_vita_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        num_patches=256,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        spatial_merge_size=2,
        out_hidden_size=4608,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.num_patches = num_patches
        self.spatial_merge_size = spatial_merge_size
        self.out_hidden_size = out_hidden_size


class YoutuVITATextConfig(YoutuConfig):
    r"""
    YoutuVITATextConfig
    """
    model_type = "youtu_vita_text"
    base_config_key = "text_config"
    pass


class YoutuVITAConfig(PreTrainedConfig):
    r"""
    YoutuVITAConfig
    """

    model_type = "youtu_vita"
    sub_configs = {"audio_config": YoutuVITAAudioConfig, "vision_config": YoutuVITAVisionConfig, "text_config": YoutuVITATextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(audio_config, dict):
            self.audio_config = self.sub_configs["audio_config"](**audio_config)
        elif audio_config is None:
            self.audio_config = self.sub_configs["audio_config"]()

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)



class YoutuVITAAudioSinusoidalPositionEncoder(torch.nn.Module):
    """ """

    def __int__(self, d_model=80, dropout_rate=0.1):
        pass

    def encode(
        self, positions: torch.Tensor = None, depth: int = None, dtype: torch.dtype = torch.float32
    ):
        batch_size = positions.size(0)
        positions = positions.type(dtype)
        device = positions.device
        log_timescale_increment = torch.log(torch.tensor([10000], dtype=dtype, device=device)) / (
            depth / 2 - 1
        )
        inv_timescales = torch.exp(
            torch.arange(depth / 2, device=device).type(dtype) * (-log_timescale_increment)
        )
        inv_timescales = torch.reshape(inv_timescales, [batch_size, -1])
        scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(
            inv_timescales, [1, 1, -1]
        )
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return encoding.type(dtype)

    def forward(self, x):
        batch_size, timesteps, input_dim = x.size()
        positions = torch.arange(1, timesteps + 1, device=x.device)[None, :]
        position_encoding = self.encode(positions, input_dim, x.dtype).to(x.device)

        return x + position_encoding


class YoutuVITAAudioPositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(YoutuVITAAudioPositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class YoutuVITAAudioMultiHeadedAttentionSANM(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self,
        n_head,
        in_feat,
        n_feat,
        dropout_rate,
        kernel_size,
        sanm_shfit=0,
        lora_list=None,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
    ):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        # self.linear_q = nn.Linear(n_feat, n_feat)
        # self.linear_k = nn.Linear(n_feat, n_feat)
        # self.linear_v = nn.Linear(n_feat, n_feat)

        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        # padding
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def forward_fsmn(self, inputs, mask, mask_shfit_chunk=None):
        b, t, d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            if mask_shfit_chunk is not None:
                mask = mask * mask_shfit_chunk
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x

    def forward_qkv(self, x):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        b, t, d = x.size()
        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        q_h = torch.reshape(q, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time1, d_k)
        k_h = torch.reshape(k, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)
        v_h = torch.reshape(v, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)

        return q_h, k_h, v_h, v

    def forward_attention(self, value, scores, mask, mask_att_chunk_encoder=None):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            if mask_att_chunk_encoder is not None:
                mask = mask * mask_att_chunk_encoder

            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)

            min_value = -float(
                "inf"
            )  # float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, x, mask, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q_h, k_h, v_h, v = self.forward_qkv(x)
        fsmn_memory = self.forward_fsmn(v, mask, mask_shfit_chunk)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask, mask_att_chunk_encoder)
        return att_outs + fsmn_memory

    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q_h, k_h, v_h, v = self.forward_qkv(x)
        if chunk_size is not None and look_back > 0 or look_back == -1:
            if cache is not None:
                k_h_stride = k_h[:, :, : -(chunk_size[2]), :]
                v_h_stride = v_h[:, :, : -(chunk_size[2]), :]
                k_h = torch.cat((cache["k"], k_h), dim=2)
                v_h = torch.cat((cache["v"], v_h), dim=2)

                cache["k"] = torch.cat((cache["k"], k_h_stride), dim=2)
                cache["v"] = torch.cat((cache["v"], v_h_stride), dim=2)
                if look_back != -1:
                    cache["k"] = cache["k"][:, :, -(look_back * chunk_size[1]) :, :]
                    cache["v"] = cache["v"][:, :, -(look_back * chunk_size[1]) :, :]
            else:
                cache_tmp = {
                    "k": k_h[:, :, : -(chunk_size[2]), :],
                    "v": v_h[:, :, : -(chunk_size[2]), :],
                }
                cache = cache_tmp
        fsmn_memory = self.forward_fsmn(v, None)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, None)
        return att_outs + fsmn_memory, cache


class YoutuVITAAudioLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


def sequence_mask(lengths, maxlen=None, dtype=torch.float32, device=None):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask = mask.detach()

    return mask.to(dtype).to(device) if device is not None else mask.to(dtype)
    # return mask.type(dtype).to(device) if device is not None else mask.type(dtype)


class YoutuVITAAudioEncoderLayerSANM(nn.Module):
    def __init__(
        self,
        in_size,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(YoutuVITAAudioEncoderLayerSANM, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = YoutuVITAAudioLayerNorm(in_size)
        self.norm2 = YoutuVITAAudioLayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.in_size = in_size
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate

    def forward(self, x, mask, cache=None, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
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
            return x, mask

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.concat_after:
            x_concat = torch.cat(
                (
                    x,
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    ),
                ),
                dim=-1,
            )
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
            else:
                x = stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
            else:
                x = stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder

    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.in_size == self.size:
            attn, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)
            x = residual + attn
        else:
            x, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm2(x)

        return x, cache


class YoutuVITAAudioEncoder(nn.Module):
    """
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        tp_blocks: int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        kernel_size: int = 11,
        sanm_shfit: int = 0,
    ):
        super().__init__()
        self._output_size = output_size

        self.embed = YoutuVITAAudioSinusoidalPositionEncoder()

        self.normalize_before = normalize_before

        positionwise_layer = YoutuVITAAudioPositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
        )

        encoder_selfattn_layer = YoutuVITAAudioMultiHeadedAttentionSANM
        encoder_selfattn_layer_args0 = (
            attention_heads,
            input_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )

        self.encoders0 = nn.ModuleList(
            [
                YoutuVITAAudioEncoderLayerSANM(
                    input_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args0),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(1)
            ]
        )
        self.encoders = nn.ModuleList(
            [
                YoutuVITAAudioEncoderLayerSANM(
                    output_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(num_blocks - 1)
            ]
        )

        self.tp_encoders = nn.ModuleList(
            [
                YoutuVITAAudioEncoderLayerSANM(
                    output_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(tp_blocks)
            ]
        )

        self.after_norm = YoutuVITAAudioLayerNorm(output_size)

        self.tp_norm = YoutuVITAAudioLayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ):
        """Embed positions in tensor."""
        masks = sequence_mask(ilens, dtype=torch.bfloat16, device=ilens.device)[:, None, :]
        # print(f"{masks=}")
        # print(f"{ilens=}")
        # print(f"{(masks>0.5).squeeze(1).sum(1).int()=}")

        xs_pad *= self.output_size() ** 0.5

        xs_pad = self.embed(xs_pad)

        # forward encoder1
        for layer_idx, encoder_layer in enumerate(self.encoders0):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        for layer_idx, encoder_layer in enumerate(self.encoders):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.after_norm(xs_pad)

        # forward encoder2
        # olens = masks.squeeze(1).sum(1).int()
        olens = (masks > 0.5).squeeze(1).sum(1).int()

        for layer_idx, encoder_layer in enumerate(self.tp_encoders):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.tp_norm(xs_pad)
        return xs_pad, olens


class YoutuVITAAudioSmall(nn.Module):
    """
    """

    def __init__(
        self,
        config,
        **kwargs,
    ):

        super().__init__()

        encoder = YoutuVITAAudioEncoder(
            input_size=config.input_size,
            output_size=config.output_size,
            attention_heads=config.attention_heads,
            linear_units=config.linear_units,
            num_blocks=config.num_blocks,
            tp_blocks=config.tp_blocks,
            dropout_rate=config.dropout_rate,
            positional_dropout_rate=config.positional_dropout_rate,
            attention_dropout_rate=config.attention_dropout_rate,
            normalize_before=config.normalize_before,
            kernel_size=config.kernel_size,
            sanm_shfit=config.sanm_shfit,
        )
        encoder_output_size = encoder.output_size()

        self.encoder = encoder

        self.encoder_output_size = encoder_output_size

        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        self.lid_int_dict = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        self.textnorm_int_dict = {25016: 14, 25017: 15}
        self.embed = torch.nn.Embedding(7 + len(self.lid_dict) + len(self.textnorm_dict), config.input_size)
        self.emo_dict = {"unk": 25009, "happy": 25001, "sad": 25002, "angry": 25003, "neutral": 25004}
        
    def forward(
        self,
        data_in,
        data_lengths=None,
        key: list = ["wav_file_tmp_name"],
        # **kwargs,
        language = "auto",
        use_itn = False,
        output_timestamp = False,
        textnorm = None,
    ):

        # fbank
        speech, speech_lengths = data_in, data_lengths
        if len(speech.shape) < 3:
            speech = speech[None, :, :]
        if speech_lengths is None:
            speech_lengths = speech.shape[1]

        # speech = speech.to(device=kwargs["device"])
        # speech_lengths = speech_lengths.to(device=kwargs["device"])
        speech = speech.to(device=self.embed.weight.data.device, dtype=self.embed.weight.data.dtype)
        speech_lengths = speech_lengths.to(device=self.embed.weight.data.device, dtype=self.embed.weight.data.dtype)

        # language = kwargs.get("language", "auto")
        language_query = self.embed(
            torch.LongTensor(
                [[self.lid_dict[language] if language in self.lid_dict else 0]]
            ).to(speech.device)
        ).repeat(speech.size(0), 1, 1)
        
        # use_itn = kwargs.get("use_itn", False)
        # output_timestamp = kwargs.get("output_timestamp", False)

        # textnorm = kwargs.get("text_norm", None)
        if textnorm is None:
            textnorm = "withitn" if use_itn else "woitn"
        textnorm_query = self.embed(
            torch.LongTensor([[self.textnorm_dict[textnorm]]]).to(speech.device)
        ).repeat(speech.size(0), 1, 1)
        speech = torch.cat((textnorm_query, speech), dim=1)
        speech_lengths += 1

        event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
            speech.size(0), 1, 1
        )
        input_query = torch.cat((language_query, event_emo_query), dim=1)
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths += 3

        # Encoder
        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, encoder_out_lens


class YoutuVITAAudioPatchMerger(nn.Module):
    def __init__(self, config: YoutuVITAAudioConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.norm = nn.RMSNorm(self.hidden_size)
        self.linear_fc1 = nn.Linear(self.hidden_size, config.out_hidden_size, bias=False)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(config.out_hidden_size, config.out_hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.reshape(x.shape[0], -1, x.shape[-1]))
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x

class YoutuVITAVisionPatchMerger(nn.Module):
    def __init__(self, config: YoutuVITAVisionConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.norm = nn.RMSNorm(self.hidden_size)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.reshape(-1, self.hidden_size))
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class YoutuVITATextRMSNorm(YoutuRMSNorm):
    pass


class YoutuVITATextRotaryEmbedding(YoutuRotaryEmbedding):
    pass


class YoutuVITATextMLP(YoutuMLP):
    pass


class YoutuVITATextAttention(YoutuAttention):
    pass


class YoutuVITATextDecoderLayer(YoutuDecoderLayer):
    pass

class YoutuVITAAudioPreTrainedModel(PreTrainedModel):
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    pass

class YoutuVITAVisionPreTrainedModel(PreTrainedModel):
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    pass


class YoutuVITAPreTrainedModel(PreTrainedModel):
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = getattr(self.config, "initializer_range", 0.02)
        embed_std = getattr(self.config, "embedding_initializer_range", 2 * std)
        if isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=embed_std)
            if module.padding_idx is not None:
                init.zeros_(module.weight.data[module.padding_idx])


class YoutuVITAAudioModel(YoutuVITAAudioPreTrainedModel):
    config: YoutuVITAAudioConfig

    def __init__(
        self,
        config: YoutuVITAAudioConfig,
        *inputs,
        **kwargs,
    ):
        super().__init__(config, *inputs, **kwargs)

        self.model = YoutuVITAAudioSmall(config)

        self.merger = YoutuVITAAudioPatchMerger(config)
    
    def forward(
        self,
        audios,
        # **kwargs,
    ):

        feats_pad = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=0.0)
        # feats_lens = torch.as_tensor([len(x) + 4 for x in audios])
        feats_lens = torch.as_tensor([len(x) for x in audios])

        # feats_pad = feats_pad.to(torch.bfloat16)

        encoder_out, encoder_out_lens = self.model(
            feats_pad,
            data_lengths=feats_lens,
            language="auto", # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=False,
            # ban_emo_unk=False,
            # **self.kwargs,
        )

        # encoder_out: bs seq hid
        # print(f"{encoder_out.size()=}")
        # print(f"{encoder_out_lens=}")
        encoder_out = encoder_out[:, 4:, :]
        encoder_out_lens = encoder_out_lens - 4
        # print(f"{encoder_out.size()=}")
        # print(f"{encoder_out_lens=}")

        assert encoder_out.shape[0] == len(audios)

        encoder_out = self.merger(encoder_out)

        return encoder_out, encoder_out_lens


class YoutuVITAVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        # inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        # self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.dim = dim
        self.theta = theta

    def forward(self, seqlen: torch.Tensor,) -> torch.Tensor:
        # seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # freqs = torch.outer(seq, self.inv_freq)

        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=seqlen.device) / self.dim))
        seq = torch.arange(seqlen, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.outer(seq, inv_freq)
        return freqs


class YoutuVITAVisionEmbeddings(nn.Module):
    def __init__(self, config: YoutuVITAVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )

    def forward(self, pixel_values: torch.FloatTensor, grid_thw: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.FloatTensor`):
                Pixel values of shape (batch_size, max_num_tokens, num_channels * patch_size * patch_size)
            grid_thw (`List[Tuple[int, int]]`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
        """

        # Apply patch embeddings to already patchified pixel values
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        return patch_embeds


def vision_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def vision_apply_rotary_pos_emb_flashatt(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
    k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def vision_apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class YoutuVITAVisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: YoutuVITAVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(seq_length, self.num_heads, self.head_dim)
        keys = keys.view(seq_length, self.num_heads, self.head_dim)
        values = values.view(seq_length, self.num_heads, self.head_dim)

        cos, sin = position_embeddings
        # print(f"{queries.size()=} {keys.size()=} {cos.size()=} {sin.size()=}")
        queries, keys = vision_apply_rotary_pos_emb(queries, keys, cos, sin)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        assert self.config._attn_implementation == "flash_attention_2"
        # print(f"{self.config._attn_implementation=}")

        queries = queries.transpose(0, 1).unsqueeze(0)
        keys = keys.transpose(0, 1).unsqueeze(0)
        values = values.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = vision_eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
        )

        attn_output = attn_output.reshape(seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

class YoutuVITAVisionFlashAttention2(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: YoutuVITAVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(seq_length, self.num_heads, self.head_dim)
        keys = keys.view(seq_length, self.num_heads, self.head_dim)
        values = values.view(seq_length, self.num_heads, self.head_dim)

        cos, sin = position_embeddings
        # print(f"{queries.size()=} {keys.size()=} {cos.size()=} {sin.size()=}")
        queries, keys = vision_apply_rotary_pos_emb_flashatt(queries.unsqueeze(0), keys.unsqueeze(0), cos, sin)
        queries = queries.squeeze(0)
        keys = keys.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        if is_aiter_available:
            attn_output = flash_attn_varlen_func(queries, keys, values, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, return_lse=True)[0].reshape(
                seq_length, -1
            )
        else:
            attn_output = flash_attn_varlen_func(queries, keys, values, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
                seq_length, -1
            )
        attn_output = self.out_proj(attn_output)
        return attn_output, None

class YoutuVITAVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class YoutuVITAVisionEncoderLayer(nn.Module):
    def __init__(self, config: YoutuVITAVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # self.self_attn = YoutuVITAVisionAttention(config)
        self.self_attn = YoutuVITAVisionFlashAttention2(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = YoutuVITAVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class YoutuVITAVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`YoutuVITAVisionEncoderLayer`].

    Args:
        config: YoutuVITAVisionConfig
    """

    def __init__(self, config: YoutuVITAVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([YoutuVITAVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        self.spatial_merge_size = 2
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_size = config.patch_size
        # self.window_size = self.patch_size * 2 * 8

        self.rotary_pos_emb = YoutuVITAVisionRotaryEmbedding(config.hidden_size // config.num_attention_heads // 2)

    def rot_pos_emb(self, grid_thw):
        pos_ids = []

        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        grid_thw,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutput:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    cu_seqlens,
                    position_embeddings,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                    cu_seqlens=cu_seqlens, 
                    position_embeddings=position_embeddings
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class YoutuVITAVisionModel(YoutuVITAVisionPreTrainedModel):
    _input_embed_layer = "patch_embedding"
    config: YoutuVITAVisionConfig

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = YoutuVITAVisionEmbeddings(config)
        self.encoder = YoutuVITAVisionEncoder(config)

        self.merger = YoutuVITAVisionPatchMerger(config)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thw: torch.LongTensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        hidden_states = self.embeddings(pixel_values, grid_thw)

        if attention_mask is not None and not self._use_flash_attention_2:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        else:
            encoder_attention_mask = attention_mask

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            grid_thw=grid_thw,
            attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        # last_hidden_state = self.post_layernorm(last_hidden_state)

        # pooler_output = self.head(last_hidden_state, attention_mask) if self.use_head else None
        pooler_output = None

        assert last_hidden_state.shape[0] == len(pixel_values)
        last_hidden_state = self.merger(last_hidden_state)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class YoutuVITATextModel(YoutuVITAPreTrainedModel, YoutuModel):
    config: YoutuVITATextConfig


class YoutuVITAModel(YoutuVITAPreTrainedModel):
    def __init__(self, config: YoutuVITAConfig):
        super().__init__(config)
        self.audio_model = YoutuVITAAudioModel._from_config(config.audio_config)
        self.vision_model = YoutuVITAVisionModel._from_config(config.vision_config)
        self.language_model = YoutuVITATextModel._from_config(config.text_config)

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        images: torch.FloatTensor | None = None,
        image_indices: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        audios: torch.FloatTensor | None = None,
        audio_indices: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:

        if images is not None:
            device = self.get_input_embeddings().weight.data.device
            dtype = self.get_input_embeddings().weight.data.dtype
            images = images.to(dtype).to(device)
            image_grid_thw = image_grid_thw.to(device)
            # print(f"{input_ids.size()=}")
            # print(f"{image_indices.size()=}")
            # print(f"{image_grid_thw.size()=}")
            # print(f"{images.size()=}")

            if len(image_grid_thw) > 64:
                image_embeds = []
                image_grid_thw = torch.split(image_grid_thw, 64, dim=0)
                chunk_num = len(image_grid_thw)
                im_st = 0
                for chunk_idx in range(chunk_num):
                    _image_grid_thw = image_grid_thw[chunk_idx]
                    im_ed = im_st + (_image_grid_thw[:, 1] * _image_grid_thw[:, 2] * _image_grid_thw[:, 0]).sum().item()
                    _image_embeds = self.vision_model(
                        pixel_values=images[im_st:im_ed],
                        attention_mask=None,
                        grid_thw=image_grid_thw[chunk_idx],
                    ).last_hidden_state
                    image_embeds.append(_image_embeds)

                    im_st = im_ed
                image_embeds = torch.cat(image_embeds, dim=0)
            else:
                image_embeds = self.vision_model(
                    pixel_values=images,
                    attention_mask=None,
                    grid_thw=image_grid_thw,
                ).last_hidden_state
            # print(f"image_embeds {image_embeds.size()}")
            # assert image_embeds.shape[0] == len(images)
            fake_images = None

            # image_embeds = image_embeds[:, 1:, :]
            # image_embeds = self.vision_projection(image_embeds)

            # torch.set_printoptions(threshold=100_000)
            # print(f"{image_embeds.size()=}")

        elif self.training:
            device = self.get_input_embeddings().weight.data.device
            dtype = self.get_input_embeddings().weight.data.dtype
            # fake_images = torch.ones((1 * self.config.vision["image_size"] // self.config.vision["patch_size"] * self.config.vision["image_size"] // self.config.vision["patch_size"], 3 * self.config.vision["patch_size"] * self.config.vision["patch_size"]), dtype=dtype, device=device)
            # image_grid_thw = torch.tensor([[1, self.config.vision["image_size"] // self.config.vision["patch_size"], self.config.vision["image_size"] // self.config.vision["patch_size"]]], dtype=torch.int64, device=device)
            fake_images = torch.ones((1 * 512 // self.vision_config.patch_size * 512 // self.vision_config.patch_size, 3 * self.vision_config.patch_size * self.vision_config.patch_size), dtype=dtype, device=device)
            image_grid_thw = torch.tensor([[1, 512 // self.vision_config.patch_size, 512 // self.vision_config.patch_size]], dtype=torch.int64, device=device)

            # print(f"{input_ids.size()=}")
            # print(f"{image_indices.size()=}")
            # print(f"{image_grid_thw.size()=}")
            # print(f"{fake_images.size()=}")

            image_embeds = self.vision_model(
                pixel_values=fake_images,
                attention_mask=None,
                grid_thw=image_grid_thw,
            ).last_hidden_state
            # image_embeds = image_embeds[:, 1:, :]
            # image_embeds = self.vision_projection(image_embeds)

            # torch.set_printoptions(threshold=100_000)
            # print(f"{image_embeds.size()=}")

        else:
            fake_images = None
            image_embeds = None

        if audios is not None:
            audio_embeds, audio_lengths = self.audio_model(audios)
            # if torch.distributed.get_rank() == 0:
            #     print(f"audio_embeds {audio_embeds.size()}")
            # assert audio_embeds.shape[0] == len(audios)
            fake_audios = None

            # audio_embeds = self.audio_projection(audio_embeds)

            # torch.set_printoptions(threshold=100_000)
            # if torch.distributed.get_rank() == 0:
            #     print(f"audio_embeds {audio_embeds.size()}")
            #     print(f"audio_embeds {audio_embeds.sum()}")
            #     print(f"audios {[x.size() for x in audios]}")
            #     print(f"audios {[x.sum() for x in audios]}")
            #     print(f"input_ids {input_ids.size()}")
            #     print(f"input_ids {input_ids.sum()}")
            #     # print(f"input_ids {input_ids}")
            #     print(f"audio_indices {[x.size() for x in audio_indices]}")
            #     print(f"audio_indices {[x.sum() for x in audio_indices]}")
            #     # print(f"audio_indices {audio_indices}")

        elif self.training:
            device = self.get_input_embeddings().weight.data.device
            dtype = self.get_input_embeddings().weight.data.dtype
            fake_audios = torch.ones((1, 1, 560), dtype=dtype, device=device)
            audio_embeds, audio_lengths = self.audio_model(fake_audios)
            # audio_embeds = self.audio_projection(audio_embeds)

        else:
            fake_audios = None
            audio_embeds = None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.language_model.embed_tokens(input_ids)
            input_ids = None

        if fake_images is not None:
            inputs_embeds = inputs_embeds + image_embeds.mean() * 0.0
        elif image_embeds is not None:
            inputs_embeds = inputs_embeds.clone()
            image_embeds = image_embeds.to(inputs_embeds.device)
            image_indices = image_indices.to(inputs_embeds.device)
            indices_b, indices_s = image_indices.unbind(dim=0)
            inputs_embeds[indices_b.view(-1), indices_s.view(-1)] = image_embeds.view(-1, image_embeds.shape[-1])
            # inputs_embeds = inputs_embeds + image_embeds.mean() * 0.0

        if fake_audios is not None:
            inputs_embeds = inputs_embeds + audio_embeds.mean() * 0.0
        elif audio_embeds is not None:
            inputs_embeds = inputs_embeds.clone()
            for audio_embeds_, audio_lengths_, audio_indices_ in zip(audio_embeds, audio_lengths, audio_indices,):
                # print(f"{audio_embeds_.size()=} {audio_lengths_=} {audio_indices_.size()=}")
                audio_embeds_ = audio_embeds_[:audio_lengths_, ...]
                audio_embeds_ = audio_embeds_.to(inputs_embeds.device)
                indices_b, indices_s = audio_indices_.to(inputs_embeds.device).unbind(dim=0)
                inputs_embeds[indices_b.view(-1), indices_s.view(-1)] = audio_embeds_.view(-1, audio_embeds_.shape[-1])
            # inputs_embeds = inputs_embeds + audio_embeds.mean() * 0.0

        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )



class YoutuVITAForCausalLM(YoutuVITAPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: YoutuVITAConfig):
        super().__init__(config)
        self.model = YoutuVITAModel(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        images: torch.FloatTensor | None = None,
        image_indices: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        audios: torch.FloatTensor | None = None,
        audio_indices: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, YoutuForCausalLM

        >>> model = YoutuForCausalLM.from_pretrained("meta-youtu/Youtu-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-youtu/Youtu-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_indices=image_indices,
            image_grid_thw=image_grid_thw,
            audios=audios,
            audio_indices=audio_indices,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        images: torch.FloatTensor | None = None,
        image_indices: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        audios: torch.FloatTensor | None = None,
        audio_indices: torch.LongTensor | None = None,
        is_first_iteration: bool | None = False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            use_cache=use_cache,
            images=images,
            image_indices=image_indices,
            image_grid_thw=image_grid_thw,
            audios=audios,
            audio_indices=audio_indices,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if not is_first_iteration and use_cache:
            model_inputs["images"] = None
            model_inputs["audios"] = None

        return model_inputs


from transformers.trainer_pt_utils import LabelSmoother
from dataclasses import dataclass, fields

@dataclass
class DEFAULT_TOKEN:

    IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
    IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
    IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
    OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

    # Model Constants
    IGNORE_INDEX = -100
    IMAGE_TOKEN_INDEX = -200
    # DEFAULT_IMAGE_TOKEN = IMG_CONTEXT_TOKEN
    # DEFAULT_IMAGE_PATCH_TOKEN = PATCH_CONTEXT_TOKEN
    # DEFAULT_IM_START_TOKEN = IMG_START_TOKEN
    # DEFAULT_IM_END_TOKEN = IMG_END_TOKEN

    IGNORE_TOKEN_ID = LabelSmoother.ignore_index

    def __init__(self):

        for field in fields(self):
            logger.info(f"♾️ {field.name} {getattr(self, field.name)}")
            print(f"♾️ {field.name} {getattr(self, field.name)}")


class UTU_VITA_TOKEN(DEFAULT_TOKEN):

    IM_START = "<|begin_of_text|>"
    IM_END = "<|end_of_text|>"
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

    THINK_START_TOKEN = "<think>"
    THINK_END_TOKEN = "</think>"
    CODE_START_TOKEN = "<code>"
    CODE_END_TOKEN = "</code>"
    ANSWER_START_TOKEN = "<answer>"
    ANSWER_END_TOKEN = "</answer>"

    TOOL_CALL_START_TOKEN = "<tool_call>"
    TOOL_CALL_END_TOKEN = "</tool_call>"
    TOOL_RESPONSE_START_TOKEN = "<tool_response>"
    TOOL_RESPONSE_END_TOKEN = "</tool_response>"

    IMG_TAG_TOKEN = "<|image|>"
    IMG_CONTEXT_TOKEN = "<|image_pad|>"
    IMG_START_TOKEN = "<|vision_start|>"
    IMG_END_TOKEN = "<|vision_end|>"

    VID_TAG_TOKEN = "<|video|>"
    VID_CONTEXT_TOKEN = "<|video_pad|>"
    VID_START_TOKEN = "<|video_start|>"
    VID_END_TOKEN = "<|video_end|>"

    AUD_TAG_TOKEN = "<|audio|>"
    AUD_CONTEXT_TOKEN = "<|audio_pad|>"
    AUD_START_TOKEN = "<|audio_start|>"
    AUD_END_TOKEN = "<|audio_end|>"

    POLY_START_TOKEN = "<poly>"
    POLY_END_TOKEN = "</poly>"
    INS_START_TOKEN = "<ins>"
    INS_END_TOKEN = "</ins>"
    CKPT_START_TOKEN = "<kpt>"
    CKPT_END_TOKEN = "</kpt>"
    BOX_START_TOKEN = "<box>"
    BOX_END_TOKEN = "</box>"
    REF_START_TOKEN = "<ref>"
    REF_END_TOKEN = "</ref>"
    FG_TOKEN = "<FG>"
    BG_TOKEN = "<BG>"
    OTHERS_TOKEN = "<OTHERS>"

    def __init__(self):
        logger.info(f"♾️ {self.__class__.__name__=}")
        print(f"♾️ {self.__class__.__name__=}")
        super().__init__()

        for i in range(2048):
            for axis in ["x", "y"]:
                setattr(self, f"{axis}_{i}_TOKEN", f"<{axis}_{i}>")

        for i in range(1, 1001):
            setattr(self, f"custom_{i}_TOKEN", f"<custom_{i}>")

    def get_special_tokens(self):
        return (
            [getattr(self, f"{axis}_{i}_TOKEN") for i in range(2048) for axis in ["x", "y"]]
            + [getattr(self, f"custom_{i}_TOKEN") for i in range(1, 1001)]
            + [
                self.POLY_START_TOKEN,
                self.POLY_END_TOKEN,
                self.INS_START_TOKEN,
                self.INS_END_TOKEN,
                self.CKPT_START_TOKEN,
                self.CKPT_END_TOKEN,
                self.BOX_START_TOKEN,
                self.BOX_END_TOKEN,
                self.REF_START_TOKEN,
                self.REF_END_TOKEN,
                self.FG_TOKEN,
                self.BG_TOKEN,
                self.OTHERS_TOKEN,
                self.THINK_START_TOKEN,
                self.THINK_END_TOKEN,
                self.CODE_START_TOKEN,
                self.CODE_END_TOKEN,
                self.ANSWER_START_TOKEN,
                self.ANSWER_END_TOKEN,
                self.TOOL_CALL_START_TOKEN,
                self.TOOL_CALL_END_TOKEN,
                self.TOOL_RESPONSE_START_TOKEN,
                self.TOOL_RESPONSE_END_TOKEN,
                self.IM_START,
                self.IM_END,
                self.IMG_TAG_TOKEN,
                self.IMG_CONTEXT_TOKEN,
                self.IMG_START_TOKEN,
                self.IMG_END_TOKEN,
                self.VID_TAG_TOKEN,
                self.VID_CONTEXT_TOKEN,
                self.VID_START_TOKEN,
                self.VID_END_TOKEN,
                self.AUD_TAG_TOKEN,
                self.AUD_CONTEXT_TOKEN,
                self.AUD_START_TOKEN,
                self.AUD_END_TOKEN,
            ]
        )

class UTU_VL_TOKEN(DEFAULT_TOKEN):

    IM_START = "<|begin_of_text|>"
    IM_END = "<|end_of_text|>"
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

    THINK_START_TOKEN = "<think>"
    THINK_END_TOKEN = "</think>"
    CODE_START_TOKEN = "<code>"
    CODE_END_TOKEN = "</code>"
    ANSWER_START_TOKEN = "<answer>"
    ANSWER_END_TOKEN = "</answer>"

    TOOL_CALL_START_TOKEN = "<tool_call>"
    TOOL_CALL_END_TOKEN = "</tool_call>"
    TOOL_RESPONSE_START_TOKEN = "<tool_response>"
    TOOL_RESPONSE_END_TOKEN = "</tool_response>"

    IMG_TAG_TOKEN = "<|image|>"
    IMG_CONTEXT_TOKEN = "<|image_pad|>"
    IMG_START_TOKEN = "<|vision_start|>"
    IMG_END_TOKEN = "<|vision_end|>"

    VID_TAG_TOKEN = "<|video|>"
    VID_CONTEXT_TOKEN = "<|video_pad|>"
    VID_START_TOKEN = "<|video_start|>"
    VID_END_TOKEN = "<|video_end|>"

    POLY_START_TOKEN = "<poly>"
    POLY_END_TOKEN = "</poly>"
    INS_START_TOKEN = "<ins>"
    INS_END_TOKEN = "</ins>"
    CKPT_START_TOKEN = "<kpt>"
    CKPT_END_TOKEN = "</kpt>"
    BOX_START_TOKEN = "<box>"
    BOX_END_TOKEN = "</box>"
    REF_START_TOKEN = "<ref>"
    REF_END_TOKEN = "</ref>"
    FG_TOKEN = "<FG>"
    BG_TOKEN = "<BG>"
    OTHERS_TOKEN = "<OTHERS>"

    def __init__(self):
        logger.info(f"♾️ {self.__class__.__name__=}")
        print(f"♾️ {self.__class__.__name__=}")
        super().__init__()

        for i in range(2048):
            for axis in ["x", "y"]:
                setattr(self, f"{axis}_{i}_TOKEN", f"<{axis}_{i}>")

        for i in range(1, 1001):
            setattr(self, f"custom_{i}_TOKEN", f"<custom_{i}>")

    def get_special_tokens(self):
        return (
            [getattr(self, f"{axis}_{i}_TOKEN") for i in range(2048) for axis in ["x", "y"]]
            + [getattr(self, f"custom_{i}_TOKEN") for i in range(1, 1001)]
            + [
                self.POLY_START_TOKEN,
                self.POLY_END_TOKEN,
                self.INS_START_TOKEN,
                self.INS_END_TOKEN,
                self.CKPT_START_TOKEN,
                self.CKPT_END_TOKEN,
                self.BOX_START_TOKEN,
                self.BOX_END_TOKEN,
                self.REF_START_TOKEN,
                self.REF_END_TOKEN,
                self.FG_TOKEN,
                self.BG_TOKEN,
                self.OTHERS_TOKEN,
                self.THINK_START_TOKEN,
                self.THINK_END_TOKEN,
                self.CODE_START_TOKEN,
                self.CODE_END_TOKEN,
                self.ANSWER_START_TOKEN,
                self.ANSWER_END_TOKEN,
                self.TOOL_CALL_START_TOKEN,
                self.TOOL_CALL_END_TOKEN,
                self.TOOL_RESPONSE_START_TOKEN,
                self.TOOL_RESPONSE_END_TOKEN,
                self.IM_START,
                self.IM_END,
                self.IMG_TAG_TOKEN,
                self.IMG_CONTEXT_TOKEN,
                self.IMG_START_TOKEN,
                self.IMG_END_TOKEN,
                self.VID_TAG_TOKEN,
                self.VID_CONTEXT_TOKEN,
                self.VID_START_TOKEN,
                self.VID_END_TOKEN,
            ]
        )


class VITA_TOKEN(DEFAULT_TOKEN):
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

    IMG_TAG_TOKEN = "<|image|>"
    IMG_CONTEXT_TOKEN = "<|context_of_image|>"
    IMG_START_TOKEN = "<|begin_of_image|>"
    IMG_END_TOKEN = "<|end_of_image|>"

    VID_TAG_TOKEN = "<|video|>"
    VID_CONTEXT_TOKEN = "<|context_of_video|>"
    VID_START_TOKEN = "<|begin_of_video|>"
    VID_END_TOKEN = "<|end_of_video|>"

    PATCH_CONTEXT_TOKEN = "<|context_of_patch|>"
    PATCH_START_TOKEN = "<|begin_of_patch|>"
    PATCH_END_TOKEN = "<|end_of_patch|>"

    AUD_TAG_TOKEN = "<|audio|>"
    AUD_CONTEXT_TOKEN = "<|context_of_audio|>"
    AUD_START_TOKEN = "<|begin_of_audio|>"
    AUD_END_TOKEN = "<|end_of_audio|>"

    QUAD_START_TOKEN = "<|begin_of_quad|>"
    QUAD_END_TOKEN = "<|end_of_quad|>"
    REF_START_TOKEN = "<|begin_of_ref|>"
    REF_END_TOKEN = "<|end_of_ref|>"
    BOX_START_TOKEN = "<|begin_of_box|>"
    BOX_END_TOKEN = "<|end_of_box|>"


    def __init__(self):
        logger.info(f"♾️ {self.__class__.__name__=}")
        print(f"♾️ {self.__class__.__name__=}")
        super().__init__()

    def get_special_tokens(self):
        return [
            self.IM_START,
            self.IM_END,
            # self.THINK_START_TOKEN,
            # self.THINK_END_TOKEN,
            # self.ANSWER_START_TOKEN,
            # self.ANSWER_END_TOKEN,
            self.IMG_START_TOKEN,
            self.IMG_END_TOKEN,
            self.IMG_CONTEXT_TOKEN,
            self.VID_START_TOKEN,
            self.VID_END_TOKEN,
            self.VID_CONTEXT_TOKEN,
            self.PATCH_START_TOKEN,
            self.PATCH_END_TOKEN,
            self.PATCH_CONTEXT_TOKEN,
            self.AUD_START_TOKEN,
            self.AUD_END_TOKEN,
            self.AUD_CONTEXT_TOKEN,
            self.QUAD_START_TOKEN,
            self.QUAD_END_TOKEN,
            self.REF_START_TOKEN,
            self.REF_END_TOKEN,
            self.BOX_START_TOKEN,
            self.BOX_END_TOKEN,
            self.IMG_TAG_TOKEN,
            self.VID_TAG_TOKEN,
            self.AUD_TAG_TOKEN,
        ]


# _GLOBAL_TOKEN = VITA_TOKEN()
# _GLOBAL_TOKEN = UTU_VL_TOKEN()
_GLOBAL_TOKEN = UTU_VITA_TOKEN()

# _GLOBAL_TOKEN = None


def get_token():
    _ensure_var_is_initialized(_GLOBAL_TOKEN, "token")
    return _GLOBAL_TOKEN


def set_token(args):
    global _GLOBAL_TOKEN

    if args.dataset_type == "vita":
        _GLOBAL_TOKEN = VITA_TOKEN()

    elif args.dataset_type == "utu_vl":
        _GLOBAL_TOKEN = UTU_VL_TOKEN()

    if args.dataset_type == "utu_vita" or args.dataset_type == "utu_vita_pretrain":
        _GLOBAL_TOKEN = UTU_VITA_TOKEN()

    else:
        raise NotImplementedError

    return _GLOBAL_TOKEN


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, "{} is not initialized.".format(name)


import uuid

from ..whisper.feature_extraction_whisper import WhisperFeatureExtractor

import torchaudio


def update_tokenizer_for_glm4voice(tokenizer):

    token_list = [f"<|audio_{i}|>" for i in range(16384)]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=False)

    # logger.info(f"tokenizer {tokenizer}")
    first_audio_token = token_list[0]
    last_audio_token = token_list[-1]
    first_audio_token_id = tokenizer.convert_tokens_to_ids(first_audio_token)
    last_audio_token_id = tokenizer.convert_tokens_to_ids(last_audio_token)
    logger.info(f"⚓ {first_audio_token=} {first_audio_token_id=}")
    logger.info(f"⚓ {last_audio_token=} {last_audio_token_id=}")

    tokenizer.first_audio_token = first_audio_token
    tokenizer.last_audio_token = last_audio_token
    tokenizer.first_audio_token_id = first_audio_token_id
    tokenizer.last_audio_token_id = last_audio_token_id

    return tokenizer


class GLM4VoiceTokenizer:

    sample_rate = 16000

    is_discrete = True
    is_contiguous = False

    _resample_buffer: dict[int, torchaudio.transforms.Resample] = {}

    tokenizer_type = "glm4voice"
    num_codebook = 1
    codebook_size = 16384

    first_audio_token = "<|audio_0|>"
    last_audio_token = "<|audio_16383|>"

    def __init__(self, model_name_or_path, flow_path=None, rank=None):
        self.model_name_or_path = model_name_or_path
        self.flow_path = flow_path

        # if rank is None and torch.distributed.is_initialized():
        #     rank = torch.distributed.get_rank()
        #     rank = rank % 8

        #     CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7,8')
        #     gpu_list = [int(x) for x in CUDA_VISIBLE_DEVICES.split(',')]
        #     if rank not in gpu_list:
        #         rank = None

        self.rank = rank
        logger.info(f"{self.rank=}")

    # @torch.compiler.disable
    def load_model(self):

        if not hasattr(self, "whisper_model") and self.model_name_or_path:
            pass
        elif not hasattr(self, "audio_decoder") and self.flow_path:
            pass
        else:
            return

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            self.rank = worker_id % 8
            logger.info(f"{self.rank=}")

        if self.rank is not None:
            self.device = f"cuda:{self.rank}"
            # torch.cuda.set_device(self.rank)
        else:
            self.device = "cuda"
            # self.device = "cpu"
        # self.device = "cpu"

        logger.info(f"{self.device=}")

        assert isinstance(self.model_name_or_path, str)

        if not hasattr(self, "whisper_model") and self.model_name_or_path:
            assert isinstance(self.model_name_or_path, str)

            from speech_tokenizer.modeling_whisper import WhisperVQEncoder
            logger.info(
                f"⏳ {self.device=} Loading {self.tokenizer_type} from {self.model_name_or_path}"
            )
            self.whisper_model = (
                WhisperVQEncoder.from_pretrained(self.model_name_or_path).eval().to(self.device)
            )
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                self.model_name_or_path
            )
            logger.info(f"⏳ {self.device=} Loading {self.tokenizer_type} Done")

        if not hasattr(self, "audio_decoder") and self.flow_path:
            assert isinstance(self.flow_path, str)

            from .flow_inference import AudioDecoder
            logger.info(f"⏳ {self.device=} Loading zai-org/glm-4-voice-decoder")
            flow_config = os.path.join(self.flow_path, "config.yaml")
            flow_checkpoint = os.path.join(self.flow_path, "flow.pt")
            hift_checkpoint = os.path.join(self.flow_path, "hift.pt")

            # Flow & Hift
            self.audio_decoder = AudioDecoder(
                config_path=flow_config,
                flow_ckpt_path=flow_checkpoint,
                hift_ckpt_path=hift_checkpoint,
                device=self.device,
            )
            logger.info(f"⏳ {self.device=} Loading zai-org/glm-4-voice-decoder Done")

    @torch.no_grad()
    # @torch.compiler.disable
    def encode(self, audio_or_path, **kwargs):

        if not hasattr(self, "whisper_model"):
            self.load_model()

        if True:
            audio_tokens = self.extract_speech_token(
                self.whisper_model, self.feature_extractor, [audio_or_path], device=self.device
            )[0]
            return audio_tokens

        return None

    @torch.no_grad()
    # @torch.compiler.disable
    def decode(self, audio_tokens, option_steps=10, **kwargs):
        if not hasattr(self, "whisper_model"):
            self.load_model()

        this_uuid = str(uuid.uuid4())
        this_uuid = "abc"

        tts_token = torch.tensor(audio_tokens, device=self.device).unsqueeze(0)

        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(self.device)
        prompt_speech_feat = torch.zeros(1, 0, 80).to(self.device)

        tts_speech, tts_mel = self.audio_decoder.token2wav(
            tts_token,
            uuid=this_uuid,
            prompt_token=flow_prompt_speech_token.to(self.device),
            prompt_feat=prompt_speech_feat.to(self.device),
            finalize=True,
            option_steps=option_steps,
        )
        tts_speechs = []
        tts_speechs.append(tts_speech.squeeze())
        tts_speech = torch.cat(tts_speechs, dim=-1).cpu()

        return tts_speech

    # @torch.compiler.disable
    def extract_speech_token(self, model, feature_extractor, utts, device="cuda"):
        with torch.no_grad():
            audios, indices = [], []
            for idx, utt in enumerate(utts):
                if isinstance(utt, tuple):
                    audio, sample_rate = utt
                else:
                    audio, sample_rate = torchaudio.load(utt)
                # audio = audio.to(device)
                if sample_rate != 16000:
                    if sample_rate not in self._resample_buffer:
                        self._resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                            orig_freq=sample_rate, new_freq=16000
                        ) #.to(device)
                    # self._resample_buffer[sample_rate].to(device)
                    audio = self._resample_buffer[sample_rate](audio)
                # if audio.shape[0] > 1:
                #     audio = audio[:1]
                # audio = audio[0]
                if audio.dim() == 2:
                    audio = audio.mean(0)
                audio = audio.cpu().numpy()
                time_step = 0
                while time_step * 16000 < audio.shape[0]:
                    audio_segment = audio[time_step * 16000 : (time_step + 30) * 16000]
                    audios.append(audio_segment)
                    indices.append(idx)
                    time_step += 30

            if len(audios) == 0:
                return [0]

            pooling_kernel_size = model.config.pooling_kernel_size or 1
            stride = (
                model.conv1.stride[0]
                * model.conv2.stride[0]
                * pooling_kernel_size
                * feature_extractor.hop_length
            )
            all_speech_tokens = [[] for _ in range(len(utts))]
            batch_size = 32
            for start in range(0, len(audios), batch_size):
                features = feature_extractor(
                    audios[start : start + batch_size],
                    sampling_rate=16000,
                    return_attention_mask=True,
                    return_tensors="pt",
                    device=device,
                    padding="longest",
                    pad_to_multiple_of=stride,
                )
                features = features.to(device=device)
                outputs = model(**features)
                speech_tokens = outputs.quantized_token_ids
                attention_mask = features.attention_mask[
                    :, :: model.conv1.stride[0] * model.conv2.stride[0]
                ]
                attention_mask = attention_mask[:, :: model.config.pooling_kernel_size]
                assert attention_mask.shape == speech_tokens.shape
                for i in range(len(speech_tokens)):
                    idx = indices[start + i]
                    speech_token = speech_tokens[i][attention_mask[i].bool()].tolist()
                    all_speech_tokens[idx].extend(speech_token)

            return all_speech_tokens





import torchaudio
from funasr.utils.load_utils import extract_fbank
from funasr.frontends.wav_frontend import WavFrontend


def update_tokenizer_for_wav_frontend(tokenizer):

    return tokenizer


class WavFrontendTokenizer:
    def __init__(self,):

        self.sample_rate = 16000

        self.is_discrete = True
        self.is_contiguous = True

        self._resample_buffer: dict[int, torchaudio.transforms.Resample] = {}

        self.tokenizer_type = "wav_frontend"

    def load_model(self):
        if hasattr(self, "frontend"):
            return

        self.device = "cpu"

        logger.info(
            f"⏳ {self.device=} Loading {self.tokenizer_type}"
        )
        self.frontend = WavFrontend(
            fs=16000,
            window="hamming",
            n_mels=80,
            frame_length=25,
            frame_shift=10,
            lfr_m=7,
            lfr_n=6,
            cmvn_file=None,
        )
        logger.info(f"⏳ {self.device=} Loading {self.tokenizer_type} Done")

    @torch.no_grad()
    def encode(self, audio_or_path, **kwargs):

        if not hasattr(self, "frontend"):
            self.load_model()

        if isinstance(audio_or_path, tuple):
            audio, sample_rate = audio_or_path
        else:
            audio, sample_rate = torchaudio.load(audio_or_path)
        # print(f"{audio.size()=} {sample_rate=}")
        if audio.dim() == 2:
            audio = audio.mean(0)

        if sample_rate != self.sample_rate:
            if sample_rate not in self._resample_buffer:
                # print(f"torchaudio.transforms.Resample {sample_rate=} {self.sample_rate=} {self.device=}", flush=True)
                self._resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.sample_rate
                ).to(self.device)
            audio = audio.to(self.device)
            self._resample_buffer[sample_rate].to(self.device)
            audio = self._resample_buffer[sample_rate](audio[None, :])[0, :]
            audio = audio.cpu()
        # resampler = torchaudio.transforms.Resample(
        #     orig_freq=sample_rate, new_freq=self.sample_rate
        # )
        # audio = resampler(audio[None, :])[0, :]
        # audio = audio.to(self.device)

        speech, speech_lengths = extract_fbank(audio, data_type="sound", frontend=self.frontend)

        speech = speech[0]
        # print(f"{audio.size()=}")
        # print(f"{speech_lengths=}")
        # print(f"{speech.size()=}")

        # if max(speech_lengths) > 512:
        #     raise Exception(f"Audio is to long {speech_lengths}")

        speech = torch.nan_to_num(speech, nan=0, posinf=0, neginf=0)

        return {
            "audio": speech,
            "audio_token_length_func": len,
            "duration_seconds": len(audio) / self.sample_rate,
        }

    @torch.no_grad()
    def decode(self, audio_tokens, **kwargs):
        return None


class VisionTokenizer:
    def __init__(self, tokenizer_contiguous=None, tokenizer_discrete=None):

        self.tokenizer_contiguous = tokenizer_contiguous
        self.tokenizer_discrete = tokenizer_discrete

        # if tokenizer_contiguous is not None:
        #     self.is_contiguous = True
        # else:
        #     self.is_contiguous = False
        self.is_contiguous = True

        if tokenizer_discrete is not None:
            self.is_discrete = True
        else:
            self.is_discrete = False

        logger.info(f"{self.first_vision_token=}")
        logger.info(f"{self.last_vision_token=}")

    @property
    def first_vision_token(self):
        if self.tokenizer_discrete is not None:
            return self.tokenizer_discrete.first_vision_token
        else:
            return None

    @property
    def last_vision_token(self):
        if self.tokenizer_discrete is not None:
            return self.tokenizer_discrete.last_vision_token
        else:
            return None

    def load_model(self):
        if hasattr(self.tokenizer_contiguous, "load_model") and callable(
            getattr(self.tokenizer_contiguous, "load_model")
        ):
            self.tokenizer_contiguous.load_model()

        if hasattr(self.tokenizer_discrete, "load_model") and callable(
            getattr(self.tokenizer_discrete, "load_model")
        ):
            self.tokenizer_discrete.load_model()

    @torch.no_grad()
    def encode(self, audio_or_path, is_discrete=False, is_contiguous=True, **kwargs):
        if is_discrete and self.tokenizer_discrete is not None:
            return self.tokenizer_discrete.encode(audio_or_path, **kwargs)

        if is_contiguous and self.tokenizer_contiguous is not None:
            return self.tokenizer_contiguous.encode(audio_or_path, **kwargs)

        return None

    @torch.no_grad()
    def decode(self, audio_tokens, **kwargs):

        if self.tokenizer_discrete is not None:
            return self.tokenizer_discrete.decode(audio_tokens, **kwargs)

        return None


class AudioTokenizer:
    def __init__(self, tokenizer_contiguous=None, tokenizer_discrete=None):

        self.tokenizer_contiguous = tokenizer_contiguous
        self.tokenizer_discrete = tokenizer_discrete

        if tokenizer_contiguous is not None:
            self.is_contiguous = True
        else:
            self.is_contiguous = False

        if tokenizer_discrete is not None:
            self.is_discrete = True
        else:
            self.is_discrete = False

        logger.info(f"{self.first_audio_token=}")
        logger.info(f"{self.last_audio_token=}")

    @property
    def first_audio_token(self):
        if self.tokenizer_discrete is not None:
            return self.tokenizer_discrete.first_audio_token
        else:
            return None

    @property
    def last_audio_token(self):
        if self.tokenizer_discrete is not None:
            return self.tokenizer_discrete.last_audio_token
        else:
            return None

    @property
    def num_codebook(self):
        if self.tokenizer_discrete is not None:
            return self.tokenizer_discrete.num_codebook
        else:
            return 0

    def load_model(self):
        if hasattr(self.tokenizer_contiguous, "load_model") and callable(
            getattr(self.tokenizer_contiguous, "load_model")
        ):
            self.tokenizer_contiguous.load_model()

        if hasattr(self.tokenizer_discrete, "load_model") and callable(
            getattr(self.tokenizer_discrete, "load_model")
        ):
            self.tokenizer_discrete.load_model()

    @torch.no_grad()
    def encode(self, audio_or_path, is_discrete=False, is_contiguous=True, **kwargs):
        if is_discrete and self.tokenizer_discrete is not None:
            return self.tokenizer_discrete.encode(audio_or_path, **kwargs)

        if is_contiguous and self.tokenizer_contiguous is not None:
            return self.tokenizer_contiguous.encode(audio_or_path, **kwargs)

        return None

    @torch.no_grad()
    def decode(self, audio_tokens, **kwargs):

        if self.tokenizer_discrete is not None:
            return self.tokenizer_discrete.decode(audio_tokens, **kwargs)

        return None


def update_tokenizer(tokenizer, audio_tokenizer_type_list=None, vision_tokenizer_type_list=None):
    tokenizer_type = type(tokenizer).__name__
    logger.info(
        f"🧱 start update_tokenizer {tokenizer_type} {len(tokenizer)=} {audio_tokenizer_type_list=} {vision_tokenizer_type_list=}"
    )

    # if tokenizer_type in ["PreTrainedTokenizer", "PreTrainedTokenizerFast"]:
    #     token_list = [
    #         "<|im_start|>",
    #         "<|im_end|>",
    #     ]
    #     num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    #     for token in token_list:
    #         token_id = tokenizer.convert_tokens_to_ids(token)
    #         logger.info(f"➕ {token=} {token_id=}")

    token_list = get_token().get_special_tokens()
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    for token in token_list[:10]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        logger.info(f"➕ {token=} {token_id=}")

    for token in token_list[-10:]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        logger.info(f"➕ {token=} {token_id=}")

    if audio_tokenizer_type_list is None:
        audio_tokenizer_type_list = []
    if vision_tokenizer_type_list is None:
        vision_tokenizer_type_list = []

    for audio_tokenizer_type in audio_tokenizer_type_list:
        if audio_tokenizer_type == "glm4voice":
            # from .glm4voice import update_tokenizer_for_glm4voice, GLM4VoiceTokenizer

            tokenizer = update_tokenizer_for_glm4voice(tokenizer)

        elif audio_tokenizer_type == "cosyvoice2":
            # from .cosyvoice2 import update_tokenizer_for_cosyvoice2, CosyVoice2Tokenizer

            tokenizer = update_tokenizer_for_cosyvoice2(tokenizer)

        elif audio_tokenizer_type == "snac24khz":
            # from .snac import update_tokenizer_for_snac, SNACTokenizer

            tokenizer = update_tokenizer_for_snac(tokenizer)

        elif audio_tokenizer_type == "sparktts":
            # from .sparktts import (
            #     update_tokenizer_for_sparktts,
            #     SparkTTSTokenizer,
            # )

            tokenizer = update_tokenizer_for_sparktts(tokenizer)

        elif audio_tokenizer_type == "sensevoice":
            # from .sensevoice import (
            #     update_tokenizer_for_sensevoice,
            #     SenseVoiceTokenizer,
            # )

            tokenizer = update_tokenizer_for_sensevoice(tokenizer)

        elif audio_tokenizer_type == "melfilterbank":
            # from .melfilterbank import (
            #     update_tokenizer_for_melfilterbank,
            #     MelFilterBankTokenizer,
            # )

            tokenizer = update_tokenizer_for_melfilterbank(tokenizer)

        elif audio_tokenizer_type == "xytokenizer":
            # from .xy import (
            #     update_tokenizer_for_xytokenizer,
            #     XYTokenizer,
            # )

            tokenizer = update_tokenizer_for_xytokenizer(tokenizer)

        else:
            print(f"{audio_tokenizer_type_list=}")
            raise NotImplementedError

    for vision_tokenizer_type in vision_tokenizer_type_list:

        if vision_tokenizer_type == "pixel":
            pass

        elif vision_tokenizer_type == "chameleon":
            # from .chameleon import (
            #     update_tokenizer_for_chameleon,
            #     ChameleonTokenizer,
            # )

            tokenizer = update_tokenizer_for_chameleon(tokenizer)

        elif vision_tokenizer_type == "emu35":
            # from .emu35 import (
            #     update_tokenizer_for_emu35,
            #     Emu35Tokenizer,
            # )

            tokenizer = update_tokenizer_for_emu35(tokenizer)

        else:
            print(f"{vision_tokenizer_type_list=}")
            raise NotImplementedError

    logger.info(
        f"🧱 finish update_tokenizer {len(tokenizer)=} {audio_tokenizer_type_list=} {vision_tokenizer_type_list=}"
    )
    return tokenizer


def get_audio_tokenizer(
    model_name_or_path_list, audio_tokenizer_type_list, flow_path=None, rank=None
):

    if audio_tokenizer_type_list is None:
        audio_tokenizer_type_list = []
        model_name_or_path_list = []

    tokenizer_contiguous = None
    tokenizer_discrete = None
    for audio_tokenizer_type, model_name_or_path in zip(
        audio_tokenizer_type_list, model_name_or_path_list
    ):

        if audio_tokenizer_type == "glm4voice":
            # from .glm4voice import update_tokenizer_for_glm4voice, GLM4VoiceTokenizer

            tokenizer_discrete = GLM4VoiceTokenizer(
                model_name_or_path, flow_path=flow_path, rank=rank
            )

        elif audio_tokenizer_type == "cosyvoice2":
            # from .cosyvoice2 import update_tokenizer_for_cosyvoice2, CosyVoice2Tokenizer

            tokenizer_discrete = CosyVoice2Tokenizer(model_name_or_path, rank=rank)

        elif audio_tokenizer_type == "snac24khz":
            # from .snac import update_tokenizer_for_snac, SNACTokenizer

            tokenizer_discrete = SNACTokenizer(model_name_or_path, rank=rank)

        elif audio_tokenizer_type == "sparktts":
            # from .sparktts import (
            #     update_tokenizer_for_sparktts,
            #     SparkTTSTokenizer,
            # )

            tokenizer_discrete = SparkTTSTokenizer(model_name_or_path, rank=rank)

        elif audio_tokenizer_type == "sensevoice":
            # from .sensevoice import update_tokenizer_for_sensevoice, SenseVoiceTokenizer

            tokenizer_contiguous = SenseVoiceTokenizer(model_name_or_path, rank=rank)

        elif audio_tokenizer_type == "wav_frontend":
            # from .wav_frontend import update_tokenizer_for_wav_frontend, WavFrontendTokenizer

            tokenizer_contiguous = WavFrontendTokenizer()

        elif audio_tokenizer_type == "melfilterbank":
            # from .melfilterbank import (
            #     update_tokenizer_for_melfilterbank,
            #     MelFilterBankTokenizer,
            # )

            tokenizer_contiguous = MelFilterBankTokenizer(model_name_or_path, rank=rank)

        elif audio_tokenizer_type == "xytokenizer":
            # from .xy import (
            #     update_tokenizer_for_xytokenizer,
            #     XYTokenizer,
            # )

            tokenizer_discrete = XYTokenizer(model_name_or_path, rank=rank)

        else:
            print(f"{audio_tokenizer_type_list=}")
            raise NotImplementedError

    audio_tokenizer = AudioTokenizer(tokenizer_contiguous, tokenizer_discrete)

    return audio_tokenizer


def get_vision_tokenizer(model_name_or_path_list, vision_tokenizer_type_list, rank=None):

    if vision_tokenizer_type_list is None:
        vision_tokenizer_type_list = []
        model_name_or_path_list = []

    tokenizer_contiguous = None
    tokenizer_discrete = None
    for vision_tokenizer_type, model_name_or_path in zip(
        vision_tokenizer_type_list, model_name_or_path_list
    ):
        if vision_tokenizer_type == "pixel":
            tokenizer_contiguous = True

        elif vision_tokenizer_type == "chameleon":
            # from .chameleon import update_tokenizer_for_chameleon, ChameleonTokenizer

            tokenizer_discrete = ChameleonTokenizer(model_name_or_path, rank=rank)

        elif vision_tokenizer_type == "emu35":
            # from .emu35 import update_tokenizer_for_emu35, Emu35Tokenizer

            tokenizer_discrete = Emu35Tokenizer(model_name_or_path, rank=rank)

        else:
            print(f"{vision_tokenizer_type_list=}")
            raise NotImplementedError

    vision_tokenizer = VisionTokenizer(tokenizer_contiguous, tokenizer_discrete)
    return vision_tokenizer


class YoutuVITAImagesKwargs(ImagesKwargs, total=False):
    """
    """
    discrete_image_idxs: list
    contiguous_image_idxs: list


class YoutuVITAAudioKwargs(AudioKwargs, total=False):
    """
    """
    discrete_audio_idxs: list
    contiguous_audio_idxs: list

    # audio_tokenizer_type: str
    # audio_tokenizer_path: str

class YoutuVITAVideosKwargs(VideosKwargs, total=False):
    """
    """
    resolution_type: str
    min_num_tokens: int
    max_num_tokens: int
    image_min_num_tokens: int
    image_max_num_tokens: int
    temporal_patch_size: int
    spatial_merge_size: int
    patch_size: int
    video_key_frame: bool
    use_audio_in_video: bool
    use_vision_in_video: bool


class YoutuVITAProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: YoutuVITAImagesKwargs
    videos_kwargs: YoutuVITAVideosKwargs
    audio_kwargs: YoutuVITAAudioKwargs

    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
        "images_kwargs": {
        },
        "videos_kwargs": {
            "resolution_type": "native",
            "min_num_tokens": 64,
            "max_num_tokens": 8192,
            "image_min_num_tokens": 4,
            "image_max_num_tokens": 256,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
            "patch_size":16,
            "video_key_frame": False,
            "use_audio_in_video": True,
            "use_vision_in_video": True,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": "max_length",
            "return_attention_mask": True,
        },
    }


class YoutuVITAFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["pixel_values", "image_grid_thw"]
    valid_kwargs = YoutuVITAAudioKwargs

    def __init__(
        self,
        audio_tokenizer_path=None,
        audio_tokenizer_type=None,
        flow_path=None,
        rank=None,
        text_audio_interval_ratio=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.audio_tokenizer = get_audio_tokenizer(
            audio_tokenizer_path,
            audio_tokenizer_type,
            flow_path=flow_path,
            rank=rank,
        )

        self.text_audio_interval_ratio = text_audio_interval_ratio

        # self.load_model()

    def to_dict(self):
        output = super().to_dict()
        # Remove the non-serializable object before returning
        if "audio_tokenizer" in output:
            del output["audio_tokenizer"]
        return output


    def load_model(self):
        self.audio_tokenizer.load_model()

    def process_audio(self, audio_or_path, is_discrete=False, is_contiguous=False, **kwargs):

        assert not (is_discrete and is_contiguous)
        assert is_discrete or is_contiguous

        if is_discrete and self.audio_tokenizer.tokenizer_discrete is not None:
            if isinstance(audio_or_path, str):
                tokenizer_type = self.audio_tokenizer.tokenizer_discrete.tokenizer_type
                cache_path = os.path.splitext(audio_or_path)[0] + f"-{tokenizer_type}.json"

                audio_data = load_cache(cache_path)
                if audio_data is not None:
                    return audio_data

            audio_data = self.audio_tokenizer.encode(
                audio_or_path, is_discrete=is_discrete, is_contiguous=is_contiguous, **kwargs
            )
            # print(f"{len(audio_data)=}")

            if isinstance(audio_or_path, str):
                save_cache(audio_data, cache_path)

            return audio_data

        if is_contiguous:
            audio_dict = self.audio_tokenizer.encode(
                audio_or_path, is_discrete=is_discrete, is_contiguous=is_contiguous, **kwargs
            )
            return audio_dict

    def process_token_to_audio(self, audio_tokens):
        audio_data = self.audio_tokenizer.decode(audio_tokens)
        return audio_data

    @property
    def is_discrete(self):
        return self.audio_tokenizer.is_discrete

    @property
    def is_contiguous(self):
        return self.audio_tokenizer.is_contiguous

    def text_audio_interval(self, content_input_id, AUD_START_ID, AUD_END_ID):

        return text_audio_interval(
            content_input_id,
            AUD_START_ID,
            AUD_END_ID,
            self.text_audio_interval_ratio,
        )

    def add_audio_input_contiguous(
        self, input_ids, audio_or_paths, tokenizer, targets=None, is_pretrain=False,
        **kwargs,
    ):
        GLOBAL_TOKEN = get_token()

        AUD_CONTEXT_ID = tokenizer(
            GLOBAL_TOKEN.AUD_CONTEXT_TOKEN, add_special_tokens=False
        ).input_ids
        AUD_TAG_ID = tokenizer(GLOBAL_TOKEN.AUD_TAG_TOKEN, add_special_tokens=False).input_ids
        AUD_START_ID = tokenizer(GLOBAL_TOKEN.AUD_START_TOKEN, add_special_tokens=False).input_ids
        AUD_END_ID = tokenizer(GLOBAL_TOKEN.AUD_END_TOKEN, add_special_tokens=False).input_ids

        assert len(AUD_CONTEXT_ID) == 1
        assert len(AUD_START_ID) == 1
        assert len(AUD_END_ID) == 1

        AUD_CONTEXT_ID = AUD_CONTEXT_ID[0]
        AUD_TAG_ID = AUD_TAG_ID[0]
        AUD_START_ID = AUD_START_ID[0]
        AUD_END_ID = AUD_END_ID[0]

        aud_positions = [i for i, x in enumerate(input_ids) if x == AUD_TAG_ID]
        assert len(aud_positions) == len(
            audio_or_paths
        ), f"{len(aud_positions)=} {len(audio_or_paths)=} {AUD_TAG_ID=}"

        audios = []
        audio_indices = []
        new_input_ids = []
        new_targets = []
        st = 0
        for aud_idx, aud_pos in enumerate(aud_positions):
            # audio = audio_tokenizer.encode(audio_or_paths[aud_idx], is_contiguous=True)
            audio, audio_token_length_func = self.process_audio(
                audio_or_paths[aud_idx], is_contiguous=True
            )
            audios.append(audio)
            # audio_token_length = audio.size(0)
            # audio_token_length = audio_token_length_func(audio.size(0))
            audio_token_length = audio_token_length_func(audio)

            new_input_ids += input_ids[st:aud_pos]
            if targets is not None:
                new_targets += targets[st:aud_pos]

            new_input_ids += [AUD_START_ID]
            if targets is not None:
                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            audio_indice_b = torch.zeros(
                1, audio_token_length, dtype=torch.int64
            )  # This will change in collate_fn
            audio_indice_s = (
                torch.arange(len(new_input_ids), len(new_input_ids) + audio_token_length)
                .unsqueeze(0)
                .repeat(1, 1)
            )
            audio_indice_b_s = torch.stack(
                [audio_indice_b, audio_indice_s], dim=0
            )  # 2, num_audio, audio_length
            audio_indices.append(audio_indice_b_s)

            new_input_ids += [AUD_CONTEXT_ID] * audio_token_length
            if targets is not None:
                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * audio_token_length

            new_input_ids += [AUD_END_ID]
            if targets is not None:
                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            st = aud_pos + 1

        new_input_ids += input_ids[st:]
        if targets is not None:
            new_targets += targets[st:]

        input_ids = new_input_ids
        if targets is not None:
            targets = new_targets

        if targets is not None:
            return input_ids, audios, audio_indices, targets

        return input_ids, audios, audio_indices

    def add_audio_input_discrete_and_contiguous(
        self, input_ids, audio_or_paths, tokenizer, targets=None, is_pretrain=False,
        **kwargs,
    ):

        GLOBAL_TOKEN = get_token()

        AUD_CONTEXT_ID = tokenizer(
            GLOBAL_TOKEN.AUD_CONTEXT_TOKEN, add_special_tokens=False
        ).input_ids
        AUD_TAG_ID = tokenizer(GLOBAL_TOKEN.AUD_TAG_TOKEN, add_special_tokens=False).input_ids
        AUD_START_ID = tokenizer(GLOBAL_TOKEN.AUD_START_TOKEN, add_special_tokens=False).input_ids
        AUD_END_ID = tokenizer(GLOBAL_TOKEN.AUD_END_TOKEN, add_special_tokens=False).input_ids

        AUD_FIRST_ID = tokenizer.convert_tokens_to_ids("<|audio_0|>")

        assert len(AUD_CONTEXT_ID) == 1
        assert len(AUD_START_ID) == 1
        assert len(AUD_END_ID) == 1

        AUD_CONTEXT_ID = AUD_CONTEXT_ID[0]
        AUD_TAG_ID = AUD_TAG_ID[0]
        AUD_START_ID = AUD_START_ID[0]
        AUD_END_ID = AUD_END_ID[0]

        aud_positions = [i for i, x in enumerate(input_ids) if x == AUD_TAG_ID]
        assert len(aud_positions) == len(
            audio_or_paths
        ), f"{len(aud_positions)=} {len(audio_or_paths)=} {AUD_TAG_ID=}"

        audios = []
        audio_indices = []
        new_input_ids = []
        new_targets = []
        st = 0
        for aud_idx, aud_pos in enumerate(aud_positions):

            new_input_ids += input_ids[st:aud_pos]
            if targets is not None:
                new_targets += targets[st:aud_pos]

            # --------------------------------------------------------------------------
            # add discrete

            audio_tokens = self.process_audio(audio_or_paths[aud_idx], is_discrete=True)
            audio_tokens = [i + AUD_FIRST_ID for i in audio_tokens]

            new_input_ids += [AUD_START_ID]
            if targets is not None:
                if is_pretrain:
                    new_targets += [AUD_START_ID]
                else:
                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            new_input_ids += audio_tokens
            if targets is not None:
                if is_pretrain:
                    new_targets += audio_tokens
                else:
                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(audio_tokens)

            new_input_ids += [AUD_END_ID]
            if targets is not None:
                if is_pretrain:
                    new_targets += [AUD_END_ID]
                else:
                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            # --------------------------------------------------------------------------
            # add contiguous

            # audio = audio_tokenizer.encode(audio_or_paths[aud_idx], is_contiguous=True)
            audio, audio_token_length_func = self.process_audio(
                audio_or_paths[aud_idx], is_contiguous=True
            )
            audios.append(audio)
            # audio_token_length = audio.size(0)
            # audio_token_length = audio_token_length_func(audio.size(0))
            audio_token_length = audio_token_length_func(audio)

            new_input_ids += [AUD_START_ID]
            if targets is not None:
                if is_pretrain:
                    new_targets += [AUD_START_ID]
                else:
                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            audio_indice_b = torch.zeros(
                1, audio_token_length, dtype=torch.int64
            )  # This will change in collate_fn
            audio_indice_s = (
                torch.arange(len(new_input_ids), len(new_input_ids) + audio_token_length)
                .unsqueeze(0)
                .repeat(1, 1)
            )
            audio_indice_b_s = torch.stack(
                [audio_indice_b, audio_indice_s], dim=0
            )  # 2, num_audio, audio_length
            audio_indices.append(audio_indice_b_s)

            new_input_ids += [AUD_CONTEXT_ID] * audio_token_length
            if targets is not None:
                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * audio_token_length

            new_input_ids += [AUD_END_ID]
            if targets is not None:
                if is_pretrain:
                    new_targets += [AUD_END_ID]
                else:
                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            st = aud_pos + 1

            # if max(audio_token_length) > 512:
            #     raise Exception(f"Audio is to long {speech_lengths}")

        new_input_ids += input_ids[st:]
        if targets is not None:
            new_targets += targets[st:]

        input_ids = new_input_ids
        if targets is not None:
            targets = new_targets

        if targets is not None:
            return input_ids, audios, audio_indices, targets

        return input_ids, audios, audio_indices

    def add_audio_input_discrete_or_contiguous(
        self,
        input_ids,
        audio_or_paths,
        tokenizer,
        discrete_audio_idxs=[],
        contiguous_audio_idxs=[],
        targets=None,
        is_pretrain=False,
        audio_chunk_min_second=30,
        audio_chunk_max_second=30,
        **kwargs,
    ):

        GLOBAL_TOKEN = get_token()

        AUD_CONTEXT_ID = tokenizer(
            GLOBAL_TOKEN.AUD_CONTEXT_TOKEN, add_special_tokens=False
        ).input_ids
        AUD_TAG_ID = tokenizer(GLOBAL_TOKEN.AUD_TAG_TOKEN, add_special_tokens=False).input_ids
        AUD_START_ID = tokenizer(GLOBAL_TOKEN.AUD_START_TOKEN, add_special_tokens=False).input_ids
        AUD_END_ID = tokenizer(GLOBAL_TOKEN.AUD_END_TOKEN, add_special_tokens=False).input_ids

        if self.audio_tokenizer.tokenizer_discrete is not None:
            AUD_FIRST_ID = tokenizer.convert_tokens_to_ids(
                self.audio_tokenizer.tokenizer_discrete.first_audio_token
            )

        assert len(AUD_CONTEXT_ID) == 1
        assert len(AUD_START_ID) == 1
        assert len(AUD_END_ID) == 1

        AUD_CONTEXT_ID = AUD_CONTEXT_ID[0]
        AUD_TAG_ID = AUD_TAG_ID[0]
        AUD_START_ID = AUD_START_ID[0]
        AUD_END_ID = AUD_END_ID[0]

        aud_positions = [i for i, x in enumerate(input_ids) if x == AUD_TAG_ID]
        assert len(aud_positions) == len(
            audio_or_paths
        ), f"{len(aud_positions)=} {len(audio_or_paths)=} {AUD_TAG_ID=}"

        if not discrete_audio_idxs and not contiguous_audio_idxs:
            contiguous_audio_idxs = list(range(len(audio_or_paths)))

        audios = []
        audio_indices = []
        new_input_ids = []
        new_targets = []
        if targets is not None and self.audio_tokenizer.num_codebook > 1:
            # additional_targets_list = [[] for _ in range(self.audio_tokenizer.num_codebook - 1)]
            additional_targets_list = [[] for _ in range(self.audio_tokenizer.num_codebook)]
        else:
            additional_targets_list = None

        st = 0
        for aud_idx, aud_pos in enumerate(aud_positions):

            new_input_ids += input_ids[st:aud_pos]
            if targets is not None:
                new_targets += targets[st:aud_pos]
            if additional_targets_list is not None:
                additional_targets_list = [
                    x + [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * (aud_pos - st)
                    for x in additional_targets_list
                ]

            # --------------------------------------------------------------------------
            # add discrete
            if aud_idx in discrete_audio_idxs:

                audio_tokens = self.process_audio(audio_or_paths[aud_idx], is_discrete=True)
                if self.audio_tokenizer.num_codebook > 1:
                    # assert audio_tokens.ndim == 2
                    # audio_tokens = audio_tokens.tolist()

                    # L7 | P70 P71 P72 P73 P74 P75 P76 C70 C71 C72 C73 C74 |
                    # L6 | P60 P61 P62 P63 P64 P65 C60 C61 C62 C63 C64 P66 |
                    # L5 | P50 P51 P52 P53 P54 C50 C51 C52 C53 C54 P55 P56 |
                    # L4 | P40 P41 P42 P43 C40 C41 C42 C43 C44 P44 P45 P46 |
                    # L3 | P30 P31 P32 C30 C31 C32 C33 C34 P33 P34 P35 P36 |
                    # L2 | P20 P21 C20 C21 C22 C23 C24 P22 P23 P24 P25 P26 |
                    # L1 | P10 C10 C11 C12 C13 C14 P11 P12 P13 P14 P15 P16 |
                    # L0 | C00 C01 C02 C03 C04 P00 P01 P02 P03 P04 P05 P06 |

                    additional_audio_tokens = []
                    for i in range(self.audio_tokenizer.num_codebook):
                        audio_tokens_ = ""
                        for j in range(i):
                            audio_tokens_ += f"<|audio_{i}_{j}_pad|>"

                        for j in audio_tokens[i]:
                            audio_tokens_ += f"<|audio_{i}_{j}|>"

                        for j in range(i, self.audio_tokenizer.num_codebook - 1):
                            audio_tokens_ += f"<|audio_{i}_{j}_pad|>"

                        audio_tokens_ = tokenizer(audio_tokens_, add_special_tokens=False).input_ids
                        if i > 0:
                            CB_AUD_FIRST_ID = tokenizer.convert_tokens_to_ids(f"<|audio_{i}_0|>")
                            audio_tokens_ = [x - CB_AUD_FIRST_ID for x in audio_tokens_]
                        additional_audio_tokens.append(audio_tokens_)

                    audio_tokens = additional_audio_tokens[0]

                elif isinstance(audio_tokens, list):
                    audio_tokens = [i + AUD_FIRST_ID for i in audio_tokens]
                else:
                    NotImplementedError

                new_input_ids += [AUD_START_ID]
                if targets is not None:
                    new_targets += [AUD_START_ID]
                if additional_targets_list is not None:
                    additional_targets_list = [x + [AUD_START_ID] for x in additional_targets_list]

                new_input_ids += audio_tokens
                if targets is not None:
                    new_targets += audio_tokens
                if additional_targets_list is not None:
                    additional_targets_list = [
                        x + y for x, y in zip(additional_targets_list, additional_audio_tokens)
                    ]

                new_input_ids += [AUD_END_ID]
                if targets is not None:
                    new_targets += [AUD_END_ID]
                if additional_targets_list is not None:
                    additional_targets_list = [x + [AUD_END_ID] for x in additional_targets_list]

            # --------------------------------------------------------------------------
            # add contiguous
            if aud_idx in contiguous_audio_idxs:

                # audio = audio_tokenizer.encode(audio_or_paths[aud_idx], is_contiguous=True)
                audio_dict = self.process_audio(
                    audio_or_paths[aud_idx], is_contiguous=True
                )
                audio = audio_dict["audio"]
                audio_token_length_func = audio_dict["audio_token_length_func"]
                duration_seconds = audio_dict["duration_seconds"]

                # audio_chunk_second = random.randint(int(audio_chunk_min_second), int(audio_chunk_max_second))
                second_per_audio = 1.0 * duration_seconds / len(audio)

                audio_chunks = []
                audio_second_chunks = []

                audio_chunk = []
                for audio_frame in audio:
                    audio_second = 1 * second_per_audio
                    audio_chunk_second = 1.0 * len(audio_chunk) * second_per_audio

                    if audio_second + audio_chunk_second < audio_chunk_min_second:
                        audio_chunk.append(audio_frame)

                    elif audio_second + audio_chunk_second <= audio_chunk_max_second:
                        audio_chunk.append(audio_frame)

                    else:
                        audio_second_chunks.append(sum([len(x) * second_per_audio for x in audio_chunks]))
                        audio_chunks.append(torch.stack(audio_chunk, dim=0))

                        audio_chunk = []
                        audio_chunk.append(audio_frame)

                if len(audio_chunk) > 0:
                    audio_second_chunks.append(sum([len(x) * second_per_audio for x in audio_chunks]))
                    audio_chunks.append(torch.stack(audio_chunk, dim=0))

                audios.extend(audio_chunks)
                # print(f"{audios=}")

                timestamp_format = "HHMMSS"

                for audio, audio_second in zip(audio_chunks, audio_second_chunks):

                    # add timestamp
                    if len(audio_chunks) > 1:
                        if timestamp_format == "HHMMSS":
                            audio_second = round(audio_second)
                            timestamp = time.strftime("%H:%M:%S", time.gmtime(audio_second))
                        else:
                            timestamp = f"{audio_second:.2f}"
                        _input_id = tokenizer(timestamp, add_special_tokens=False).input_ids
                        new_input_ids += _input_id
                        if targets is not None:
                            if is_pretrain:
                                new_targets += _input_id
                            else:
                                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(_input_id)

                    new_input_ids += [AUD_START_ID]
                    if targets is not None:
                        if is_pretrain:
                            new_targets += [AUD_START_ID]
                        else:
                            new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]
                    if additional_targets_list is not None:
                        additional_targets_list = [
                            x + [GLOBAL_TOKEN.IGNORE_TOKEN_ID] for x in additional_targets_list
                        ]

                    audio_token_length = audio_token_length_func(audio)
                    audio_indice_b = torch.zeros(
                        1, audio_token_length, dtype=torch.int64
                    )  # This will change in collate_fn
                    audio_indice_s = (
                        torch.arange(len(new_input_ids), len(new_input_ids) + audio_token_length)
                        .unsqueeze(0)
                        .repeat(1, 1)
                    )
                    audio_indice_b_s = torch.stack(
                        [audio_indice_b, audio_indice_s], dim=0
                    )  # 2, num_audio, audio_length
                    audio_indices.append(audio_indice_b_s)

                    new_input_ids += [AUD_CONTEXT_ID] * audio_token_length
                    if targets is not None:
                        new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * audio_token_length
                    if additional_targets_list is not None:
                        additional_targets_list = [
                            x + [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * audio_token_length
                            for x in additional_targets_list
                        ]

                    new_input_ids += [AUD_END_ID]
                    if targets is not None:
                        if is_pretrain:
                            new_targets += [AUD_END_ID]
                        else:
                            new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]
                    if additional_targets_list is not None:
                        additional_targets_list = [
                            x + [GLOBAL_TOKEN.IGNORE_TOKEN_ID] for x in additional_targets_list
                        ]

            st = aud_pos + 1

        new_input_ids += input_ids[st:]
        if targets is not None:
            new_targets += targets[st:]
        if additional_targets_list is not None:
            additional_targets_list = [
                x + [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * (len(targets) - st)
                for x in additional_targets_list
            ]

        input_ids = new_input_ids
        if targets is not None:
            targets = new_targets

        if targets is not None:
            return (
                input_ids,
                audios,
                audio_indices,
                targets,
                additional_targets_list,
                additional_targets_list,
            )

        return input_ids, audios, audio_indices


def has_audio(video_path):
    if not isinstance(video_path, str):
        return False

    if os.path.isdir(video_path):
        return False

    try:
        # Probe for audio streams only
        probe_result = ffmpeg.probe(video_path, select_streams="a")
        # If 'streams' list is not empty, it indicates an audio stream exists
        return bool(probe_result["streams"])
    except ffmpeg.Error as e:
        print(f"Error probing video: {e.stderr.decode()}", flush=True)
        return False


class YoutuVITAVideoProcessor(BaseVideoProcessor):
    model_input_names = ["pixel_values", "image_grid_thw"]
    valid_kwargs = YoutuVITAVideosKwargs


    def __init__(
        self,
        image_processor=None,
        audio_processor=None,
        resolution_type=None,
        min_num_tokens=64,
        max_num_tokens=8192,
        image_min_num_tokens=4,
        image_max_num_tokens=256,
        temporal_patch_size=1,
        spatial_merge_size=2,
        patch_size=14,
        video_key_frame=False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.image_processor = image_processor
        self.audio_processor = audio_processor

        self.resolution_type = resolution_type
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.max_num_tokens = max_num_tokens
        self.min_num_tokens = min_num_tokens
        self.image_max_num_tokens = image_max_num_tokens
        self.image_min_num_tokens = image_min_num_tokens

        self.sample_rate = 16000

        self.video_key_frame = video_key_frame

    def to_dict(self):
        output = super().to_dict()
        # Remove the non-serializable object before returning
        if "image_processor" in output:
            del output["image_processor"]
        if "audio_processor" in output:
            del output["audio_processor"]
        return output

    def get_frame_paths(self, frame_root, num_frames=8):
        os.makedirs(frame_root, exist_ok=True)

        self.frame_tmpl = "frame-{}-of-{}.jpg"
        return [
            os.path.join(frame_root, self.frame_tmpl.format(i, num_frames))
            for i in range(1, num_frames + 1)
        ]

    def save_video_frames(self, vid_path, max_fps=1, num_frames=8):

        vid = decord.VideoReader(vid_path, num_threads=1)

        # step_size = len(vid) / (num_frames + 1)
        step_size = len(vid) / num_frames

        # step_size = max(1, step_size)
        fps = vid.get_avg_fps()
        fps = round(fps)
        step_size = max(fps / max_fps, step_size)

        # indices = [int(i * step_size) for i in range(1, num_frames + 1)]
        indices = [int(i * step_size) for i in range(0, num_frames)]
        indices = [i for i in indices if i < len(vid)]

        num_frames = len(indices)

        frame_paths = self.get_frame_paths(vid_path + ".saved_frames", num_frames)
        flag = np.all([os.path.exists(p) for p in frame_paths])
        if flag:
            return frame_paths

        images = [vid[i].asnumpy() for i in indices]
        images = [PIL.Image.fromarray(arr) for arr in images]

        for im, pth in zip(images, frame_paths):
            # if not os.path.exists(pth):
            #     im.save(pth)
            im.save(pth)
        # print(f"save_video_frames vid_path {vid_path} fps {fps} len(vid) {len(vid)} frame_paths {frame_paths}")
        return frame_paths

    def get_video_frames(self, vid_path, max_fps=1, num_frames=8):

        vid = decord.VideoReader(vid_path, num_threads=1)

        fps = vid.get_avg_fps()
        fps = round(fps)

        if self.video_key_frame:
            indices = get_video_keyframe(vid_path)
            # TODO: fix this
            sample_fps = None
            sample_fps = 1
            if len(indices) > num_frames:
                random_indices = sorted(random.sample(range(len(indices)), num_frames))
                indices = [indices[i] for i in random_indices]
        else:
            # step_size = len(vid) / (num_frames + 1)
            step_size = len(vid) / num_frames
            step_size = max(fps / max_fps, step_size)

            # indices = [int(i * step_size) for i in range(1, num_frames + 1)]
            indices = [int(i * step_size) for i in range(0, num_frames)]
            sample_fps = 1 / (step_size / fps)

        indices = [i for i in indices if i < len(vid)]

        images = [vid[i].asnumpy() for i in indices]
        images = [PIL.Image.fromarray(arr) for arr in images]

        timestamps = [1.0 / fps * i for i in indices]

        duration_seconds = len(vid) / vid.get_avg_fps()

        # print(f"get_video_frames vid_path {vid_path} fps {fps} len(vid) {len(vid)} frame_paths {frame_paths}")
        return images, sample_fps, timestamps, duration_seconds

    def get_image_and_audio(self, video_file_or_dir, max_num_frames=8, max_fps=1):

        if isinstance(video_file_or_dir, str) and os.path.isfile(video_file_or_dir):
            mime_type, _ = mimetypes.guess_type(video_file_or_dir)
        else:
            mime_type = None

        if mime_type is None:
            mime_type = "video/"

        if isinstance(video_file_or_dir, PIL.Image.Image):
            img_or_path_list = [
                video_file_or_dir,
            ]
            fps = 1
            timestamps = [1 / fps * i for i in range(len(img_or_path_list))]
            duration_seconds = len(img_or_path_list) / fps

        elif os.path.isfile(video_file_or_dir) and mime_type.startswith("image/"):
            img_or_path_list = [PIL.Image.open(x).convert("RGB") for x in [video_file_or_dir]]
            fps = 1
            timestamps = [1 / fps * i for i in range(len(img_or_path_list))]
            duration_seconds = len(img_or_path_list) / fps

        elif os.path.isfile(video_file_or_dir) and mime_type.startswith("video/"):
            # frame_paths = self.save_video_frames(
            #     video_file_or_dir, num_frames=max_num_frames, max_fps=max_fps
            # )
            # img_or_path_list = frame_paths
            img_or_path_list, fps, timestamps, duration_seconds = self.get_video_frames(
                video_file_or_dir, num_frames=max_num_frames, max_fps=max_fps
            )

        elif os.path.isdir(video_file_or_dir):
            all_filepath = []
            for root, dirs, files in os.walk(video_file_or_dir):
                for filename in files:
                    if (
                        filename.endswith("png")
                        or filename.endswith("jpeg")
                        or filename.endswith("jpg")
                    ):
                        filepath = os.path.join(root, filename)
                        all_filepath.append(filepath)

            if len(all_filepath) == 0:
                return None

            # all_filepath.sort()
            all_filepath = natsort.natsorted(all_filepath)
            total_frames = len(all_filepath)
            if "ShareGPTVideo" in video_file_or_dir:
                fps = 2
            else:
                fps = 1
            target_frame = int(min(total_frames / fps * max_fps, max_num_frames))
            index = [int(1.0 * total_frames / target_frame) * x for x in range(target_frame)]

            selected_filepath = [all_filepath[x] for x in index]

            img_or_path_list = selected_filepath

            fps = target_frame / total_frames * fps
            second_per_frame = [1.0 / fps] * len(img_or_path_list)
            timestamps = [1.0 / fps * i for i in range(len(img_or_path_list))]

            duration_seconds = total_frames / fps

        else:
            # print(f"FileNotFoundError {video_file_or_dir}")
            if isinstance(video_file_or_dir, str):
                raise FileNotFoundError(video_file_or_dir)
            else:
                raise NotImplementedError(video_file_or_dir)

        audio = None
        # if has_audio_track(video_file_or_dir):
        if has_audio(video_file_or_dir):
            audio, sample_rate = torchaudio.load(video_file_or_dir)
            # print(f"{audio.size()=} {sample_rate=}")
            if audio.dim() == 2:
                audio = audio.mean(0)

            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.sample_rate
                )
                audio = resampler(audio[None, :])[0, :]

        return img_or_path_list, fps, timestamps, (audio, self.sample_rate), duration_seconds

    def process_video(self, video_file_or_dir, max_num_frames=8, max_fps=1):

        images, fps, timestamps, (audio, sample_rate), duration_seconds = self.get_image_and_audio(
            video_file_or_dir,
            max_num_frames=max_num_frames,
            max_fps=max_fps,
        )

        if self.resolution_type == "native":
            min_pixels = (
                (self.patch_size * self.spatial_merge_size) ** 2
                * self.min_num_tokens
                // len(images)
            )
            max_pixels = (
                (self.patch_size * self.spatial_merge_size) ** 2
                * self.max_num_tokens
                // len(images)
            )

            image_min_pixels = (
                self.patch_size * self.spatial_merge_size
            ) ** 2 * self.image_min_num_tokens
            image_max_pixels = (
                self.patch_size * self.spatial_merge_size
            ) ** 2 * self.image_max_num_tokens

            min_pixels = max(min_pixels, image_min_pixels)
            max_pixels = min(max_pixels, image_max_pixels)

            # print(f"{len(images)=} {min_pixels=} {max_pixels=}")
            image_data = self.image_processor.process_images(
                images,
                is_contiguous=True,
                resolution_type=self.resolution_type,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            image_frames = image_data["images"]
            best_height = image_data["image_height"]
            best_width = image_data["image_width"]
        else:
            image_data = self.image_processor.process_images_to_tensor(
                images
            )
            image_frames = image_data["images"]
            best_height = image_data["image_height"]
            best_width = image_data["image_width"]

        if audio is not None:
            total_time = len(audio) / sample_rate
            # print(f"{duration_seconds=} {total_time=}", flush=True)

            audio_dict = self.audio_processor.process_audio(
                (audio, sample_rate), is_discrete=False, is_contiguous=True
            )
            audio = audio_dict["audio"]
            audio_token_length_func = audio_dict["audio_token_length_func"]

            # audio_frames = torch.chunk(audio, chunks=len(image_frames), dim=0)
            full_timestamps = timestamps + [total_time]
            audio_frames = []
            for split_idx in range(len(image_frames)):
                st = full_timestamps[split_idx]
                ed = full_timestamps[split_idx + 1]

                st = int(st / total_time * len(audio))
                ed = int(ed / total_time * len(audio))

                audio_frame = audio[st:ed]
                audio_frames.append(audio_frame)

        else:
            audio_frames = None
            audio_token_length_func = None

        # grid_t = image_frames.size(1) // self.temporal_patch_size
        grid_t = 1 // self.temporal_patch_size
        grid_h = best_height // self.patch_size
        grid_w = best_width // self.patch_size

        video_grid_thw = [[grid_t, grid_h, grid_w]]
        video_grid_thw = video_grid_thw * image_frames.size(0)

        if fps is not None:
            second_per_grids = [1.0 / fps] * image_frames.size(0)
        else:
            second_per_grids = None

        # print(f"{image_frames.size()=} {video_grid_thw=} {max_fps=} {max_num_frames=} {fps=} {second_per_grids=} ")

        return (
            image_frames,
            audio_frames,
            audio_token_length_func,
            video_grid_thw,
            second_per_grids,
            timestamps,
            duration_seconds,
        )

    def process_video_raw(
        self, video_file_or_dir, max_num_frames=8, max_fps=1, use_audio_in_video=True
    ):

        images, fps, timestamps, (audio, sample_rate) = self.get_image_and_audio(
            video_file_or_dir,
            max_num_frames=max_num_frames,
            max_fps=max_fps,
            use_audio_in_video=use_audio_in_video,
        )

        image_frames = images

        if audio is not None:
            total_time = len(audio) / sample_rate

            # audio_frames = torch.chunk(audio, chunks=len(image_frames), dim=0)
            full_timestamps = timestamps + [total_time]
            audio_frames = []
            for split_idx in range(len(image_frames)):
                st = full_timestamps[split_idx]
                ed = full_timestamps[split_idx + 1]

                st = int(st / total_time * len(audio))
                ed = int(ed / total_time * len(audio))

                audio_frame = audio[st:ed]
                audio_frames.append(audio_frame)

        else:
            audio_frames = None

        # grid_t = image_frames.size(1) // self.temporal_patch_size
        grid_t = 1 // self.temporal_patch_size
        grid_h = self.image_processor.tile_image_size // self.patch_size
        grid_w = self.image_processor.tile_image_size // self.patch_size

        video_grid_thw = [[grid_t, grid_h, grid_w]]
        video_grid_thw = video_grid_thw * len(image_frames)

        if fps is not None:
            second_per_grids = [1.0 / fps] * len(image_frames)
        else:
            second_per_grids = None

        # print(f"{len(image_frames)=} {video_grid_thw=} {max_fps=} {max_num_frames=} {fps=} {second_per_grids=} ")

        return image_frames, audio_frames, video_grid_thw, second_per_grids, timestamps

    def add_video_input_contiguous(
        self,
        input_ids,
        video_paths,
        tokenizer,
        targets=None,
        is_pretrain=False,
        max_num_frames=4096,
        max_fps=1,
        use_audio_in_video=True,
        use_vision_in_video=True,
        video_audio_chunk_min_second=2,
        video_audio_chunk_max_second=30,
        **kwargs,
    ):

        GLOBAL_TOKEN = get_token()

        IMG_CONTEXT_ID = tokenizer(
            GLOBAL_TOKEN.IMG_CONTEXT_TOKEN, add_special_tokens=False
        ).input_ids
        IMG_START_ID = tokenizer(GLOBAL_TOKEN.IMG_START_TOKEN, add_special_tokens=False).input_ids
        IMG_END_ID = tokenizer(GLOBAL_TOKEN.IMG_END_TOKEN, add_special_tokens=False).input_ids

        AUD_CONTEXT_ID = tokenizer(
            GLOBAL_TOKEN.AUD_CONTEXT_TOKEN, add_special_tokens=False
        ).input_ids
        AUD_START_ID = tokenizer(GLOBAL_TOKEN.AUD_START_TOKEN, add_special_tokens=False).input_ids
        AUD_END_ID = tokenizer(GLOBAL_TOKEN.AUD_END_TOKEN, add_special_tokens=False).input_ids

        VID_CONTEXT_ID = tokenizer(
            GLOBAL_TOKEN.VID_CONTEXT_TOKEN, add_special_tokens=False
        ).input_ids
        VID_START_ID = tokenizer(GLOBAL_TOKEN.VID_START_TOKEN, add_special_tokens=False).input_ids
        VID_END_ID = tokenizer(GLOBAL_TOKEN.VID_END_TOKEN, add_special_tokens=False).input_ids

        PATCH_CONTEXT_ID = tokenizer(
            GLOBAL_TOKEN.PATCH_CONTEXT_TOKEN, add_special_tokens=False
        ).input_ids
        PATCH_START_ID = tokenizer(
            GLOBAL_TOKEN.PATCH_START_TOKEN, add_special_tokens=False
        ).input_ids
        PATCH_END_ID = tokenizer(GLOBAL_TOKEN.PATCH_END_TOKEN, add_special_tokens=False).input_ids

        IMG_TAG_ID = tokenizer(GLOBAL_TOKEN.IMG_TAG_TOKEN, add_special_tokens=False).input_ids
        AUD_TAG_ID = tokenizer(GLOBAL_TOKEN.AUD_TAG_TOKEN, add_special_tokens=False).input_ids
        VID_TAG_ID = tokenizer(GLOBAL_TOKEN.VID_TAG_TOKEN, add_special_tokens=False).input_ids

        assert len(IMG_CONTEXT_ID) == 1
        assert len(IMG_START_ID) == 1
        assert len(IMG_END_ID) == 1

        assert len(AUD_CONTEXT_ID) == 1
        assert len(AUD_START_ID) == 1
        assert len(AUD_END_ID) == 1

        assert len(VID_CONTEXT_ID) == 1
        assert len(VID_START_ID) == 1
        assert len(VID_END_ID) == 1

        assert len(PATCH_CONTEXT_ID) == 1
        assert len(PATCH_START_ID) == 1
        assert len(PATCH_END_ID) == 1

        IMG_CONTEXT_ID = IMG_CONTEXT_ID[0]
        IMG_START_ID = IMG_START_ID[0]
        IMG_END_ID = IMG_END_ID[0]

        AUD_CONTEXT_ID = AUD_CONTEXT_ID[0]
        AUD_START_ID = AUD_START_ID[0]
        AUD_END_ID = AUD_END_ID[0]

        VID_CONTEXT_ID = VID_CONTEXT_ID[0]
        VID_START_ID = VID_START_ID[0]
        VID_END_ID = VID_END_ID[0]

        PATCH_CONTEXT_ID = PATCH_CONTEXT_ID[0]
        PATCH_START_ID = PATCH_START_ID[0]
        PATCH_END_ID = PATCH_END_ID[0]

        IMG_TAG_ID = IMG_TAG_ID[0]
        AUD_TAG_ID = AUD_TAG_ID[0]
        VID_TAG_ID = VID_TAG_ID[0]

        nl_tokens = tokenizer("\n", add_special_tokens=False).input_ids

        vid_positions = [i for i, x in enumerate(input_ids) if x == VID_TAG_ID]
        assert len(vid_positions) == len(video_paths), video_paths

        images = []
        image_indices = []
        audios = []
        audio_indices = []
        video_grid_thw = []
        second_per_grids = []

        new_input_ids = []
        new_targets = []
        st = 0
        for vid_idx, vid_pos in enumerate(vid_positions):
            (
                image_frames,
                audio_frames,
                audio_token_length_func,
                _video_grid_thw,
                _second_per_grids,
                second_frames,
                duration_seconds,
            ) = self.process_video(video_paths[vid_idx], max_num_frames, max_fps)

            if audio_frames is not None:
                # print(f"{len(image_frames)=} {len(audio_frames)=}")
                # print(f"{[x.size() for x in image_frames]=}")
                # print(f"{[x.size() for x in audio_frames]=}")

                num_frames = min(len(image_frames), len(audio_frames))
                image_frames = image_frames[:num_frames]
                audio_frames = audio_frames[:num_frames]

            new_input_ids += input_ids[st:vid_pos]
            if targets is not None:
                new_targets += targets[st:vid_pos]

            if audio_frames is not None:
                second_per_audio = 1.0 * duration_seconds / sum([len(x) for x in audio_frames])

                image_chunks = []
                image_second_chunks = []
                audio_chunks = []
                audio_second_chunks = []

                image_chunk = []
                image_second_chunk = []
                audio_chunk = []
                audio_second_chunk = []
                for image_frame, audio_frame, second_frame in zip(
                    image_frames, audio_frames, second_frames
                ):
                    audio_second = len(audio_frame) * second_per_audio
                    audio_chunk_second = sum([len(x) * second_per_audio for x in audio_chunk])

                    if audio_second + audio_chunk_second < video_audio_chunk_min_second:
                        image_chunk.append(image_frame)
                        image_second_chunk.append(second_frame)

                        audio_chunk.append(audio_frame)
                        audio_second_chunk.append(second_frame)

                    elif audio_second + audio_chunk_second <= video_audio_chunk_max_second:
                        image_chunk.append(image_frame)
                        image_second_chunk.append(second_frame)

                        audio_chunk.append(audio_frame)
                        audio_second_chunk.append(second_frame)

                        image_chunks.append(image_chunk)
                        image_second_chunks.append(image_second_chunk)

                        audio_chunk = [torch.cat(audio_chunk, dim=0)]
                        audio_second_chunk = [audio_second_chunk[0]]
                        audio_chunks.append(audio_chunk)
                        audio_second_chunks.append(audio_second_chunk)

                        image_chunk = []
                        image_second_chunk = []

                        audio_chunk = []
                        audio_second_chunk = []

                    elif audio_second + audio_chunk_second > video_audio_chunk_max_second:
                        assert audio_chunk_second == 0

                        image_chunk.append(image_frame)
                        image_second_chunk.append(second_frame)

                        audio_chunk_size = int(video_audio_chunk_max_second / second_per_audio)
                        audio_chunk = torch.split(audio_frame, audio_chunk_size)
                        cur_second = second_frame
                        for x in audio_chunk:
                            audio_second_chunk.append(cur_second)
                            cur_second += len(x) * second_per_audio

                        image_chunks.append(image_chunk)
                        image_second_chunks.append(image_second_chunk)

                        audio_chunks.append(audio_chunk)
                        audio_second_chunks.append(audio_second_chunk)

                        image_chunk = []
                        image_second_chunk = []

                        audio_chunk = []
                        audio_second_chunk = []

                if len(image_chunk) > 0:
                    image_chunks.append(image_chunk)
                    image_second_chunks.append(image_second_chunk)

                    audio_chunk = [torch.cat(audio_chunk, dim=0)]
                    audio_second_chunk = [audio_second_chunk[0]]
                    audio_chunks.append(audio_chunk)
                    audio_second_chunks.append(audio_second_chunk)

            else:
                image_chunks = [image_frames]
                image_second_chunks = [second_frames]
                audio_chunks = [[]]
                audio_second_chunks = [[]]

            assert len(image_frames) == sum([len(x) for x in image_chunks])
            # assert len(audio_frames) == sum([len(x) for x in audio_chunks])

            if use_vision_in_video:
                if self.image_processor.resolution_type == "native":
                    images.append(
                        torch.cat(
                            [
                                self.image_processor.convert_image_to_patches_with_pixel_shuffle(x)
                                for x in image_frames
                            ],
                            dim=0,
                        )
                    )

                else:
                    images.append(image_frames)

            if use_audio_in_video and audio_frames is not None:
                # audios.extend(audio_frames)
                audios.extend([xx for x in audio_chunks for xx in x])

            for (
                image_chunk_frames,
                image_second_chunk_frames,
                audio_chunk_frames,
                audio_second_chunk_frames,
            ) in zip(
                image_chunks,
                image_second_chunks,
                audio_chunks,
                audio_second_chunks,
            ):

                if not use_vision_in_video:
                    image_chunk_frames = []
                    image_second_chunk_frames = []
                if not use_audio_in_video:
                    audio_chunk_frames = []
                    audio_second_chunk_frames = []

                for image_chunk_frame, image_second_chunk_frame in zip(
                    image_chunk_frames, image_second_chunk_frames
                ):

                    # add timestamp
                    timestamp = time.strftime("%H:%M:%S", time.gmtime(round(image_second_chunk_frame)))
                    _input_id = tokenizer(timestamp, add_special_tokens=False).input_ids
                    _target = [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(_input_id)
                    new_input_ids += _input_id
                    if targets is not None:
                        new_targets += _target

                    new_input_ids += [VID_START_ID]
                    if targets is not None:
                        new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

                    image_token_length = (
                        _video_grid_thw[0][0]
                        * _video_grid_thw[0][1]
                        * _video_grid_thw[0][2]
                        // self.spatial_merge_size
                        // self.spatial_merge_size
                    )
                    image_indice_b = torch.zeros(
                        1, image_token_length, dtype=torch.int64
                    )  # This will change in collate_fn
                    image_indice_s = (
                        torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
                        .unsqueeze(0)
                        .repeat(1, 1)
                    )
                    image_indice_b_s = torch.stack(
                        [image_indice_b, image_indice_s], dim=0
                    )  # 2, num_image, image_length
                    if self.image_processor.resolution_type == "native":
                        image_indices.append(image_indice_b_s.view(2, -1))
                    else:
                        image_indices.append(image_indice_b_s)

                    new_input_ids += [VID_CONTEXT_ID] * image_token_length
                    if targets is not None:
                        new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * image_token_length

                    new_input_ids += [VID_END_ID]
                    if targets is not None:
                        new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

                for audio_chunk_frame, audio_second_chunk_frame in zip(
                    audio_chunk_frames, audio_second_chunk_frames
                ):

                    # add timestamp
                    timestamp = time.strftime("%H:%M:%S", time.gmtime(round(audio_second_chunk_frame)))
                    _input_id = tokenizer(timestamp, add_special_tokens=False).input_ids
                    _target = [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(_input_id)
                    new_input_ids += _input_id
                    if targets is not None:
                        new_targets += _target

                    new_input_ids += [AUD_START_ID]
                    if targets is not None:
                        new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

                    # audio_token_length = audio_chunk_frame.size(0)
                    # audio_token_length = audio_token_length_func(audio_chunk_frame.size(0))
                    audio_token_length = audio_token_length_func(audio_chunk_frame)
                    audio_indice_b = torch.zeros(
                        1, audio_token_length, dtype=torch.int64
                    )  # This will change in collate_fn
                    audio_indice_s = (
                        torch.arange(len(new_input_ids), len(new_input_ids) + audio_token_length)
                        .unsqueeze(0)
                        .repeat(1, 1)
                    )
                    audio_indice_b_s = torch.stack(
                        [audio_indice_b, audio_indice_s], dim=0
                    )  # 2, num_audio, audio_length
                    audio_indices.append(audio_indice_b_s)

                    new_input_ids += [AUD_CONTEXT_ID] * audio_token_length
                    if targets is not None:
                        new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * audio_token_length

                    new_input_ids += [AUD_END_ID]
                    if targets is not None:
                        new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            video_grid_thw.extend(_video_grid_thw)
            second_per_grids.extend(_second_per_grids)

            st = vid_pos + 1

        new_input_ids += input_ids[st:]
        if targets is not None:
            new_targets += targets[st:]

        input_ids = new_input_ids
        if targets is not None:
            targets = new_targets

        video_grid_thw = torch.tensor(video_grid_thw, dtype=torch.long)
        second_per_grids = torch.tensor(second_per_grids, dtype=torch.long)

        if targets is not None:
            return (
                input_ids,
                images,
                image_indices,
                audios,
                audio_indices,
                video_grid_thw,
                second_per_grids,
                targets,
            )

        if len(images) == 0:
            images = None
            image_indices = None

        else:
            images = torch.cat(images, dim=0)
            image_indices = torch.cat(image_indices, dim=1)

            image_indices = image_indices.contiguous().to(torch.cuda.current_device())
            if True:
                images = (
                    torch.tensor(images, dtype=torch.bfloat16)
                    .contiguous()
                    .to(torch.cuda.current_device())
                )

            else:
                images = (
                    torch.tensor(images, dtype=torch.float16)
                    .contiguous()
                    .to(torch.cuda.current_device())
                )

        if len(audios) == 0:
            audios = None
            audio_indices = None

        return (
            input_ids,
            images,
            image_indices,
            audios,
            audio_indices,
            video_grid_thw,
            second_per_grids,
        )


    def add_video_input_discrete_and_contiguous(
        self,
        input_ids,
        video_paths,
        tokenizer,
        targets=None,
        is_pretrain=False,
        image_token_length=256,
        max_num_frames=4096,
        max_fps=1,
        use_audio_in_video=True,
        video_chunk_size=2,
        video_audio_chunk_size=None,
        video_audio_num_subchunk=None,
        **kwargs,
    ):

        GLOBAL_TOKEN = get_token()

        IMG_CONTEXT_ID = tokenizer(
            GLOBAL_TOKEN.IMG_CONTEXT_TOKEN, add_special_tokens=False
        ).input_ids
        IMG_START_ID = tokenizer(GLOBAL_TOKEN.IMG_START_TOKEN, add_special_tokens=False).input_ids
        IMG_END_ID = tokenizer(GLOBAL_TOKEN.IMG_END_TOKEN, add_special_tokens=False).input_ids

        AUD_CONTEXT_ID = tokenizer(
            GLOBAL_TOKEN.AUD_CONTEXT_TOKEN, add_special_tokens=False
        ).input_ids
        AUD_START_ID = tokenizer(GLOBAL_TOKEN.AUD_START_TOKEN, add_special_tokens=False).input_ids
        AUD_END_ID = tokenizer(GLOBAL_TOKEN.AUD_END_TOKEN, add_special_tokens=False).input_ids

        VID_CONTEXT_ID = tokenizer(
            GLOBAL_TOKEN.VID_CONTEXT_TOKEN, add_special_tokens=False
        ).input_ids
        VID_START_ID = tokenizer(GLOBAL_TOKEN.VID_START_TOKEN, add_special_tokens=False).input_ids
        VID_END_ID = tokenizer(GLOBAL_TOKEN.VID_END_TOKEN, add_special_tokens=False).input_ids

        PATCH_CONTEXT_ID = tokenizer(
            GLOBAL_TOKEN.PATCH_CONTEXT_TOKEN, add_special_tokens=False
        ).input_ids
        PATCH_START_ID = tokenizer(
            GLOBAL_TOKEN.PATCH_START_TOKEN, add_special_tokens=False
        ).input_ids
        PATCH_END_ID = tokenizer(GLOBAL_TOKEN.PATCH_END_TOKEN, add_special_tokens=False).input_ids

        IMG_TAG_ID = tokenizer(GLOBAL_TOKEN.IMG_TAG_TOKEN, add_special_tokens=False).input_ids
        AUD_TAG_ID = tokenizer(GLOBAL_TOKEN.AUD_TAG_TOKEN, add_special_tokens=False).input_ids
        VID_TAG_ID = tokenizer(GLOBAL_TOKEN.VID_TAG_TOKEN, add_special_tokens=False).input_ids

        assert len(IMG_CONTEXT_ID) == 1
        assert len(IMG_START_ID) == 1
        assert len(IMG_END_ID) == 1

        assert len(AUD_CONTEXT_ID) == 1
        assert len(AUD_START_ID) == 1
        assert len(AUD_END_ID) == 1

        assert len(VID_CONTEXT_ID) == 1
        assert len(VID_START_ID) == 1
        assert len(VID_END_ID) == 1

        assert len(PATCH_CONTEXT_ID) == 1
        assert len(PATCH_START_ID) == 1
        assert len(PATCH_END_ID) == 1

        IMG_CONTEXT_ID = IMG_CONTEXT_ID[0]
        IMG_START_ID = IMG_START_ID[0]
        IMG_END_ID = IMG_END_ID[0]

        AUD_CONTEXT_ID = AUD_CONTEXT_ID[0]
        AUD_START_ID = AUD_START_ID[0]
        AUD_END_ID = AUD_END_ID[0]

        VID_CONTEXT_ID = VID_CONTEXT_ID[0]
        VID_START_ID = VID_START_ID[0]
        VID_END_ID = VID_END_ID[0]

        PATCH_CONTEXT_ID = PATCH_CONTEXT_ID[0]
        PATCH_START_ID = PATCH_START_ID[0]
        PATCH_END_ID = PATCH_END_ID[0]

        IMG_TAG_ID = IMG_TAG_ID[0]
        AUD_TAG_ID = AUD_TAG_ID[0]
        VID_TAG_ID = VID_TAG_ID[0]

        nl_tokens = tokenizer("\n", add_special_tokens=False).input_ids

        vid_positions = [i for i, x in enumerate(input_ids) if x == VID_TAG_ID]
        assert len(vid_positions) == len(video_paths), video_paths

        images = []
        image_indices = []
        audios = []
        audio_indices = []
        video_grid_thw = []
        second_per_grids = []

        new_input_ids = []
        new_targets = []
        st = 0
        for vid_idx, vid_pos in enumerate(vid_positions):
            (
                image_frames,
                audio_frames,
                _video_grid_thw,
                _second_per_grids,
                second_frames,
            ) = self.process_video_raw(
                video_paths[vid_idx], max_num_frames, max_fps, use_audio_in_video=use_audio_in_video
            )

            if audio_frames is not None:
                # print(f"{len(image_frames)=} {len(audio_frames)=}")
                # print(f"{[x.size() for x in image_frames]=}")
                # print(f"{[x.size() for x in audio_frames]=}")

                num_frames = min(len(image_frames), len(audio_frames))
                image_frames = image_frames[:num_frames]
                audio_frames = audio_frames[:num_frames]

            new_input_ids += input_ids[st:vid_pos]
            if targets is not None:
                new_targets += targets[st:vid_pos]

            second_chunks = [
                second_frames[i : i + video_chunk_size]
                for i in range(0, len(second_frames), video_chunk_size)
            ]

            image_chunks = [
                image_frames[i : i + video_chunk_size]
                for i in range(0, len(image_frames), video_chunk_size)
            ]
            if audio_frames is not None:
                audio_chunks = [
                    audio_frames[i : i + video_chunk_size]
                    for i in range(0, len(audio_frames), video_chunk_size)
                ]
            else:
                audio_chunks = [[] for i in range(0, len(image_frames), video_chunk_size)]

            # resplit audio frame
            if audio_frames is not None:

                if video_audio_chunk_size is not None:
                    audio_chunks = [torch.cat(x, dim=0) for x in audio_chunks]
                    audio_chunks = [
                        torch.split(x, video_audio_chunk_size, dim=0) for x in audio_chunks
                    ]

                if video_audio_num_subchunk is not None:
                    audio_chunks = [x[:video_audio_num_subchunk] for x in audio_chunks]
            # print([[xx.size() for xx in x] for x in audio_chunks])

            images.extend(image_frames)
            if audio_frames is not None:
                # audios.extend(audio_frames)
                audios.extend([xx for x in audio_chunks for xx in x])

            new_input_ids += [VID_START_ID]
            if targets is not None:
                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            for image_chunk_frames, audio_chunk_frames, second_chunk_frames in zip(
                image_chunks, audio_chunks, second_chunks
            ):
                for image_chunk_frame, second_chunk_frame in zip(
                    image_chunk_frames, second_chunk_frames
                ):

                    # add timestamp
                    timestamp = time.strftime("%H:%M:%S", time.gmtime(round(second_chunk_frame)))
                    _input_id = tokenizer(timestamp, add_special_tokens=False).input_ids
                    _target = [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(_input_id)
                    new_input_ids += _input_id
                    if targets is not None:
                        new_targets += _target

                    new_input_ids += [IMG_TAG_ID]
                    if targets is not None:
                        new_targets += [IMG_TAG_ID]

                for audio_chunk_frame in audio_chunk_frames:

                    new_input_ids += [AUD_TAG_ID]
                    if targets is not None:
                        new_targets += [AUD_TAG_ID]

            new_input_ids += [VID_END_ID]
            if targets is not None:
                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            video_grid_thw.extend(_video_grid_thw)
            second_per_grids.extend(_second_per_grids)

            st = vid_pos + 1

        new_input_ids += input_ids[st:]
        if targets is not None:
            new_targets += targets[st:]

        input_ids = new_input_ids
        if targets is not None:
            targets = new_targets

        if targets is not None:
            (
                input_ids,
                images,
                image_indices,
                image_grid_thw,
                targets,
            ) = self.image_processor.add_image_input_discrete_and_contiguous(
                input_ids,
                images,
                tokenizer,
                image_token_length=image_token_length,
                use_tile=False,
                targets=targets,
                is_pretrain=is_pretrain,
            )
        else:
            (
                input_ids,
                images,
                image_indices,
                image_grid_thw,
            ) = self.image_processor.add_image_input_discrete_and_contiguous(
                input_ids,
                images,
                tokenizer,
                image_token_length=image_token_length,
                use_tile=False,
            )

        if audio_frames is not None:
            if video_audio_chunk_size is None:
                video_audio_chunk_size = max(max([len(audio) for audio in audios]), 400)
            audios = [
                (
                    audio
                    if len(audio) == video_audio_chunk_size
                    else torch.cat(
                        [audio, torch.zeros(video_audio_chunk_size - len(audio))], dim=0
                    ),
                    self.sample_rate,
                )
                for audio in audios
            ]
            # print(f"{[x[0].size() for x in audios]=}")

            if targets is not None:
                (
                    input_ids,
                    audios,
                    audio_indices,
                    targets,
                ) = self.audio_processor.add_audio_input_discrete_and_contiguous(
                    input_ids, audios, tokenizer, targets=targets, is_pretrain=is_pretrain
                )
            else:
                (
                    input_ids,
                    audios,
                    audio_indices,
                ) = self.audio_processor.add_audio_input_discrete_and_contiguous(
                    input_ids, audios, tokenizer
                )

        video_grid_thw = torch.tensor(video_grid_thw, dtype=torch.long)
        second_per_grids = torch.tensor(second_per_grids, dtype=torch.long)

        if targets is not None:
            return (
                input_ids,
                images,
                image_indices,
                audios,
                audio_indices,
                video_grid_thw,
                second_per_grids,
                targets,
            )

        return (
            input_ids,
            images,
            image_indices,
            audios,
            audio_indices,
            video_grid_thw,
            second_per_grids,
        )

        images = torch.cat(images, dim=0)
        image_indices = torch.cat(image_indices, dim=1)

        image_indices = image_indices.contiguous().to(torch.cuda.current_device())
        if True:
            images = (
                torch.tensor(images, dtype=torch.bfloat16)
                .contiguous()
                .to(torch.cuda.current_device())
            )

        else:
            images = (
                torch.tensor(images, dtype=torch.float16)
                .contiguous()
                .to(torch.cuda.current_device())
            )

        if len(audios) == 0:
            audios = None
            audio_indices = None

        return (
            input_ids,
            images,
            image_indices,
            audios,
            audio_indices,
            video_grid_thw,
            second_per_grids,
        )


    def add_video_input_discrete_or_contiguous(
        self,
        input_ids,
        video_paths,
        tokenizer,
        targets=None,
        discrete_video_idxs=[],
        contiguous_video_idxs=[],
        is_pretrain=False,
        max_num_frames=4096,
        max_fps=1,
        use_audio_in_video=True,
        use_vision_in_video=True,
        video_audio_chunk_min_second=2,
        video_audio_chunk_max_second=30,
        **kwargs,
    ):

        GLOBAL_TOKEN = get_token()

        IMG_CONTEXT_ID = tokenizer(
            GLOBAL_TOKEN.IMG_CONTEXT_TOKEN, add_special_tokens=False
        ).input_ids
        IMG_START_ID = tokenizer(GLOBAL_TOKEN.IMG_START_TOKEN, add_special_tokens=False).input_ids
        IMG_END_ID = tokenizer(GLOBAL_TOKEN.IMG_END_TOKEN, add_special_tokens=False).input_ids

        AUD_CONTEXT_ID = tokenizer(
            GLOBAL_TOKEN.AUD_CONTEXT_TOKEN, add_special_tokens=False
        ).input_ids
        AUD_START_ID = tokenizer(GLOBAL_TOKEN.AUD_START_TOKEN, add_special_tokens=False).input_ids
        AUD_END_ID = tokenizer(GLOBAL_TOKEN.AUD_END_TOKEN, add_special_tokens=False).input_ids

        VID_CONTEXT_ID = tokenizer(
            GLOBAL_TOKEN.VID_CONTEXT_TOKEN, add_special_tokens=False
        ).input_ids
        VID_START_ID = tokenizer(GLOBAL_TOKEN.VID_START_TOKEN, add_special_tokens=False).input_ids
        VID_END_ID = tokenizer(GLOBAL_TOKEN.VID_END_TOKEN, add_special_tokens=False).input_ids

        IMG_TAG_ID = tokenizer(GLOBAL_TOKEN.IMG_TAG_TOKEN, add_special_tokens=False).input_ids
        AUD_TAG_ID = tokenizer(GLOBAL_TOKEN.AUD_TAG_TOKEN, add_special_tokens=False).input_ids
        VID_TAG_ID = tokenizer(GLOBAL_TOKEN.VID_TAG_TOKEN, add_special_tokens=False).input_ids

        assert len(IMG_CONTEXT_ID) == 1
        assert len(IMG_START_ID) == 1
        assert len(IMG_END_ID) == 1

        assert len(AUD_CONTEXT_ID) == 1
        assert len(AUD_START_ID) == 1
        assert len(AUD_END_ID) == 1

        assert len(VID_CONTEXT_ID) == 1
        assert len(VID_START_ID) == 1
        assert len(VID_END_ID) == 1

        IMG_CONTEXT_ID = IMG_CONTEXT_ID[0]
        IMG_START_ID = IMG_START_ID[0]
        IMG_END_ID = IMG_END_ID[0]

        AUD_CONTEXT_ID = AUD_CONTEXT_ID[0]
        AUD_START_ID = AUD_START_ID[0]
        AUD_END_ID = AUD_END_ID[0]

        VID_CONTEXT_ID = VID_CONTEXT_ID[0]
        VID_START_ID = VID_START_ID[0]
        VID_END_ID = VID_END_ID[0]

        IMG_TAG_ID = IMG_TAG_ID[0]
        AUD_TAG_ID = AUD_TAG_ID[0]
        VID_TAG_ID = VID_TAG_ID[0]

        nl_tokens = tokenizer("\n", add_special_tokens=False).input_ids

        vid_positions = [i for i, x in enumerate(input_ids) if x == VID_TAG_ID]
        assert len(vid_positions) == len(video_paths), video_paths

        if not discrete_video_idxs and not contiguous_video_idxs:
            contiguous_video_idxs = list(range(len(video_paths)))

        images = []
        image_indices = []
        audios = []
        audio_indices = []
        video_grid_thw = []
        second_per_grids = []

        new_input_ids = []
        new_targets = []
        st = 0
        for vid_idx, vid_pos in enumerate(vid_positions):
            (
                image_frames,
                audio_frames,
                audio_token_length_func,
                _video_grid_thw,
                _second_per_grids,
                second_frames,
                duration_seconds,
            ) = self.process_video(video_paths[vid_idx], max_num_frames, max_fps)

            if audio_frames is not None:
                # print(f"{len(image_frames)=} {len(audio_frames)=}")
                # print(f"{[x.size() for x in image_frames]=}")
                # print(f"{[x.size() for x in audio_frames]=}")

                num_frames = min(len(image_frames), len(audio_frames))
                image_frames = image_frames[:num_frames]
                audio_frames = audio_frames[:num_frames]

            new_input_ids += input_ids[st:vid_pos]
            if targets is not None:
                new_targets += targets[st:vid_pos]

            if audio_frames is not None:
                second_per_audio = 1.0 * duration_seconds / sum([len(x) for x in audio_frames])

                image_chunks = []
                image_second_chunks = []
                audio_chunks = []
                audio_second_chunks = []

                image_chunk = []
                image_second_chunk = []
                audio_chunk = []
                audio_second_chunk = []
                for image_frame, audio_frame, second_frame in zip(
                    image_frames, audio_frames, second_frames
                ):
                    audio_second = len(audio_frame) * second_per_audio
                    audio_chunk_second = sum([len(x) * second_per_audio for x in audio_chunk])

                    if audio_second + audio_chunk_second < video_audio_chunk_min_second:
                        image_chunk.append(image_frame)
                        image_second_chunk.append(second_frame)

                        audio_chunk.append(audio_frame)
                        audio_second_chunk.append(second_frame)

                    elif audio_second + audio_chunk_second <= video_audio_chunk_max_second:
                        image_chunk.append(image_frame)
                        image_second_chunk.append(second_frame)

                        audio_chunk.append(audio_frame)
                        audio_second_chunk.append(second_frame)

                        image_chunks.append(image_chunk)
                        image_second_chunks.append(image_second_chunk)

                        audio_chunk = [torch.cat(audio_chunk, dim=0)]
                        audio_second_chunk = [audio_second_chunk[0]]
                        audio_chunks.append(audio_chunk)
                        audio_second_chunks.append(audio_second_chunk)

                        image_chunk = []
                        image_second_chunk = []

                        audio_chunk = []
                        audio_second_chunk = []

                    else:
                        assert audio_chunk_second == 0

                        image_chunk.append(image_frame)
                        image_second_chunk.append(second_frame)

                        audio_chunk_size = int(video_audio_chunk_max_second / second_per_audio)
                        audio_chunk = torch.split(audio_frame, audio_chunk_size)
                        cur_second = second_frame
                        for x in audio_chunk:
                            audio_second_chunk.append(cur_second)
                            cur_second += len(x) * second_per_audio

                        image_chunks.append(image_chunk)
                        image_second_chunks.append(image_second_chunk)

                        audio_chunks.append(audio_chunk)
                        audio_second_chunks.append(audio_second_chunk)

                        image_chunk = []
                        image_second_chunk = []

                        audio_chunk = []
                        audio_second_chunk = []

                if len(image_chunk) > 0:
                    image_chunks.append(image_chunk)
                    image_second_chunks.append(image_second_chunk)

                    audio_chunk = [torch.cat(audio_chunk, dim=0)]
                    audio_second_chunk = [audio_second_chunk[0]]
                    audio_chunks.append(audio_chunk)
                    audio_second_chunks.append(audio_second_chunk)

            else:
                image_chunks = [image_frames]
                image_second_chunks = [second_frames]
                audio_chunks = [[]]
                audio_second_chunks = [[]]

            assert len(image_frames) == sum([len(x) for x in image_chunks])
            # assert len(audio_frames) == sum([len(x) for x in audio_chunks])

            if use_vision_in_video:
                if self.image_processor.resolution_type == "native":
                    images.append(
                        torch.cat(
                            [
                                self.image_processor.convert_image_to_patches_with_pixel_shuffle(x)
                                for x in image_frames
                            ],
                            dim=0,
                        )
                    )

                else:
                    images.append(image_frames)

            if use_audio_in_video and audio_frames is not None:
                # audios.extend(audio_frames)
                audios.extend([xx for x in audio_chunks for xx in x])

            new_input_ids += [VID_START_ID]
            if targets is not None:
                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            timestamp_format = "HHMMSS"

            for (
                image_chunk_frames,
                image_second_chunk_frames,
                audio_chunk_frames,
                audio_second_chunk_frames,
            ) in zip(
                image_chunks,
                image_second_chunks,
                audio_chunks,
                audio_second_chunks,
            ):

                if not use_vision_in_video:
                    image_chunk_frames = []
                    image_second_chunk_frames = []
                if not use_audio_in_video:
                    audio_chunk_frames = []
                    audio_second_chunk_frames = []

                for image_chunk_frame, image_second_chunk_frame in zip(
                    image_chunk_frames, image_second_chunk_frames
                ):

                    # add timestamp
                    if timestamp_format == "HHMMSS":
                        timestamp = time.strftime("%H:%M:%S", time.gmtime(round(image_second_chunk_frame)))
                    else:
                        timestamp = f"{image_second_chunk_frame:.2f}"
                    _input_id = tokenizer(timestamp, add_special_tokens=False).input_ids
                    new_input_ids += _input_id
                    if targets is not None:
                        if is_pretrain:
                            new_targets += _input_id
                        else:
                            new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(_input_id)

                    if vid_idx in contiguous_video_idxs:
                        new_input_ids += [IMG_START_ID]
                        if targets is not None:
                            if is_pretrain:
                                new_targets += [IMG_START_ID]
                            else:
                                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

                        if self.image_processor.resolution_type == "native":
                            resolution = f"{_video_grid_thw[0][1] * self.patch_size}*{_video_grid_thw[0][2] * self.patch_size}"
                            _input_id = tokenizer(resolution, add_special_tokens=False).input_ids
                            new_input_ids += _input_id
                            if targets is not None:
                                if is_pretrain:
                                    # new_targets += _input_id
                                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(_input_id)
                                else:
                                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(_input_id)

                            for _ in range(
                                _video_grid_thw[0][0]
                                * _video_grid_thw[0][1]
                                // self.spatial_merge_size
                            ):
                                image_token_length = (
                                    _video_grid_thw[0][2] // self.spatial_merge_size
                                )
                                image_indice_b = torch.zeros(
                                    1, image_token_length, dtype=torch.int64
                                )  # This will change in collate_fn
                                image_indice_s = (
                                    torch.arange(
                                        len(new_input_ids), len(new_input_ids) + image_token_length
                                    )
                                    .unsqueeze(0)
                                    .repeat(1, 1)
                                )
                                image_indice_b_s = torch.stack(
                                    [image_indice_b, image_indice_s], dim=0
                                )  # 2, num_image, image_length
                                image_indices.append(image_indice_b_s.view(2, -1))

                                new_input_ids += [IMG_CONTEXT_ID] * image_token_length
                                if targets is not None:
                                    new_targets += [
                                        GLOBAL_TOKEN.IGNORE_TOKEN_ID
                                    ] * image_token_length

                                new_input_ids += nl_tokens
                                if targets is not None:
                                    if is_pretrain:
                                        new_targets += nl_tokens
                                    else:
                                        new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(
                                            nl_tokens
                                        )

                        else:
                            image_token_length = (
                                _video_grid_thw[0][0]
                                * _video_grid_thw[0][1]
                                * _video_grid_thw[0][2]
                                // self.spatial_merge_size
                                // self.spatial_merge_size
                            )
                            image_indice_b = torch.zeros(
                                1, image_token_length, dtype=torch.int64
                            )  # This will change in collate_fn
                            image_indice_s = (
                                torch.arange(
                                    len(new_input_ids), len(new_input_ids) + image_token_length
                                )
                                .unsqueeze(0)
                                .repeat(1, 1)
                            )
                            image_indice_b_s = torch.stack(
                                [image_indice_b, image_indice_s], dim=0
                            )  # 2, num_image, image_length
                            image_indices.append(image_indice_b_s)

                            new_input_ids += [IMG_CONTEXT_ID] * image_token_length
                            if targets is not None:
                                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * image_token_length

                        new_input_ids += [IMG_END_ID]
                        if targets is not None:
                            if is_pretrain:
                                new_targets += [IMG_END_ID]
                            else:
                                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

                    if vid_idx in discrete_video_idxs:
                        raise NotImplementedError

                for audio_chunk_frame, audio_second_chunk_frame in zip(
                    audio_chunk_frames, audio_second_chunk_frames
                ):

                    # add timestamp
                    if timestamp_format == "HHMMSS":
                        timestamp = time.strftime("%H:%M:%S", time.gmtime(round(audio_second_chunk_frame)))
                    else:
                        timestamp = f"{audio_second_chunk_frame:.2f}"
                    _input_id = tokenizer(timestamp, add_special_tokens=False).input_ids
                    new_input_ids += _input_id
                    if targets is not None:
                        if is_pretrain:
                            new_targets += _input_id
                        else:
                            new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(_input_id)

                    if vid_idx in contiguous_video_idxs:
                        new_input_ids += [AUD_START_ID]
                        if targets is not None:
                            if is_pretrain:
                                new_targets += [AUD_START_ID]
                            else:
                                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

                        # audio_token_length = audio_chunk_frame.size(0)
                        # audio_token_length = audio_token_length_func(audio_chunk_frame.size(0))
                        audio_token_length = audio_token_length_func(audio_chunk_frame)
                        audio_indice_b = torch.zeros(
                            1, audio_token_length, dtype=torch.int64
                        )  # This will change in collate_fn
                        audio_indice_s = (
                            torch.arange(
                                len(new_input_ids), len(new_input_ids) + audio_token_length
                            )
                            .unsqueeze(0)
                            .repeat(1, 1)
                        )
                        audio_indice_b_s = torch.stack(
                            [audio_indice_b, audio_indice_s], dim=0
                        )  # 2, num_audio, audio_length
                        audio_indices.append(audio_indice_b_s)

                        new_input_ids += [AUD_CONTEXT_ID] * audio_token_length
                        if targets is not None:
                            new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * audio_token_length

                        new_input_ids += [AUD_END_ID]
                        if targets is not None:
                            if is_pretrain:
                                new_targets += [AUD_END_ID]
                            else:
                                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

                    if vid_idx in discrete_video_idxs:
                        raise NotImplementedError

            new_input_ids += [VID_END_ID]
            if targets is not None:
                if is_pretrain:
                    new_targets += [VID_END_ID]
                else:
                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            video_grid_thw.extend(_video_grid_thw)
            second_per_grids.extend(_second_per_grids)

            st = vid_pos + 1

        new_input_ids += input_ids[st:]
        if targets is not None:
            new_targets += targets[st:]

        input_ids = new_input_ids
        if targets is not None:
            targets = new_targets

        video_grid_thw = torch.tensor(video_grid_thw, dtype=torch.long)
        second_per_grids = torch.tensor(second_per_grids, dtype=torch.long)

        if targets is not None:
            return (
                input_ids,
                images,
                image_indices,
                audios,
                audio_indices,
                video_grid_thw,
                second_per_grids,
                targets,
            )

        if len(images) == 0:
            images = None
            image_indices = None

        else:
            images = torch.cat(images, dim=0)
            image_indices = torch.cat(image_indices, dim=1)

            image_indices = image_indices.contiguous().to(torch.cuda.current_device())
            if True:
                images = (
                    torch.tensor(images, dtype=torch.bfloat16)
                    .contiguous()
                    .to(torch.cuda.current_device())
                )

            else:
                images = (
                    torch.tensor(images, dtype=torch.float16)
                    .contiguous()
                    .to(torch.cuda.current_device())
                )

        if len(audios) == 0:
            audios = None
            audio_indices = None

        return (
            input_ids,
            images,
            image_indices,
            audios,
            audio_indices,
            video_grid_thw,
            second_per_grids,
        )




class YoutuVITAImageProcessor(BaseImageProcessor):
    model_input_names = ["images", "image_indices", "image_grid_thw"]
    valid_kwargs = YoutuVITAImagesKwargs

    def __init__(
        self,
        image_size=448,
        image_size_discrete=None,
        normalize_type="imagenet",
        resolution_type="dynamic",
        min_tile_grid=1,
        max_tile_grid=6,
        min_num_tokens=4,
        max_num_tokens=256,
        temporal_patch_size=1,
        spatial_merge_size=2,
        patch_size=14,
        vision_tokenizer_path=None,
        vision_tokenizer_type=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.image_size = image_size
        self.image_size_discrete = image_size_discrete
        self.resolution_type = resolution_type
        self.min_tile_grid = min_tile_grid
        self.max_tile_grid = max_tile_grid
        self.tile_image_size = image_size
        self.max_num_tokens = max_num_tokens
        self.min_num_tokens = min_num_tokens

        GLOBAL_TOKEN = get_token()
        if normalize_type == "imagenet":
            MEAN, STD = GLOBAL_TOKEN.IMAGENET_DEFAULT_MEAN, GLOBAL_TOKEN.IMAGENET_DEFAULT_STD
        elif normalize_type == "clip":
            MEAN, STD = GLOBAL_TOKEN.OPENAI_CLIP_MEAN, GLOBAL_TOKEN.OPENAI_CLIP_STD
        elif normalize_type == "siglip":
            MEAN, STD = GLOBAL_TOKEN.IMAGENET_STANDARD_MEAN, GLOBAL_TOKEN.IMAGENET_STANDARD_STD
        else:
            raise NotImplementedError(normalize_type)
        self.mean = MEAN
        self.std = STD

        if self.resolution_type == "anyres":
            raise NotImplemented
            self.grid_pinpoints = [
                (i, j)
                for i in range(min_tile_grid, max_tile_grid + 1)
                for j in range(min_tile_grid, max_tile_grid + 1)
            ]
            self.possible_resolutions = [
                [dim * self.tile_image_size for dim in pair] for pair in self.grid_pinpoints
            ]
            print(f"{self.grid_pinpoints=}")
            print(f"{self.possible_resolutions=}")

        if self.resolution_type == "dynamic":
            max_num = self.max_tile_grid
            min_num = self.min_tile_grid
            # calculate the existing image aspect ratio
            target_ratios = set(
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if i * j <= max_num and i * j >= min_num
            )
            self.target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
            self.possible_resolutions = [
                [dim * self.tile_image_size for dim in pair] for pair in self.target_ratios
            ]
            print(f"{self.target_ratios=}")
            print(f"{self.possible_resolutions=}")

        if self.resolution_type == "native":
            self.min_pixels = (patch_size * spatial_merge_size) ** 2 * min_num_tokens
            self.max_pixels = (patch_size * spatial_merge_size) ** 2 * max_num_tokens
            print(f"{self.min_pixels=} {self.max_pixels=}")

        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size

        self.vision_tokenizer = get_vision_tokenizer(
            vision_tokenizer_path,
            vision_tokenizer_type,
            rank=None,
        )

    def to_dict(self):
        output = super().to_dict()
        # Remove the non-serializable object before returning
        if "vision_tokenizer" in output:
            del output["vision_tokenizer"]
        return output

    def load_model(self):
        self.vision_tokenizer.load_model()

    def process_images_to_tensor(self, image_or_path_list):

        if isinstance(image_or_path_list[0], str):
            images = [PIL.Image.open(x).convert("RGB") for x in image_or_path_list]
        elif isinstance(image_or_path_list[0], PIL.Image.Image):
            images = [x.convert("RGB") for x in image_or_path_list]
        else:
            images = image_or_path_list

        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = PIL.Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = PIL.Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        image_tensor = torch.ones([len(images), 3, self.image_size, self.image_size])

        for i, image in enumerate(images):
            image = expand2square(image, tuple(int(x * 255) for x in self.mean))

            image = image.resize(
                (self.image_size, self.image_size), resample=PIL.Image.Resampling.BICUBIC
            )

            image = np.array(image, dtype=np.float32)
            image = image * 1.0 / 255.0

            mean = np.array(self.mean, dtype=image.dtype)
            std = np.array(self.std, dtype=image.dtype)
            image = (image - mean) / std

            image = torch.tensor(image, dtype=torch.float32)
            image = image.permute(2, 0, 1)

            image_tensor[i] = image

        return {
            "images": image_tensor,
            "image_height": self.image_size,
            "image_width": self.image_size,
        }
        return image_tensor, (self.image_size, self.image_size)

    def process_tensor_to_image(self, image_tensor):

        image_tensor = image_tensor.permute(1, 2, 0)
        image = image_tensor.numpy()

        mean = np.array(self.mean, dtype=image.dtype)
        std = np.array(self.std, dtype=image.dtype)
        image = image * std + mean
        image = image * 255
        image = image.astype(np.uint8)
        image = PIL.Image.fromarray(image)

        return image

    def process_image_to_tiles(self, image_or_path, **kwargs):
        if self.resolution_type == "anyres":
            return self.process_anyres(image_or_path)
        if self.resolution_type == "dynamic":
            return self.process_dynamic(image_or_path)
        if self.resolution_type == "native":
            return self.process_native(image_or_path, **kwargs)

        if isinstance(image_or_path, str):
            image = PIL.Image.open(image_or_path).convert("RGB")
        elif isinstance(image_or_path, PIL.Image.Image):
            image = image_or_path.convert("RGB")
        else:
            image = image_or_path

        return self.process_images_to_tensor([image])

    def process_image(self, image_or_path, is_discrete=False, is_contiguous=False, **kwargs):

        assert not (is_discrete and is_contiguous)
        assert is_discrete or is_contiguous

        if is_discrete and self.vision_tokenizer.tokenizer_discrete is not None:
            if isinstance(image_or_path, str):
                tokenizer_type = self.vision_tokenizer.tokenizer_discrete.tokenizer_type

                if "image_height" in kwargs and "image_width" in kwargs:
                    image_height = kwargs["image_height"]
                    image_width = kwargs["image_width"]
                    suffix = f"-{image_height=}-{image_width=}-{tokenizer_type}.json"

                else:
                    area = self.image_size_discrete * self.image_size_discrete
                    kwargs["area"] = area
                    suffix = f"-{area=}-{tokenizer_type}.json"

                cache_path = os.path.splitext(image_or_path)[0] + suffix
                image_data = load_cache(cache_path)
                if image_data:
                    return image_data

            image_data = self.vision_tokenizer.encode(
                image_or_path,
                is_discrete=is_discrete,
                is_contiguous=is_contiguous,
                **kwargs,
            )
            image_data["image_tokens"] = image_data["image_tokens"].tolist()

            if isinstance(image_or_path, str):
                save_cache(image_data, cache_path)

            return image_data

        if is_contiguous:
            resolution_type = kwargs.get("resolution_type", self.resolution_type)
            if self.resolution_type == "anyres":
                return self.process_anyres(image_or_path)
            if self.resolution_type == "dynamic":
                return self.process_dynamic(image_or_path)
            if self.resolution_type == "native":
                return self.process_native(image_or_path, **kwargs)

            if isinstance(image_or_path, str):
                image = PIL.Image.open(image_or_path).convert("RGB")
            elif isinstance(image_or_path, PIL.Image.Image):
                image = image_or_path.convert("RGB")
            else:
                image = image_or_path

            return self.process_images_to_tensor([image])

    def process_images(self, image_or_paths, is_discrete=False, is_contiguous=False, **kwargs):
        images = []
        best_width = None
        best_height = None
        for image_or_path in image_or_paths:
            image_data = self.process_image(
                image_or_path, is_discrete=is_discrete, is_contiguous=is_contiguous, **kwargs
            )
            _images = image_data["images"]
            best_width = image_data["image_width"]
            best_height = image_data["image_height"]
            images.append(_images)

        images = torch.cat(images, dim=0)
        return {
            "images": images,
            "image_height": best_height,
            "image_width": best_width,
        }
        return images, (best_width, best_height)

    def process_token_to_image(self, image_tokens, **kwargs):
        image_data = self.vision_tokenizer.decode(image_tokens, **kwargs)
        return image_data

    def process_anyres(self, image_or_path):
        if isinstance(image_or_path, str):
            image = PIL.Image.open(image_or_path).convert("RGB")
        elif isinstance(image_or_path, PIL.Image.Image):
            image = image_or_path.convert("RGB")
        else:
            image = image_or_path

        best_resolution = select_best_resolution(image.size, self.possible_resolutions)
        image_padded = resize_and_pad_image(image, best_resolution)
        patches = divide_to_patches(image_padded, self.tile_image_size)

        if best_resolution == (self.tile_image_size, self.tile_image_size):
            image_patches = [image]
        else:
            image_patches = [image] + patches

        image_patches, _ = self.process_images_to_tensor(image_patches)

        # print(f"image {image.size} best_resolution {best_resolution} image_padded {image_padded.size} patches {len(patches)} image_patches {image_patches.size()}")
        return {
            "images": image_patches,
            "image_height": best_resolution[1],
            "image_width": best_resolution[0],
        }

        return image_patches, best_resolution

    def process_dynamic(self, image_or_path):
        if isinstance(image_or_path, str):
            image = PIL.Image.open(image_or_path).convert("RGB")
        elif isinstance(image_or_path, PIL.Image.Image):
            image = image_or_path.convert("RGB")
        else:
            image = image_or_path

        image_patches, best_resolution = dynamic_preprocess(
            image,
            min_num=self.min_tile_grid,
            max_num=self.max_tile_grid,
            image_size=self.tile_image_size,
            use_thumbnail=True,
        )

        image_data = self.process_images_to_tensor(image_patches)
        image_patches = image_data["images"]

        # print(f"{image.size()=} {best_resolution=} {image_patches.size()=}")
        return {
            "images": image_patches,
            "image_height": best_resolution[1],
            "image_width": best_resolution[0],
        }

        return image_patches, best_resolution

    def process_native(self, image_or_path, **kwargs):
        if isinstance(image_or_path, str):
            image = PIL.Image.open(image_or_path).convert("RGB")
        elif isinstance(image_or_path, PIL.Image.Image):
            image = image_or_path.convert("RGB")
        else:
            image = image_or_path

        width, height = image.size

        min_pixels = kwargs.get("min_pixels", self.min_pixels)
        max_pixels = kwargs.get("max_pixels", self.max_pixels)

        factor = self.patch_size * self.spatial_merge_size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        # if height - factor > resized_height or width - factor > resized_width:
        #     print(f"Image is downsampled: {height=} {width=} {resized_height=} {resized_width=}", flush=True)

        # resized_height, resized_width = get_image_size_for_max_num_patches(
        #     height,
        #     width,
        #     patch_size=self.patch_size * self.spatial_merge_size,
        #     max_num_patches=self.max_num_tokens,
        # )

        image = image.resize((resized_width, resized_height), resample=PIL.Image.Resampling.BICUBIC)

        image = np.array(image, dtype=np.float32)
        image = image * 1.0 / 255.0

        mean = np.array(self.mean, dtype=image.dtype)
        std = np.array(self.std, dtype=image.dtype)
        image = (image - mean) / std

        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)

        # print(f"{image.size()=} {resized_width=} {resized_height=}")

        return {
            "images": image[None, ...],
            "image_height": resized_height,
            "image_width": resized_width,
        }

        return image[None, ...], (resized_width, resized_height)

    def convert_image_to_patches_with_pixel_shuffle(self, image):
        """
        image: (num_channels, image_height, image_width)
        patched_image: (_, patch_size * patch_size * num_channels)
        """

        patches = image[None, ...]
        _, num_channels, image_height, image_width = patches.shape

        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = image_height // self.patch_size, image_width // self.patch_size
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.spatial_merge_size,
            self.spatial_merge_size,
            self.patch_size,
            grid_w // self.spatial_merge_size,
            self.spatial_merge_size,
            self.patch_size,
        )
        patches = patches.permute(
            0, 3, 6, 4, 7, 2, 1, 5, 8
        )  # grid_t, grid_h // merge_size, grid_w // merge_size, merge_size, merge_size, channel, temporal_patch_size, patch_size, patch_size
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * self.temporal_patch_size * self.patch_size * self.patch_size,
        )

        return flatten_patches

    def get_image_grid_thw(self, images):

        image_grid_thw = []
        for image in images:
            assert isinstance(images, torch.Tensor)
            _, height, width = image.shape

            grid_t = 1 // self.temporal_patch_size
            grid_h = height // self.patch_size
            grid_w = width // self.patch_size

            image_grid_thw.append([grid_t, grid_h, grid_w])
        # image_grid_thw = torch.tensor(image_grid_thw, dtype=torch.int64)

        return image_grid_thw

    @property
    def is_discrete(self):
        return self.vision_tokenizer.is_discrete

    @property
    def is_contiguous(self):
        return self.vision_tokenizer.is_contiguous

    def get_resize_factor(self, image_or_path):

        if isinstance(image_or_path, str):
            image = PIL.Image.open(image_or_path).convert("RGB")
        elif isinstance(image_or_path, PIL.Image.Image):
            image = image_or_path.convert("RGB")
        else:
            image = image_or_path

        width, height = image.size
        image_data = self.process_image(image_or_path, is_contiguous=True)
        best_width = image_data["image_width"]
        best_height = image_data["image_height"]

        scale_x = 1.0 * best_width / width
        scale_y = 1.0 * best_height / height

        return scale_x, scale_y

    def add_image_input_contiguous(
        self,
        input_ids,
        image_or_paths,
        tokenizer,
        # image_token_length=256,
        targets=None,
        is_pretrain=False,
        **kwargs,
    ):
        GLOBAL_TOKEN = get_token()

        IMG_CONTEXT_ID = tokenizer(
            GLOBAL_TOKEN.IMG_CONTEXT_TOKEN, add_special_tokens=False
        ).input_ids
        IMG_START_ID = tokenizer(GLOBAL_TOKEN.IMG_START_TOKEN, add_special_tokens=False).input_ids
        IMG_END_ID = tokenizer(GLOBAL_TOKEN.IMG_END_TOKEN, add_special_tokens=False).input_ids

        PATCH_CONTEXT_ID = tokenizer(
            GLOBAL_TOKEN.PATCH_CONTEXT_TOKEN, add_special_tokens=False
        ).input_ids
        PATCH_START_ID = tokenizer(
            GLOBAL_TOKEN.PATCH_START_TOKEN, add_special_tokens=False
        ).input_ids
        PATCH_END_ID = tokenizer(GLOBAL_TOKEN.PATCH_END_TOKEN, add_special_tokens=False).input_ids

        IMG_TAG_ID = tokenizer(GLOBAL_TOKEN.IMG_TAG_TOKEN, add_special_tokens=False).input_ids

        assert len(IMG_CONTEXT_ID) == 1
        assert len(IMG_START_ID) == 1
        assert len(IMG_END_ID) == 1

        assert len(PATCH_CONTEXT_ID) == 1
        assert len(PATCH_START_ID) == 1
        assert len(PATCH_END_ID) == 1

        IMG_CONTEXT_ID = IMG_CONTEXT_ID[0]
        IMG_START_ID = IMG_START_ID[0]
        IMG_END_ID = IMG_END_ID[0]

        PATCH_CONTEXT_ID = PATCH_CONTEXT_ID[0]
        PATCH_START_ID = PATCH_START_ID[0]
        PATCH_END_ID = PATCH_END_ID[0]

        IMG_TAG_ID = IMG_TAG_ID[0]

        nl_tokens = tokenizer("\n", add_special_tokens=False).input_ids

        img_positions = [i for i, x in enumerate(input_ids) if x == IMG_TAG_ID]

        images = []
        image_indices = []
        image_grid_thw = []

        new_input_ids = []
        new_targets = []

        st = 0
        for img_idx, img_pos in enumerate(img_positions):
            (
                image_patches,
                (best_width, best_height),
            ) = self.process_image_to_tiles(image_or_paths[img_idx])
            # image_patches, _ = self.process_images_to_tensor([image_or_paths[img_idx]])

            _image_grid_thw = self.get_image_grid_thw(image_patches)

            if self.resolution_type == "native":
                images.append(
                    torch.cat(
                        [
                            self.convert_image_to_patches_with_pixel_shuffle(x)
                            for x in image_patches
                        ],
                        dim=0,
                    )
                )
            else:
                images.append(image_patches)

            new_input_ids += input_ids[st:img_pos]
            if targets is not None:
                new_targets += targets[st:img_pos]

            new_input_ids += [IMG_START_ID]
            if targets is not None:
                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            image_token_length = (
                _image_grid_thw[0][0]
                * _image_grid_thw[0][1]
                * _image_grid_thw[0][2]
                // self.spatial_merge_size
                // self.spatial_merge_size
            )
            image_indice_b = torch.zeros(
                1, image_token_length, dtype=torch.int64
            )  # This will change in collate_fn
            image_indice_s = (
                torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
                .unsqueeze(0)
                .repeat(1, 1)
            )
            image_indice_b_s = torch.stack(
                [image_indice_b, image_indice_s], dim=0
            )  # 2, num_image, image_length
            if self.resolution_type == "native":
                image_indices.append(image_indice_b_s.view(2, -1))
            else:
                image_indices.append(image_indice_b_s)

            new_input_ids += [IMG_CONTEXT_ID] * image_token_length
            if targets is not None:
                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * image_token_length

            new_input_ids += [IMG_END_ID]
            if targets is not None:
                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            if len(image_patches) > 1:
                for _ in range(0, best_height, self.tile_image_size):
                    new_input_ids += nl_tokens
                    if targets is not None:
                        new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(nl_tokens)

                    for _ in range(0, best_width, self.tile_image_size):
                        new_input_ids += [PATCH_START_ID]
                        if targets is not None:
                            new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

                        image_indice_b = torch.zeros(
                            1, image_token_length, dtype=torch.int64
                        )  # This will change in collate_fn
                        image_indice_s = (
                            torch.arange(
                                len(new_input_ids), len(new_input_ids) + image_token_length
                            )
                            .unsqueeze(0)
                            .repeat(1, 1)
                        )
                        image_indice_b_s = torch.stack(
                            [image_indice_b, image_indice_s], dim=0
                        )  # 2, num_image, image_length
                        image_indices.append(image_indice_b_s)

                        new_input_ids += [PATCH_CONTEXT_ID] * image_token_length
                        if targets is not None:
                            new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * image_token_length

                        new_input_ids += [PATCH_END_ID]
                        if targets is not None:
                            new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            image_grid_thw.extend(_image_grid_thw)
            st = img_pos + 1

        new_input_ids += input_ids[st:]
        if targets is not None:
            new_targets += targets[st:]

        input_ids = new_input_ids
        if targets is not None:
            targets = new_targets

        image_grid_thw = torch.tensor(image_grid_thw, dtype=torch.long)

        if targets is not None:
            return input_ids, images, image_indices, image_grid_thw, targets

        images = torch.cat(images, dim=0)
        image_indices = torch.cat(image_indices, dim=1)

        image_indices = image_indices.contiguous().to(torch.cuda.current_device())
        if True:
            images = (
                torch.tensor(images, dtype=torch.bfloat16)
                .contiguous()
                .to(torch.cuda.current_device())
            )

        else:
            images = (
                torch.tensor(images, dtype=torch.float16)
                .contiguous()
                .to(torch.cuda.current_device())
            )

        return input_ids, images, image_indices, image_grid_thw

    def add_image_input_discrete_and_contiguous(
        self,
        input_ids,
        image_or_paths,
        tokenizer,
        # image_token_length=256,
        use_tile=True,
        targets=None,
        is_pretrain=False,
        **kwargs,
    ):

        GLOBAL_TOKEN = get_token()

        IMG_CONTEXT_ID = tokenizer(
            GLOBAL_TOKEN.IMG_CONTEXT_TOKEN, add_special_tokens=False
        ).input_ids
        IMG_START_ID = tokenizer(GLOBAL_TOKEN.IMG_START_TOKEN, add_special_tokens=False).input_ids
        IMG_END_ID = tokenizer(GLOBAL_TOKEN.IMG_END_TOKEN, add_special_tokens=False).input_ids

        PATCH_CONTEXT_ID = tokenizer(
            GLOBAL_TOKEN.PATCH_CONTEXT_TOKEN, add_special_tokens=False
        ).input_ids
        PATCH_START_ID = tokenizer(
            GLOBAL_TOKEN.PATCH_START_TOKEN, add_special_tokens=False
        ).input_ids
        PATCH_END_ID = tokenizer(GLOBAL_TOKEN.PATCH_END_TOKEN, add_special_tokens=False).input_ids

        IMG_TAG_ID = tokenizer(GLOBAL_TOKEN.IMG_TAG_TOKEN, add_special_tokens=False).input_ids

        IMG_FIRST_ID = tokenizer.convert_tokens_to_ids("<|vision_0|>")
        IMG_EOL_ID = tokenizer.convert_tokens_to_ids("<|vision_eol|>")

        assert len(IMG_CONTEXT_ID) == 1
        assert len(IMG_START_ID) == 1
        assert len(IMG_END_ID) == 1

        assert len(PATCH_CONTEXT_ID) == 1
        assert len(PATCH_START_ID) == 1
        assert len(PATCH_END_ID) == 1

        IMG_CONTEXT_ID = IMG_CONTEXT_ID[0]
        IMG_START_ID = IMG_START_ID[0]
        IMG_END_ID = IMG_END_ID[0]

        PATCH_CONTEXT_ID = PATCH_CONTEXT_ID[0]
        PATCH_START_ID = PATCH_START_ID[0]
        PATCH_END_ID = PATCH_END_ID[0]

        IMG_TAG_ID = IMG_TAG_ID[0]

        nl_tokens = tokenizer("\n", add_special_tokens=False).input_ids

        img_positions = [i for i, x in enumerate(input_ids) if x == IMG_TAG_ID]

        images = []
        image_indices = []
        image_grid_thw = []

        new_input_ids = []
        new_targets = []

        st = 0
        for img_idx, img_pos in enumerate(img_positions):

            new_input_ids += input_ids[st:img_pos]
            if targets is not None:
                new_targets += targets[st:img_pos]

            if use_tile:

                (
                    image_patches,
                    (best_width, best_height),
                ) = self.process_image_to_tiles(image_or_paths[img_idx])
                patch_idx = 0
            else:
                image_patches, _ = self.process_images_to_tensor([image_or_paths[img_idx]])

            _image_grid_thw = self.get_image_grid_thw(image_patches)

            if self.resolution_type == "native":
                images.append(
                    torch.cat(
                        [
                            self.convert_image_to_patches_with_pixel_shuffle(x)
                            for x in image_patches
                        ],
                        dim=0,
                    )
                )
            else:
                images.append(image_patches)

            # --------------------------------------------------------------------------
            # add discrete

            if use_tile:
                image_patch = self.process_tensor_to_image(image_patches[patch_idx])
                patch_idx += 1
            else:
                image_patch = image_or_paths[img_idx]
            image_data = self.process_image(image_patch, is_discrete=True)
            image_tokens = image_data["image_tokens"]
            # h, w = image_tokens.shape
            h, w = len(image_token), len(image_token[0])
            # image_tokens = image_tokens.tolist()
            image_input_ids = []
            for _h in range(h):
                for _w in range(w):
                    image_input_ids.append(image_tokens[_h][_w] + IMG_FIRST_ID)
                if _h < h - 1:
                    image_input_ids += [IMG_EOL_ID]

            size_input_ids = tokenizer(f"{h}*{w}", add_special_tokens=False).input_ids

            new_input_ids += [IMG_START_ID]
            if targets is not None:
                if is_pretrain:
                    new_targets += [IMG_START_ID]
                else:
                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            new_input_ids += size_input_ids
            if targets is not None:
                if is_pretrain:
                    new_targets += size_input_ids
                else:
                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(size_input_ids)

            new_input_ids += image_input_ids
            if targets is not None:
                if is_pretrain:
                    new_targets += image_input_ids
                else:
                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(image_input_ids)

            new_input_ids += [IMG_END_ID]
            if targets is not None:
                if is_pretrain:
                    new_targets += [IMG_END_ID]
                else:
                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            # --------------------------------------------------------------------------
            # add contiguous

            new_input_ids += [IMG_START_ID]
            if targets is not None:
                if is_pretrain:
                    new_targets += [IMG_START_ID]
                else:
                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            image_token_length = (
                _image_grid_thw[0][0]
                * _image_grid_thw[0][1]
                * _image_grid_thw[0][2]
                // self.spatial_merge_size
                // self.spatial_merge_size
            )
            image_indice_b = torch.zeros(
                1, image_token_length, dtype=torch.int64
            )  # This will change in collate_fn
            image_indice_s = (
                torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
                .unsqueeze(0)
                .repeat(1, 1)
            )
            image_indice_b_s = torch.stack(
                [image_indice_b, image_indice_s], dim=0
            )  # 2, num_image, image_length
            if self.resolution_type == "native":
                image_indices.append(image_indice_b_s.view(2, -1))
            else:
                image_indices.append(image_indice_b_s)

            new_input_ids += [IMG_CONTEXT_ID] * image_token_length
            if targets is not None:
                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * image_token_length

            new_input_ids += [IMG_END_ID]
            if targets is not None:
                if is_pretrain:
                    new_targets += [IMG_END_ID]
                else:
                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            if len(image_patches) > 1:
                for _ in range(0, best_height, self.tile_image_size):
                    new_input_ids += nl_tokens
                    if targets is not None:
                        if is_pretrain:
                            new_targets += nl_tokens
                        else:
                            new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(nl_tokens)

                    for _ in range(0, best_width, self.tile_image_size):

                        # --------------------------------------------------------------------------
                        # add discrete

                        image_patch = self.process_tensor_to_image(image_patches[patch_idx])
                        patch_idx += 1
                        image_data = self.process_image(image_patch, is_discrete=True)
                        image_tokens = image_data["image_tokens"]
                        # h, w = image_tokens.shape
                        h, w = len(image_token), len(image_token[0])
                        # image_tokens = image_tokens.tolist()
                        image_input_ids = []
                        for _h in range(h):
                            for _w in range(w):
                                image_input_ids.append(image_tokens[_h][_w] + IMG_FIRST_ID)
                            if _h < h - 1:
                                image_input_ids += [IMG_EOL_ID]

                        size_input_ids = tokenizer(f"{h}*{w}", add_special_tokens=False).input_ids

                        new_input_ids += [PATCH_START_ID]
                        if targets is not None:
                            if is_pretrain:
                                new_targets += [PATCH_START_ID]
                            else:
                                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

                        new_input_ids += size_input_ids
                        if targets is not None:
                            if is_pretrain:
                                new_targets += size_input_ids
                            else:
                                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(size_input_ids)

                        new_input_ids += image_input_ids
                        if targets is not None:
                            if is_pretrain:
                                new_targets += image_input_ids
                            else:
                                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(image_input_ids)

                        new_input_ids += [PATCH_END_ID]
                        if targets is not None:
                            if is_pretrain:
                                new_targets += [PATCH_END_ID]
                            else:
                                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

                        # --------------------------------------------------------------------------
                        # add contiguous

                        new_input_ids += [PATCH_START_ID]
                        if targets is not None:
                            if is_pretrain:
                                new_targets += [PATCH_START_ID]
                            else:
                                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

                        image_indice_b = torch.zeros(
                            1, image_token_length, dtype=torch.int64
                        )  # This will change in collate_fn
                        image_indice_s = (
                            torch.arange(
                                len(new_input_ids), len(new_input_ids) + image_token_length
                            )
                            .unsqueeze(0)
                            .repeat(1, 1)
                        )
                        image_indice_b_s = torch.stack(
                            [image_indice_b, image_indice_s], dim=0
                        )  # 2, num_image, image_length
                        image_indices.append(image_indice_b_s)

                        new_input_ids += [PATCH_CONTEXT_ID] * image_token_length
                        if targets is not None:
                            new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * image_token_length

                        new_input_ids += [PATCH_END_ID]
                        if targets is not None:
                            if is_pretrain:
                                new_targets += [PATCH_END_ID]
                            else:
                                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            image_grid_thw.extend(_image_grid_thw)

            # --------------------------------------------------------------------------

            st = img_pos + 1

        new_input_ids += input_ids[st:]
        if targets is not None:
            new_targets += targets[st:]

        input_ids = new_input_ids
        if targets is not None:
            targets = new_targets

        image_grid_thw = torch.tensor(image_grid_thw, dtype=torch.long)

        if targets is not None:
            return input_ids, images, image_indices, image_grid_thw, targets

        images = torch.cat(images, dim=0)
        image_indices = torch.cat(image_indices, dim=1)

        image_indices = image_indices.contiguous().to(torch.cuda.current_device())
        if True:
            images = (
                torch.tensor(images, dtype=torch.bfloat16)
                .contiguous()
                .to(torch.cuda.current_device())
            )

        else:
            images = (
                torch.tensor(images, dtype=torch.float16)
                .contiguous()
                .to(torch.cuda.current_device())
            )

        return input_ids, images, image_indices, image_grid_thw

    def add_image_input_discrete_or_contiguous(
        self,
        input_ids,
        image_or_paths,
        tokenizer,
        # image_token_length=256,
        # use_tile=True,
        discrete_image_idxs=[],
        contiguous_image_idxs=[],
        targets=None,
        is_pretrain=False,
        **kwargs,
    ):

        GLOBAL_TOKEN = get_token()

        IMG_CONTEXT_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.IMG_CONTEXT_TOKEN)
        IMG_START_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.IMG_START_TOKEN)
        IMG_END_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.IMG_END_TOKEN)
        IMG_TAG_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.IMG_TAG_TOKEN)

        if self.resolution_type == "native":
            pass
        else:
            PATCH_CONTEXT_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.PATCH_CONTEXT_TOKEN)
            PATCH_START_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.PATCH_START_TOKEN)
            PATCH_END_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.PATCH_END_TOKEN)

        if self.vision_tokenizer.first_vision_token is not None:
            IMG_FIRST_ID = tokenizer.convert_tokens_to_ids(self.vision_tokenizer.first_vision_token)
            IMG_EOL_ID = tokenizer.convert_tokens_to_ids("<|vision_eol|>")

        nl_tokens = tokenizer("\n", add_special_tokens=False).input_ids

        img_positions = [i for i, x in enumerate(input_ids) if x == IMG_TAG_ID]

        if not discrete_image_idxs and not contiguous_image_idxs:
            contiguous_image_idxs = list(range(len(image_or_paths)))

        images = []
        image_indices = []
        image_grid_thw = []

        new_input_ids = []
        new_targets = []

        st = 0
        for img_idx, img_pos in enumerate(img_positions):

            new_input_ids += input_ids[st:img_pos]
            if targets is not None:
                new_targets += targets[st:img_pos]

            # --------------------------------------------------------------------------
            # add discrete
            if img_idx in discrete_image_idxs:
                assert self.resolution_type == "native"
                image_data = self.process_image(
                    image_or_paths[img_idx],
                    is_contiguous=True,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels // (self.spatial_merge_size * self.spatial_merge_size * 8 * 8),
                )
                image_height = image_data["image_height"]
                image_width = image_data["image_width"]
                image_data = self.process_image(
                    image_or_paths[img_idx],
                    is_discrete=True,
                    image_height=image_height,
                    image_width=image_width,
                )
                image_tokens = image_data["image_tokens"]
                image_height = image_data["image_height"]
                image_width = image_data["image_width"]

                resolution = f"{image_height}*{image_width}"
                size_input_ids = tokenizer(resolution, add_special_tokens=False).input_ids

                image_input_ids = []
                for image_token in image_tokens:
                    image_input_ids += [_ + IMG_FIRST_ID for _ in image_token]
                    image_input_ids += [IMG_EOL_ID]

                new_input_ids += [IMG_START_ID]
                if targets is not None:
                    new_targets += [IMG_START_ID]

                new_input_ids += size_input_ids
                if targets is not None:
                    new_targets += size_input_ids

                new_input_ids += [IMG_EOL_ID]
                if targets is not None:
                    new_targets += [IMG_EOL_ID]

                new_input_ids += image_input_ids
                if targets is not None:
                    new_targets += image_input_ids

                new_input_ids += [IMG_END_ID]
                if targets is not None:
                    new_targets += [IMG_END_ID]

            # --------------------------------------------------------------------------
            # add contiguous
            if img_idx in contiguous_image_idxs:

                image_data = self.process_image(image_or_paths[img_idx], is_contiguous=True)
                image_patches = image_data["images"]
                best_width = image_data["image_width"]
                best_height = image_data["image_height"]

                _image_grid_thw = self.get_image_grid_thw(image_patches)
                image_grid_thw.extend(_image_grid_thw)

                if self.resolution_type == "native":
                    images.append(
                        torch.cat(
                            [
                                self.convert_image_to_patches_with_pixel_shuffle(x)
                                for x in image_patches
                            ],
                            dim=0,
                        )
                    )
                else:
                    images.append(image_patches)

                new_input_ids += [IMG_START_ID]
                if targets is not None:
                    if is_pretrain:
                        new_targets += [IMG_START_ID]
                    else:
                        new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

                if self.resolution_type == "native":
                    resolution = f"{_image_grid_thw[0][1] * self.patch_size}*{_image_grid_thw[0][2] * self.patch_size}"
                    size_input_id = tokenizer(resolution, add_special_tokens=False).input_ids
                    new_input_ids += size_input_id
                    if targets is not None:
                        if is_pretrain:
                            # new_targets += size_input_id
                            new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(size_input_id)
                        else:
                            new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(size_input_id)

                    new_input_ids += nl_tokens
                    if targets is not None:
                        if is_pretrain:
                            new_targets += [IMG_EOL_ID]
                        else:
                            new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(nl_tokens)

                    for _h in range(
                        _image_grid_thw[0][0] * _image_grid_thw[0][1] // self.spatial_merge_size
                    ):
                        image_token_length = _image_grid_thw[0][2] // self.spatial_merge_size
                        image_indice_b = torch.zeros(
                            1, image_token_length, dtype=torch.int64
                        )  # This will change in collate_fn
                        image_indice_s = (
                            torch.arange(
                                len(new_input_ids), len(new_input_ids) + image_token_length
                            )
                            .unsqueeze(0)
                            .repeat(1, 1)
                        )
                        image_indice_b_s = torch.stack(
                            [image_indice_b, image_indice_s], dim=0
                        )  # 2, num_image, image_length
                        image_indices.append(image_indice_b_s.view(2, -1))

                        new_input_ids += [IMG_CONTEXT_ID] * image_token_length
                        if targets is not None:
                            new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * image_token_length

                        new_input_ids += nl_tokens
                        if targets is not None:
                            if is_pretrain:
                                new_targets += nl_tokens
                            else:
                                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(nl_tokens)

                else:
                    image_token_length = (
                        _image_grid_thw[0][0]
                        * _image_grid_thw[0][1]
                        * _image_grid_thw[0][2]
                        // self.spatial_merge_size
                        // self.spatial_merge_size
                    )
                    image_indice_b = torch.zeros(
                        1, image_token_length, dtype=torch.int64
                    )  # This will change in collate_fn
                    image_indice_s = (
                        torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
                        .unsqueeze(0)
                        .repeat(1, 1)
                    )
                    image_indice_b_s = torch.stack(
                        [image_indice_b, image_indice_s], dim=0
                    )  # 2, num_image, image_length
                    image_indices.append(image_indice_b_s)

                    new_input_ids += [IMG_CONTEXT_ID] * image_token_length
                    if targets is not None:
                        new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * image_token_length

                new_input_ids += [IMG_END_ID]
                if targets is not None:
                    if is_pretrain:
                        new_targets += [IMG_END_ID]
                    else:
                        new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

                if len(image_patches) > 1:
                    for _ in range(0, best_height, self.tile_image_size):
                        new_input_ids += nl_tokens
                        if targets is not None:
                            if is_pretrain:
                                new_targets += nl_tokens
                            else:
                                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(nl_tokens)

                        for _ in range(0, best_width, self.tile_image_size):
                            new_input_ids += [PATCH_START_ID]
                            if targets is not None:
                                if is_pretrain:
                                    new_targets += [PATCH_START_ID]
                                else:
                                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

                            image_indice_b = torch.zeros(
                                1, image_token_length, dtype=torch.int64
                            )  # This will change in collate_fn
                            image_indice_s = (
                                torch.arange(
                                    len(new_input_ids), len(new_input_ids) + image_token_length
                                )
                                .unsqueeze(0)
                                .repeat(1, 1)
                            )
                            image_indice_b_s = torch.stack(
                                [image_indice_b, image_indice_s], dim=0
                            )  # 2, num_image, image_length
                            image_indices.append(image_indice_b_s)

                            new_input_ids += [PATCH_CONTEXT_ID] * image_token_length
                            if targets is not None:
                                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * image_token_length

                            new_input_ids += [PATCH_END_ID]
                            if targets is not None:
                                if is_pretrain:
                                    new_targets += [PATCH_END_ID]
                                else:
                                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            st = img_pos + 1

        new_input_ids += input_ids[st:]
        if targets is not None:
            new_targets += targets[st:]

        input_ids = new_input_ids
        if targets is not None:
            targets = new_targets

        image_grid_thw = torch.tensor(image_grid_thw, dtype=torch.long)

        if targets is not None:
            return input_ids, images, image_indices, image_grid_thw, targets

        images = torch.cat(images, dim=0)
        image_indices = torch.cat(image_indices, dim=1)

        image_indices = image_indices.contiguous().to(torch.cuda.current_device())
        if True:
            images = (
                torch.tensor(images, dtype=torch.bfloat16)
                .contiguous()
                .to(torch.cuda.current_device())
            )

        else:
            images = (
                torch.tensor(images, dtype=torch.float16)
                .contiguous()
                .to(torch.cuda.current_device())
            )

        return input_ids, images, image_indices, image_grid_thw

    def add_image_input_contiguous_to_discrete(
        self,
        input_ids,
        image_or_paths,
        tokenizer,
        targets=None,
        is_pretrain=False,
        **kwargs,
    ):

        assert self.resolution_type == "native"
        GLOBAL_TOKEN = get_token()

        IMG_CONTEXT_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.IMG_CONTEXT_TOKEN)
        IMG_START_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.IMG_START_TOKEN)
        IMG_END_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.IMG_END_TOKEN)
        IMG_TAG_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.IMG_TAG_TOKEN)

        if targets is not None:
            IMG_FIRST_ID = tokenizer.convert_tokens_to_ids(self.vision_tokenizer.first_vision_token)
            IMG_EOL_ID = tokenizer.convert_tokens_to_ids("<|vision_eol|>")

        nl_tokens = tokenizer("\n", add_special_tokens=False).input_ids

        img_positions = [i for i, x in enumerate(input_ids) if x == IMG_TAG_ID]

        images = []
        image_indices = []
        image_grid_thw = []

        new_input_ids = []
        new_targets = []

        st = 0
        for img_idx, img_pos in enumerate(img_positions):

            new_input_ids += input_ids[st:img_pos]
            if targets is not None:
                new_targets += targets[st:img_pos]

            image_data = self.process_image(image_or_paths[img_idx], is_contiguous=True)
            image_patches = image_data["images"]
            best_width = image_data["image_width"]
            best_height = image_data["image_height"]

            _image_grid_thw = self.get_image_grid_thw(image_patches)
            image_grid_thw.extend(_image_grid_thw)

            images.append(
                torch.cat(
                    [self.convert_image_to_patches_with_pixel_shuffle(x) for x in image_patches],
                    dim=0,
                )
            )

            if targets is not None:
                image_data = self.process_image(
                    image_or_paths[img_idx],
                    is_discrete=True,
                    image_height=best_height // self.spatial_merge_size,
                    image_width=best_width // self.spatial_merge_size,
                )
                image_tokens = image_data["image_tokens"]
                assert (
                    len(image_tokens) == _image_grid_thw[0][1] // self.spatial_merge_size
                ), f"{len(image_tokens)=} {_image_grid_thw=} {best_width=} {best_height}"
                assert (
                    len(image_tokens[0]) == _image_grid_thw[0][2] // self.spatial_merge_size
                ), f"{len(image_tokens[0])=} {_image_grid_thw=} {best_width=} {best_height}"

            resolution = f"{_image_grid_thw[0][1] * self.patch_size}*{_image_grid_thw[0][2] * self.patch_size}"
            size_input_ids = tokenizer(resolution, add_special_tokens=False).input_ids

            new_input_ids += [IMG_START_ID]
            if targets is not None:
                if is_pretrain:
                    new_targets += [IMG_START_ID]
                else:
                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            new_input_ids += size_input_ids
            if targets is not None:
                if is_pretrain:
                    # new_targets += size_input_ids
                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(size_input_ids)
                else:
                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(size_input_ids)

            new_input_ids += nl_tokens
            if targets is not None:
                if is_pretrain:
                    new_targets += [IMG_EOL_ID]
                else:
                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(nl_tokens)

            for image_token in image_tokens:
                image_token_length = _image_grid_thw[0][2] // self.spatial_merge_size
                image_indice_b = torch.zeros(
                    1, image_token_length, dtype=torch.int64
                )  # This will change in collate_fn
                image_indice_s = (
                    torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
                    .unsqueeze(0)
                    .repeat(1, 1)
                )
                image_indice_b_s = torch.stack(
                    [image_indice_b, image_indice_s], dim=0
                )  # 2, num_image, image_length
                image_indices.append(image_indice_b_s.view(2, -1))

                new_input_ids += [IMG_CONTEXT_ID] * image_token_length
                if targets is not None:
                    new_targets += [_ + IMG_FIRST_ID for _ in image_token]

                new_input_ids += nl_tokens
                if targets is not None:
                    if is_pretrain:
                        new_targets += [IMG_EOL_ID]
                    else:
                        new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(nl_tokens)

            new_input_ids += [IMG_END_ID]
            if targets is not None:
                if is_pretrain:
                    new_targets += [IMG_END_ID]
                else:
                    new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]

            st = img_pos + 1

        new_input_ids += input_ids[st:]
        if targets is not None:
            new_targets += targets[st:]

        input_ids = new_input_ids
        if targets is not None:
            targets = new_targets

        image_grid_thw = torch.tensor(image_grid_thw, dtype=torch.long)

        if targets is not None:
            return input_ids, images, image_indices, image_grid_thw, targets

        images = torch.cat(images, dim=0)
        image_indices = torch.cat(image_indices, dim=1)

        image_indices = image_indices.contiguous().to(torch.cuda.current_device())
        if True:
            images = (
                torch.tensor(images, dtype=torch.bfloat16)
                .contiguous()
                .to(torch.cuda.current_device())
            )

        else:
            images = (
                torch.tensor(images, dtype=torch.float16)
                .contiguous()
                .to(torch.cuda.current_device())
            )

        return input_ids, images, image_indices, image_grid_thw




# https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip2/image_processing_siglip2.py
def get_image_size_for_max_num_patches(
    image_height: int, image_width: int, patch_size: int, max_num_patches: int, eps: float = 1e-5
) -> tuple[int, int]:
    """
    Determine image size based on max number of patches, ensure dimensions are divisible by patch size and image is at least 1 patch.

    Args:
        image_height (`int`):
            Original image height.
        image_width (`int`):
            Original image width.
        patch_size (`int`):
            Patch size for processing.
        max_num_patches (`int`):
            Maximum number of patches.
        eps (`float`):
            Small threshold for binary search.

    Returns:
        Tuple: (target_height, target_width)
    """

    def get_scaled_image_size(scale: float, size: int, patch_size: int) -> int:
        scaled_size = size * scale
        scaled_size = (
            math.ceil(scaled_size / patch_size) * patch_size
        )  # make divisible by patch_size
        scaled_size = max(patch_size, scaled_size)  # ensure at least 1 patch
        return int(scaled_size)

    # Binary search for optimal scale
    scale_min, scale_max = eps / 10, 100.0
    while (scale_max - scale_min) >= eps:
        scale = (scale_min + scale_max) / 2
        target_height = get_scaled_image_size(scale, image_height, patch_size)
        target_width = get_scaled_image_size(scale, image_width, patch_size)
        num_patches = (target_height / patch_size) * (target_width / patch_size)

        if num_patches <= max_num_patches:
            scale_min = scale
        else:
            scale_max = scale

    scale = scale_min
    target_height = get_scaled_image_size(scale, image_height, patch_size)
    target_width = get_scaled_image_size(scale, image_width, patch_size)
    return target_height, target_width



class YoutuVITAProcessor(ProcessorMixin):
    def __init__(
        self, image_processor=None, video_processor=None, feature_extractor=None, tokenizer=None, chat_template=None
    ):
        super().__init__(image_processor, video_processor, feature_extractor, tokenizer, chat_template=chat_template)

        audio_processor = feature_extractor
        self.audio_processor = audio_processor

        video_processor.image_processor = image_processor
        video_processor.audio_processor = audio_processor

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        images_or_paths: ImageInput | None = None,
        videos_or_paths: VideoInput | None = None,
        audios_or_paths: AudioInput | None = None,
        **kwargs: Unpack[YoutuVITAProcessorKwargs],
    ) -> BatchFeature:
        print(f"{text=}")
        print(f"{images_or_paths=}")
        print(f"{videos_or_paths=}")
        print(f"{audios_or_paths=}")
        print(f"{kwargs=}")

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        output_kwargs = self._merge_kwargs(
            YoutuVITAProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        print(f"{output_kwargs=}")

        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        print(f"{texts_inputs=}")
        input_ids = texts_inputs["input_ids"]

        images_inputs = {}
        videos_inputs = {}
        audio_inputs = {}

        if audios_or_paths:
            input_ids, audios, audio_indices = self.audio_processor.add_audio_input_discrete_or_contiguous(
                input_ids,
                audios_or_paths,
                self.tokenizer,
                discrete_audio_idxs=kwargs.get("discrete_audio_idxs", []),
                **output_kwargs["audio_kwargs"],
            )

            audio_seqlens = [len(x) for x in audios]

            print(f"{audios_or_paths=} {len(input_ids)=} {len(audios)=} {sum(x.abs().sum() for x in audios)=} {len(audio_indices)=}")

            audio_inputs["audios"] = audios
            audio_inputs["audio_indices"] = audio_indices
            # audio_inputs["audio_feature_lengths"] = audio_seqlens

        if images_or_paths:
            input_ids, images, image_indices, image_grid_thw = self.image_processor.add_image_input_discrete_or_contiguous(
                input_ids,
                images_or_paths,
                self.tokenizer,
                **output_kwargs["images_kwargs"],
            )
            print(f"{images_or_paths=} {len(input_ids)=} {images.size()=} {image_indices.size()=} {image_grid_thw=}")

            images_inputs["images"] = images
            images_inputs["image_indices"] = image_indices
            images_inputs["image_grid_thw"] = image_grid_thw

        if videos_or_paths:
            (
                input_ids,
                images,
                image_indices,
                audios,
                audio_indices,
                image_grid_thw,
                second_per_grids,
                # ) = self.video_processor.add_video_input_contiguous(
            ) = self.video_processor.add_video_input_discrete_or_contiguous(
                input_ids,
                videos_or_paths,
                self.tokenizer,
                **output_kwargs["videos_kwargs"],
            )
            if images is not None:
                print(f"{len(input_ids)=} {images.size()=} {image_indices.size()=} {image_grid_thw=}")
            if audios is not None:
                print(f"{len(input_ids)=} {len(audios)=} {[x.size() for x in audios]=} {len(audio_indices)=}")

            if audios is None:
                audio_seqlens = None
            else:
                audio_seqlens = [len(x) for x in audios]
            videos_inputs["images"] = images
            videos_inputs["image_indices"] = image_indices
            videos_inputs["audios"] = audios
            videos_inputs["audio_indices"] = audio_indices
            videos_inputs["image_grid_thw"] = image_grid_thw
            # videos_inputs["second_per_grids"] = second_per_grids

        input_ids = torch.tensor([input_ids], dtype=torch.long)
        texts_inputs["input_ids"] = input_ids
        texts_inputs["attention_mask"] = torch.ones_like(input_ids)

        return BatchFeature(
            data={**texts_inputs, **images_inputs, **videos_inputs, **audio_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        is_batched = False
        if isinstance(conversations[0], dict):
            conversations = [conversations]
            is_batched = True

        if is_batched:
            conversations = conversations[0]

        return super().apply_chat_template(conversations, chat_template, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        video_processor_input_names = self.video_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + feature_extractor_input_names
                + image_processor_input_names
                + video_processor_input_names
                + ["feature_attention_mask"]
                + ["video_second_per_grid"]
            )
        )


__all__ = [
    "YoutuVITAConfig",
    "YoutuVITAPreTrainedModel",
    "YoutuVITAModel",
    "YoutuVITAForCausalLM",
    "YoutuVITAProcessor",
    "YoutuVITAImageProcessor",
    "YoutuVITAVideoProcessor",
    "YoutuVITAFeatureExtractor",
]
