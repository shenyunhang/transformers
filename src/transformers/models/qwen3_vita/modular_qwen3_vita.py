
"""PyTorch Qwen3-VITA model."""

import mimetypes
import os
import time
import uuid
from typing import Any, Callable, Optional, Tuple, Union
import math

import numpy as np
import PIL.Image
import torch
from torch import nn

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...generation import GenerationMixin
from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ImageInput
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs, AudioKwargs, ImagesKwargs
from ...tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput
from ...utils import is_decord_available, is_flash_attn_2_available, is_torchaudio_available, logging
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import VideoInput
from ..siglip2.configuration_siglip2 import Siglip2VisionConfig
from ..qwen3.configuration_qwen3 import Qwen3Config
from ..qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3Model,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb

if is_decord_available():
    import decord

if is_torchaudio_available():
    import torchaudio

from funasr.frontends.wav_frontend import WavFrontend
from funasr.utils.load_utils import extract_fbank

is_aiter_available = False

logger = logging.get_logger(__name__)


class Qwen3VITAAudioConfig(PreTrainedConfig):
    r"""
    Qwen3VITAAudioConfig
    """
    model_type = "qwen3_vita_audio"
    base_config_key = "audio_config"

    def __init__(
        self,
        hidden_size=512,
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
        temporal_merge_size=1,
        out_hidden_size=4608,
        merger_hidden_size=4608,
        # CNN
        num_mel_bins=128,
        downsample_hidden_size=512,
        n_window=50,
        n_window_infer=800,
        conv_chunksize=500,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # SANM
        self.input_size = input_size
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

        self.hidden_size = hidden_size
        self.temporal_merge_size = temporal_merge_size
        self.out_hidden_size = out_hidden_size
        self.merger_hidden_size = merger_hidden_size

        # CNN
        self.downsample_hidden_size = downsample_hidden_size
        self.num_mel_bins = num_mel_bins
        self.n_window = n_window
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.num_hidden_layers = 0


class Qwen3VITAVisionConfig(PreTrainedConfig):
    r"""
    Qwen3VITAVisionConfig
    """
    model_type = "qwen3_vita_vision"
    base_config_key = "vision_config"

    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 32
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: int | None = 4096
    max_window_layers: int = 28
    layer_types: list[str] | None = None
    attention_dropout: float | int = 0.0
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

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
        merger_hidden_size=4608,
        # use_llm=False,
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
        self.merger_hidden_size = merger_hidden_size

        # self.use_llm = use_llm
    
    def __post_init__(self, **kwargs):
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        super().__post_init__(**kwargs)


class Qwen3VITATextConfig(Qwen3Config):
    r"""
    Qwen3VITATextConfig
    """
    model_type = "qwen3_vita_text"
    base_config_key = "text_config"
    pass


class Qwen3VITAOmniConfig(Qwen3VITATextConfig):

    model_type = "qwen3_vita_omni"
    base_config_key = "omni_config"

    # ---- Omni-specific fields ------------------------------------------------
    # Vision front-end
    num_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    # Audio Conv2d front-end (mirrors :class:`Qwen3VITACNNAudioEmbeddings`)
    num_mel_bins: int = 128
    downsample_hidden_size: int = 512
    n_window: int = 50
    n_window_infer: int = 800
    conv_chunksize: int = 500
    temporal_merge_size: int = 2
    # Projection to LM hidden size
    merger_hidden_size: int = 4608
    out_hidden_size: int = 4608


class Qwen3VITAConfig(PreTrainedConfig):
    r"""
    Qwen3VITAConfig
    """

    model_type = "qwen3_vita"
    sub_configs = {
        "audio_config": Qwen3VITAAudioConfig,
        "vision_config": Qwen3VITAVisionConfig,
        "text_config": Qwen3VITATextConfig,
        "omni_config": Qwen3VITAOmniConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        vision_config=None,
        omni_config=None,
        # image_token_id=133375,
        # video_token_id=133379,
        # audio_token_id=133383,
        # image_pad_token_id=133376,
        # audio_pad_token_id=133384,
        # vision_start_token_id=133377,
        # vision_end_token_id=133378,
        tie_word_embeddings=False,
        **kwargs,
    ):
        def _build_sub_config(key, value):
            if value is None:
                return None
            if isinstance(value, dict):
                return self.sub_configs[key](**value)
            return value

        self.audio_config = _build_sub_config("audio_config", audio_config)
        self.vision_config = _build_sub_config("vision_config", vision_config)
        self.omni_config = _build_sub_config("omni_config", omni_config)

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        # self.image_token_id = image_token_id
        # self.video_token_id = video_token_id
        # self.audio_token_id = audio_token_id
        # self.image_pad_token_id = image_pad_token_id
        # self.audio_pad_token_id = audio_pad_token_id
        # self.vision_start_token_id = vision_start_token_id
        # self.vision_end_token_id = vision_end_token_id

        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


def _get_feat_extract_output_lengths(input_lengths):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    """

    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


class Qwen3VITACNNAudioEmbeddings(nn.Module):
    def __init__(self, config: Qwen3VITAAudioConfig):
        super().__init__()

        self.config = config

        self.n_window = config.n_window
        self.n_window_infer = self.config.n_window_infer
        self.conv_chunksize = self.config.conv_chunksize

        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)

    def forward(
        self,
        input_features,
        feature_lens=None,
    ):
        input_features = input_features.to(self.conv2d1.weight.device)
        input_features = input_features.to(self.conv2d1.weight.dtype)

        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = torch.nn.functional.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = torch.nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)
        # Split to chunk to avoid OOM during convolution
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = torch.nn.functional.gelu(self.conv2d1(chunk))
            padded_embed = torch.nn.functional.gelu(self.conv2d2(padded_embed))
            padded_embed = torch.nn.functional.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        padded_embed = padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        hidden_states = padded_embed[padded_mask_after_cnn]

        return hidden_states, aftercnn_lens


class Qwen3VITACNNAudioEncoderLayer(nn.Module):

    def __init__(self, config: Qwen3VITAAudioConfig):
        super().__init__()

    def forward(
            self,
            hidden_states: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        return hidden_states


class Qwen3VITACNNAudioEncoder(nn.Module):

    def __init__(self, config: Qwen3VITAAudioConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            Qwen3VITACNNAudioEncoderLayer(config) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = True

    def forward(self, x):

        for idx, encoder_layer in enumerate(self.layers):
            x = encoder_layer(x)
        return x


class Qwen3VITACNNAudio(nn.Module):

    def __init__(self, config: Qwen3VITAAudioConfig):
        super().__init__()
        self.config = config

        self.embeddings = Qwen3VITACNNAudioEmbeddings(config)
        self.encoder = Qwen3VITACNNAudioEncoder(config)

    def forward(self, audios):

        audio_lengths = torch.as_tensor([len(x) for x in audios])
        # audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=0.0)
        audios = torch.cat(audios, dim=0).transpose(1, 0)

        features, feature_lengths = self.embeddings(audios, audio_lengths)

        features = self.encoder(features)

        features = features.split(feature_lengths.tolist(), dim=0)
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)

        return features, feature_lengths


class Qwen3VITAAudioSinusoidalPositionEncoder(torch.nn.Module):
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


class Qwen3VITAAudioPositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(Qwen3VITAAudioPositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class Qwen3VITAAudioMultiHeadedAttentionSANM(nn.Module):
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


class Qwen3VITAAudioLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = torch.nn.functional.layer_norm(
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


class Qwen3VITAAudioEncoderLayerSANM(nn.Module):
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
        super(Qwen3VITAAudioEncoderLayerSANM, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = Qwen3VITAAudioLayerNorm(in_size)
        self.norm2 = Qwen3VITAAudioLayerNorm(size)
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

        param_dtype = next(self.parameters()).dtype
        param_device = next(self.parameters()).device
        x = x.to(device=param_device, dtype=param_dtype)
        mask = mask.to(device=param_device, dtype=param_dtype)

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


class Qwen3VITAAudioEncoder(nn.Module):
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

        self.embed = Qwen3VITAAudioSinusoidalPositionEncoder()

        self.normalize_before = normalize_before

        positionwise_layer = Qwen3VITAAudioPositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
        )

        encoder_selfattn_layer = Qwen3VITAAudioMultiHeadedAttentionSANM
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
                Qwen3VITAAudioEncoderLayerSANM(
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
                Qwen3VITAAudioEncoderLayerSANM(
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
                Qwen3VITAAudioEncoderLayerSANM(
                    output_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(tp_blocks)
            ]
        )

        self.after_norm = Qwen3VITAAudioLayerNorm(output_size)

        self.tp_norm = Qwen3VITAAudioLayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ):
        """Embed positions in tensor."""
        masks = sequence_mask(ilens, dtype=torch.float32, device=ilens.device)[:, None, :]

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


class Qwen3VITASANMAudio(nn.Module):
    """
    """

    def __init__(
        self,
        config,
        **kwargs,
    ):

        super().__init__()

        encoder = Qwen3VITAAudioEncoder(
            input_size=config.input_size,
            output_size=config.hidden_size,
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
        audios,
        key: list = ["wav_file_tmp_name"],
        language = "auto",
        use_itn = False,
        output_timestamp = False,
        textnorm = None,
    ):

        speech = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=0.0)
        speech_lengths = torch.as_tensor([len(x) for x in audios])

        # fbank
        if len(speech.shape) < 3:
            speech = speech[None, :, :]
        if speech_lengths is None:
            speech_lengths = speech.shape[1]

        param_dtype = self.embed.weight.data.dtype
        param_device = self.embed.weight.data.device
        speech = speech.to(device=param_device, dtype=param_dtype)
        speech_lengths = speech_lengths.to(device=param_device, dtype=torch.int64)

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

        encoder_out = encoder_out[:, 4:, :]
        encoder_out_lens = encoder_out_lens - 4
        assert encoder_out.shape[0] == len(audios)

        return encoder_out, encoder_out_lens


def pad_and_reshape(A, M):
    B, S, D = A.shape

    # 1. Calculate required padding for the S dimension
    pad_size = (M - (S % M)) % M

    # 2. Pad (0, 0) for D, and (0, pad_size) for S
    # torch.nn.functional.pad expects padding for dimensions in reverse order:
    # (last_dim_front, last_dim_back, second_to_last_front, second_to_last_back, ...)
    if pad_size > 0:
        A = torch.nn.functional.pad(A, (0, 0, 0, pad_size))

    # Updated sequence length
    new_S = S + pad_size

    # 3. Reshape to (B, S // M, D * M)
    return A.view(B, new_S // M, M, D).flatten(2)


class Qwen3VITAAudioPatchMerger(nn.Module):
    def __init__(self, config: Qwen3VITAAudioConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size * (config.temporal_merge_size**1)
        self.norm = nn.RMSNorm(self.hidden_size)
        self.linear_fc1 = nn.Linear(self.hidden_size, config.merger_hidden_size, bias=False)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(config.merger_hidden_size, config.out_hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.temporal_merge_size > 1:
            x = pad_and_reshape(x, self.config.temporal_merge_size)
        x = self.norm(x.reshape(x.shape[0], -1, x.shape[-1]))
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class Qwen3VITAVisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3VITAVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.norm = nn.RMSNorm(self.hidden_size)
        self.linear_fc1 = nn.Linear(self.hidden_size, config.merger_hidden_size, bias=False)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(config.merger_hidden_size, config.out_hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.reshape(-1, self.hidden_size))
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class Qwen3VITATextRMSNorm(Qwen3RMSNorm):
    pass


class Qwen3VITATextRotaryEmbedding(Qwen3RotaryEmbedding):
    pass


class Qwen3VITATextMLP(Qwen3MLP):
    pass


class Qwen3VITATextAttention(Qwen3Attention):
    pass

    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     position_embeddings: tuple[torch.Tensor, torch.Tensor],
    #     attention_mask: torch.Tensor | None,
    #     past_key_values: Cache | None = None,
    #     **kwargs: Unpack[FlashAttentionKwargs],
    # ) -> tuple[torch.Tensor, torch.Tensor | None]:
    #     input_shape = hidden_states.shape[:-1]
    #     hidden_shape = (*input_shape, -1, self.head_dim)

    #     query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    #     key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    #     value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    #     cos, sin = position_embeddings
    #     if self.config.use_llm:
    #         # [B, H, S, D] -> [S, H, D]
    #         query_states = query_states.squeeze(0).transpose(0, 1)
    #         key_states = key_states.squeeze(0).transpose(0, 1)
    #         query_states, key_states = vision_apply_rotary_pos_emb(query_states, key_states, cos, sin)
    #         # [S, H, D] -> [B, H, S, D]
    #         query_states = query_states.transpose(0, 1).unsqueeze(0)
    #         key_states = key_states.transpose(0, 1).unsqueeze(0)
    #     else:
    #         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    #     if past_key_values is not None:
    #         key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    #     attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
    #         self.config._attn_implementation, eager_attention_forward
    #     )

    #     attn_output, attn_weights = attention_interface(
    #         self,
    #         query_states,
    #         key_states,
    #         value_states,
    #         attention_mask,
    #         dropout=0.0 if not self.training else self.attention_dropout,
    #         scaling=self.scaling,
    #         sliding_window=self.sliding_window,  # diff with Llama
    #         **kwargs,
    #     )

    #     attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    #     attn_output = self.o_proj(attn_output)
    #     return attn_output, attn_weights


class Qwen3VITATextDecoderLayer(Qwen3DecoderLayer):
    pass


class Qwen3VITAAudioPreTrainedModel(PreTrainedModel):
    _supports_flash_attn = True
    _supports_sdpa = True
    pass


class Qwen3VITAVisionPreTrainedModel(PreTrainedModel):
    _supports_flash_attn = True
    _supports_sdpa = True
    pass


class Qwen3VITAPreTrainedModel(PreTrainedModel):
    _supports_flash_attn = True
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


class Qwen3VITAAudioModel(Qwen3VITAAudioPreTrainedModel):
    config: Qwen3VITAAudioConfig

    def __init__(
        self,
        config: Qwen3VITAAudioConfig,
        *inputs,
        **kwargs,
    ):
        super().__init__(config, *inputs, **kwargs)

        if config.num_blocks == 0 and config.tp_blocks == 0:
            self.model = Qwen3VITACNNAudio(config)
        else:
            self.model = Qwen3VITASANMAudio(config)
        self.merger = Qwen3VITAAudioPatchMerger(config)
    
    def forward(
        self,
        audios,
    ):
        encoder_out, encoder_out_lens = self.model(audios)
        encoder_out = self.merger(encoder_out)

        encoder_out_lens = -(-encoder_out_lens // self.config.temporal_merge_size)

        return encoder_out, encoder_out_lens


class Qwen3VITAVisionRotaryEmbedding(nn.Module):
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


class Qwen3VITAVisionEmbeddings(nn.Module):
    def __init__(self, config: Qwen3VITAVisionConfig):
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


class Qwen3VITAVisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3VITAVisionConfig):
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
        # output_attentions: Optional[bool] = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        seq_length, embed_dim = hidden_states.shape
        # _, seq_length, embed_dim = hidden_states.shape

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

        # if not output_attentions:
        #     attn_weights = None

        return attn_output, attn_weights


class Qwen3VITAVisionFlashAttention2(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3VITAVisionConfig):
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
        # _, seq_length, embed_dim = hidden_states.shape

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


class Qwen3VITAVisionMLP(nn.Module):
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


class Qwen3VITAVisionEncoderLayer(nn.Module):
    def __init__(self, config: Qwen3VITAVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # self.self_attn = Qwen3VITAVisionAttention(config)
        self.self_attn = Qwen3VITAVisionFlashAttention2(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Qwen3VITAVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        # output_attentions: Optional[bool] = False,
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
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            # output_attentions=output_attentions,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
        
        # outputs = (hidden_states,)

        # if output_attentions:
        #     outputs += (attn_weights,)

        # return outputs


class Qwen3VITAVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Qwen3VITAVisionEncoderLayer`].

    Args:
        config: Qwen3VITAVisionConfig
    """

    def __init__(self, config: Qwen3VITAVisionConfig):
        super().__init__()
        self.config = config
        # if self.config.use_llm:
        #     self.layers = nn.ModuleList([Qwen3VITATextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        # else:
        self.layers = nn.ModuleList([Qwen3VITAVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        self.spatial_merge_size = 2
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_size = config.patch_size
        # self.window_size = self.patch_size * 2 * 8

        # self.rotary_emb = Qwen3VITATextRotaryEmbedding(config=config)
        self.rotary_pos_emb = Qwen3VITAVisionRotaryEmbedding(config.head_dim // 2)

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

    def forward(
        self,
        inputs_embeds,
        grid_thw,
        attention_mask: Optional[torch.Tensor] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
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
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )

        # encoder_states = () if output_hidden_states else None
        # all_attentions = () if output_attentions else None

        # if self.config.use_llm and False:
        #     # position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        #     # position_ids = position_ids.unsqueeze(0)
        #     # position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
            
        #     rotary_pos_emb = self.rot_pos_emb(grid_thw)
        #     emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        #     position_embeddings = (emb.cos().unsqueeze(0), emb.sin().unsqueeze(0))
        # else:
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        # print(f"{position_embeddings[0].shape=} {position_embeddings[1].shape=} {inputs_embeds.shape=}")

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            encoder_layer.self_attn.is_causal = False
            # if output_hidden_states:
            #     encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    # output_attentions,
                    cu_seqlens,
                    position_embeddings,
                )
            else:
                hidden_states = encoder_layer(
                    hidden_states,
                    attention_mask,
                    # output_attentions=output_attentions,
                    cu_seqlens=cu_seqlens, 
                    position_embeddings=position_embeddings
                )

            # hidden_states = layer_outputs[0]

            # if output_attentions:
            #     all_attentions = all_attentions + (layer_outputs[1],)

        # if output_hidden_states:
        #     encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            # hidden_states=encoder_states,
            # attentions=all_attentions,
        )


class Qwen3VITAVisionModel(Qwen3VITAVisionPreTrainedModel):
    _input_embed_layer = "patch_embedding"
    config: Qwen3VITAVisionConfig

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Qwen3VITAVisionEmbeddings(config)
        self.encoder = Qwen3VITAVisionEncoder(config)

        self.merger = Qwen3VITAVisionPatchMerger(config)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thw: torch.LongTensor,
        attention_mask: torch.Tensor,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        r"""
        Returns:

        """
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )

        hidden_states = self.embeddings(pixel_values, grid_thw)
        # if self.config.use_llm:
        #     hidden_states = hidden_states.unsqueeze(0)

        if attention_mask is not None and not self._use_flash_attention_2:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        else:
            encoder_attention_mask = attention_mask

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            grid_thw=grid_thw,
            attention_mask=encoder_attention_mask,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        # last_hidden_state = self.post_layernorm(last_hidden_state)

        # pooler_output = self.head(last_hidden_state, attention_mask) if self.use_head else None
        # pooler_output = None

        # if self.config.use_llm:
        #     last_hidden_state = last_hidden_state.squeeze(0)
        
        assert last_hidden_state.shape[0] == len(pixel_values)
        last_hidden_state = self.merger(last_hidden_state)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            # pooler_output=pooler_output,
            # hidden_states=encoder_outputs.hidden_states,
            # attentions=encoder_outputs.attentions,
        )


class Qwen3VITATextModel(Qwen3VITAPreTrainedModel, Qwen3Model):
    config: Qwen3VITATextConfig


# ---------------------------------------------------------------------------
# Omni encoder (shared Qwen3 transformer for vision + audio).
#
# Mirrors ``vita_megatron/core/models/omni`` (``omni_model.py``,
# ``qwen3_model.py``): a single Qwen3-style transformer body consumes packed
# features from both modalities with 2D (vision) / 1D (audio) rotary positions,
# followed by a modality-specific merger+MLP projector into the LM hidden dim.
# ---------------------------------------------------------------------------


class Qwen3VITAOmniPreTrainedModel(PreTrainedModel):
    _supports_flash_attn = True
    _supports_sdpa = True
    config: Qwen3VITAOmniConfig


class Qwen3VITAOmniRMSNorm(Qwen3RMSNorm):
    pass


class Qwen3VITAOmniRotaryEmbedding(nn.Module):
    """1D rotary building block shared by vision (2D) and audio (1D) RoPE.

    Mirrors ``_RotaryEmbedding`` in ``vita_megatron/core/models/omni/qwen3_model.py``.
    """

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, seqlen) -> torch.Tensor:
        # ``seqlen`` may be a python int or a 0-D / 1-D tensor.
        if isinstance(seqlen, torch.Tensor):
            device = seqlen.device
            length = int(seqlen.max().item()) if seqlen.dim() > 0 else int(seqlen.item())
        else:
            device = torch.device("cpu")
            length = int(seqlen)
        inv_freq = 1.0 / (
            self.theta
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
        )
        seq = torch.arange(length, device=device, dtype=inv_freq.dtype)
        return torch.outer(seq, inv_freq)


class Qwen3VITAOmniVisionEmbeddings(Qwen3VITAVisionEmbeddings):
    """Linear patch embedding for already-patchified vision pixel values.

    Identical to :class:`Qwen3VITAVisionEmbeddings`; only the config type
    differs (omni reuses ``hidden_size``/``patch_size``/``num_channels`` from
    :class:`Qwen3VITAOmniConfig`). Mirrors ``LinearVisionEncoder`` in
    ``vita_megatron/core/models/vision/linear_model.py``.
    """

    def __init__(self, config: Qwen3VITAOmniConfig):
        super().__init__(config)


class Qwen3VITAOmniAudioEmbeddings(Qwen3VITACNNAudioEmbeddings):
    """Conv2d audio front-end producing 8x temporally down-sampled features.

    Inherits the three Conv2d stack from :class:`Qwen3VITACNNAudioEmbeddings`
    and adds a final linear projection so the CNN output is aligned with the
    shared omni transformer ``hidden_size`` (mirrors
    ``Conv2dAudioEncoderWithProj`` in
    ``vita_megatron/core/models/audio/cnn_model.py``).
    """

    def __init__(self, config: Qwen3VITAOmniConfig):
        super().__init__(config)
        # After three stride-2, padding=1 convolutions the feature axis is
        # reduced as ``ceil(n / 2)`` at each step.
        mel_after_cnn = (config.num_mel_bins + 1) // 2
        mel_after_cnn = (mel_after_cnn + 1) // 2
        mel_after_cnn = (mel_after_cnn + 1) // 2
        self.proj_in_features = config.downsample_hidden_size * mel_after_cnn
        self.linear_proj = nn.Linear(self.proj_in_features, config.hidden_size, bias=False)

    def forward(self, input_features, feature_lens=None):
        hidden_states, aftercnn_lens = super().forward(input_features, feature_lens)
        hidden_states = self.linear_proj(hidden_states)
        return hidden_states, aftercnn_lens


class Qwen3VITAOmniMLP(Qwen3MLP):
    """SwiGLU MLP (Qwen3 style). Inherits :class:`Qwen3MLP` directly."""

    pass


class Qwen3VITAOmniFlashAttention2(Qwen3Attention):
    """Packed ``thd``-layout Qwen3-style attention (qk-norm, GQA) that takes
    ``cu_seqlens`` and per-token ``(cos, sin)`` rotary positions.

    Inherits :class:`Qwen3Attention` for the projection layers / qk-norm /
    GQA wiring; only the forward path differs (varlen flash attention with
    packed sequences instead of the standard causal LM attention).
    """

    def __init__(self, config: Qwen3VITAOmniConfig, layer_idx: int = 0):
        super().__init__(config, layer_idx=layer_idx)
        # Encoder use-case: bidirectional attention.
        self.is_causal = False
        # Mirror naming used by :class:`Qwen3VITAVisionFlashAttention2`.
        self.dropout = config.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # ``hidden_states``: [seq_len, hidden] (packed ``thd``).
        seq_length = hidden_states.shape[0]
        num_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads

        queries = self.q_proj(hidden_states).view(seq_length, num_heads, self.head_dim)
        keys = self.k_proj(hidden_states).view(seq_length, num_key_value_heads, self.head_dim)
        values = self.v_proj(hidden_states).view(seq_length, num_key_value_heads, self.head_dim)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        cos, sin = position_embeddings
        queries, keys = vision_apply_rotary_pos_emb_flashatt(
            queries.unsqueeze(0), keys.unsqueeze(0), cos, sin
        )
        queries = queries.squeeze(0)
        keys = keys.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        if is_aiter_available:
            attn_output = flash_attn_varlen_func(
                queries, keys, values,
                cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                return_lse=True,
            )[0].reshape(seq_length, -1)
        else:
            attn_output = flash_attn_varlen_func(
                queries, keys, values,
                cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
            ).reshape(seq_length, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3VITAOmniEncoderLayer(nn.Module):
    """Qwen3-style transformer block (pre-norm, SwiGLU, qk-norm)."""

    def __init__(self, config: Qwen3VITAOmniConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_layernorm = Qwen3VITAOmniRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3VITAOmniFlashAttention2(config)
        self.post_attention_layernorm = Qwen3VITAOmniRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3VITAOmniMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3VITAOmniEncoder(Qwen3VITAOmniPreTrainedModel):
    """Shared Qwen3 transformer body used by both vision and audio inputs.

    Equivalent of ``Qwen3Model`` in
    ``vita_megatron/core/models/omni/qwen3_model.py``. Consumes already-projected
    features of shape ``[seq_len, hidden]`` together with packed-sequence
    metadata (``cu_seqlens``, per-token ``(cos, sin)`` rotary positions).
    """

    config: Qwen3VITAOmniConfig
    _no_split_modules = ["Qwen3VITAOmniEncoderLayer"]

    def __init__(self, config: Qwen3VITAOmniConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.spatial_merge_size = int(config.spatial_merge_size)
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        # Rotary building block. ``head_dim // 2`` matches the megatron reference
        # (``kv_channels // 2``) — vision concatenates ``(h, w)`` pos so the
        # final freqs dim equals ``head_dim``, audio duplicates to match.
        self.rotary_pos_emb = Qwen3VITAOmniRotaryEmbedding(config.head_dim // 2)

        self.layers = nn.ModuleList(
            [Qwen3VITAOmniEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

        self.post_init()

    # ------------------------------------------------------------------ RoPE
    def vision_rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """2D rotary positions for Qwen-VL ``grid_thw``."""
        pos_ids = []
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def audio_rot_pos_emb(self, lens: torch.Tensor) -> torch.Tensor:
        """1D rotary positions for packed audio sequences."""
        max_len = int(lens.max().item())
        rotary_pos_emb_full = self.rotary_pos_emb(max_len)  # [max_len, dim/2]
        out = []
        for length in lens.tolist():
            out.append(rotary_pos_emb_full[: int(length)])
        rotary_pos_emb = torch.cat(out, dim=0)
        # Duplicate along feature axis so we can reuse the same (cos, sin)
        # application as the 2D vision path.
        rotary_pos_emb = torch.cat([rotary_pos_emb, rotary_pos_emb], dim=-1)
        return rotary_pos_emb

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
    ) -> BaseModelOutput:
        """Run the transformer body on packed features.

        Args:
            hidden_states: ``[seq_len, hidden]`` already-projected features.
            cu_seqlens: cumulative sequence lengths (``thd`` varlen format).
            rotary_pos_emb: ``[seq_len, head_dim]`` rotary positions (already
                concatenated for both ``h`` and ``w`` / duplicated for audio).
        """
        hidden_states = hidden_states.contiguous()

        # Per-token (cos, sin) used by ``vision_apply_rotary_pos_emb_flashatt``.
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    cu_seqlens,
                    position_embeddings,
                )
            else:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                )

        return BaseModelOutput(last_hidden_state=hidden_states)


class Qwen3VITAOmniVisionPatchMerger(Qwen3VITAVisionPatchMerger):
    """2x2 spatial merge (4x token reduction) + MLP projection to
    ``out_hidden_size``. Inherits :class:`Qwen3VITAVisionPatchMerger`; only the
    config type differs (omni reuses ``hidden_size`` / ``spatial_merge_size``
    / ``merger_hidden_size`` / ``out_hidden_size`` from
    :class:`Qwen3VITAOmniConfig`).
    """

    def __init__(self, config: Qwen3VITAOmniConfig) -> None:
        super().__init__(config)


class Qwen3VITAOmniAudioPatchMerger(Qwen3VITAAudioPatchMerger):
    """2x temporal merge + MLP projection to ``out_hidden_size``. Inherits
    :class:`Qwen3VITAAudioPatchMerger`; only the config type differs.
    """

    def __init__(self, config: Qwen3VITAOmniConfig) -> None:
        super().__init__(config)


class Qwen3VITAOmniModel(Qwen3VITAOmniPreTrainedModel):
    """Top-level omni encoder: shared Qwen3 transformer + modality-specific
    front-ends and mergers. Mirrors :class:`MegatronOmniModel`.

    Forward dispatch by modality:

    * ``modality == "vision"`` or only vision inputs given → returns
      ``image_embeddings`` of shape ``[N, out_hidden_size]``.
    * ``modality == "audio"`` or only audio inputs given → returns
      ``(audio_embeddings, audio_lengths)``.
    * Both given → returns ``{"vision": ..., "audio": (..., ...)}``.
    """

    config: Qwen3VITAOmniConfig
    _input_embed_layer = "patch_embedding"

    def __init__(self, config: Qwen3VITAOmniConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config

        # Modality front-ends.
        self.vision_embeddings = Qwen3VITAOmniVisionEmbeddings(config)
        self.audio_embeddings = Qwen3VITAOmniAudioEmbeddings(config)

        # Shared Qwen3 transformer body.
        self.encoder = Qwen3VITAOmniEncoder(config)

        # Modality-specific post-encoder mergers + projectors.
        self.vision_merger = Qwen3VITAOmniVisionPatchMerger(config)
        self.audio_merger = Qwen3VITAOmniAudioPatchMerger(config)

        self.post_init()

    # ------------------------------------------------------------------ vision
    def forward_vision(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Vision path: patch linear → shared encoder (2D RoPE, packed) → 2x2
        spatial merge → projector. Returns ``[N, out_hidden_size]``."""
        hidden_states = self.vision_embeddings(pixel_values, image_grid_thw)

        rotary_pos_emb = self.encoder.vision_rot_pos_emb(image_grid_thw).to(hidden_states.device)
        cu_seqlens = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2],
            image_grid_thw[:, 0],
        ).cumsum(
            dim=0,
            dtype=image_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        hidden_states = self.encoder(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        ).last_hidden_state

        hidden_states = self.vision_merger(hidden_states)
        return hidden_states

    # ------------------------------------------------------------------- audio
    def forward_audio(
        self,
        audios,
    ):
        """Audio path: Conv2d front-end (8x down-sample) → shared encoder
        (1D RoPE, packed) → 2x temporal merge → projector.

        Args:
            audios: a list of ``[T_i, num_mel_bins]`` tensors (the same input
                convention used by :class:`Qwen3VITACNNAudio` elsewhere in this
                file).

        Returns:
            A tuple ``(features, feature_lens)`` where ``features`` has shape
            ``[B, S_out, out_hidden_size]`` and ``feature_lens`` holds the
            valid length per sample after temporal merge.
        """
        audio_lengths = torch.as_tensor([len(x) for x in audios])
        stacked = torch.cat(audios, dim=0).transpose(1, 0)  # [num_mel, total_T]
        hidden_states, feature_lens = self.audio_embeddings(stacked, audio_lengths)

        # Packed 1D rotary positions + cu_seqlens.
        feature_lens = feature_lens.to(hidden_states.device)
        cu_seqlens = torch.nn.functional.pad(
            feature_lens.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0
        )
        rotary_pos_emb = self.encoder.audio_rot_pos_emb(feature_lens).to(hidden_states.device)

        hidden_states = self.encoder(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        ).last_hidden_state

        # Re-batch to ``[B, S, H]`` and apply temporal merge + projector.
        features = hidden_states.split(feature_lens.tolist(), dim=0)
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
        features = self.audio_merger(features)

        merged_lens = -(-feature_lens // int(self.config.temporal_merge_size))
        return features, merged_lens

    # ------------------------------------------------------------------ video
    def forward_video(
        self,
        video_images: torch.Tensor,
        video_image_grid_thw: torch.Tensor,
        video_audios: Optional[list] = None,
        video_split: Optional[torch.Tensor] = None,
    ):
        """Joint video path: vision frames and audio chunks of the same video
        share an attention window inside ``self.encoder`` while remaining
        attention-isolated from other videos in the batch.

        Mirrors :meth:`MegatronOmniModel.forward_video` in
        ``vita_megatron/core/models/omni/omni_model.py``.

        Args:
            video_images: ``[N_patch_rows, 3 * patch_dim ** 2]`` -- all video
                frames concatenated along dim 0.
            video_image_grid_thw: ``[N_grid_rows, 3]`` -- per-frame
                ``(T, H, W)`` rows. Multiple rows may belong to the same video.
            video_audios: list of ``[T_i, num_mel_bins]`` per audio-chunk
                tensors (same convention as :meth:`forward_audio`); multiple
                chunks may belong to the same video. ``None`` or empty list
                if no audio.
            video_split: ``[N_video, 2]`` -- per-video ``(num_images,
                num_audios)`` deltas. The number of vision patch rows belonging
                to a video is fully determined by the corresponding rows of
                ``video_image_grid_thw``.

        Returns:
            tuple ``(video_image_embeddings, video_audio_embeddings,
            video_audio_lens_after_merge)``:

            * ``video_image_embeddings``: ``[N_v_after_merge, out_hidden_size]``
            * ``video_audio_embeddings``: ``[N_audios, max_S_after_merge,
              out_hidden_size]``
            * ``video_audio_lens_after_merge``: ``[N_audios]``
        """
        if video_split is None or video_split.numel() == 0:
            raise ValueError(
                "Qwen3VITAOmniModel.forward_video requires a non-empty "
                "`video_split` tensor of shape [N_video, 2]."
            )

        device = video_images.device

        # 1. Per-modality frontends.
        vision_features = self.vision_embeddings(video_images, video_image_grid_thw)

        has_audio = video_audios is not None and len(video_audios) > 0
        if has_audio:
            video_audio_lens = torch.as_tensor(
                [x.shape[0] for x in video_audios], dtype=torch.long, device=device
            )
            packed_audio = torch.cat(list(video_audios), dim=0).transpose(1, 0)
            audio_features, audio_token_lens = self.audio_embeddings(
                packed_audio, video_audio_lens
            )
            audio_token_lens = audio_token_lens.to(device)
        else:
            video_audio_lens = torch.zeros((0,), dtype=torch.long, device=device)
            audio_features = vision_features.new_zeros((0, vision_features.size(-1)))
            audio_token_lens = torch.zeros((0,), dtype=torch.long, device=device)

        # 2. Per-video segmentation: split flat per-modality buffers, build
        # rotary embeddings, interleave image / audio chunks in temporal order.
        num_videos = int(video_split.shape[0])

        image_cursor = 0
        audio_chunk_cursor = 0
        vision_patch_cursor = 0
        audio_token_cursor = 0

        segment_features = []
        segment_rotary = []
        segment_lengths = []
        segment_modality_masks = []  # True = vision, False = audio
        per_video_audio_chunk_lens = []

        for video_index in range(num_videos):
            num_images = int(video_split[video_index, 0].item())
            num_audio_chunks = int(video_split[video_index, 1].item())

            # ---- vision slice ----
            video_grid_thw = video_image_grid_thw[
                image_cursor : image_cursor + num_images
            ]
            if num_images > 0:
                num_patch_rows = int(
                    (video_grid_thw[:, 0] * video_grid_thw[:, 1] * video_grid_thw[:, 2])
                    .sum()
                    .item()
                )
            else:
                num_patch_rows = 0
            video_vision_features = vision_features[
                vision_patch_cursor : vision_patch_cursor + num_patch_rows
            ]
            image_cursor += num_images
            vision_patch_cursor += num_patch_rows

            # ---- audio slice ----
            video_audio_chunk_lens = audio_token_lens[
                audio_chunk_cursor : audio_chunk_cursor + num_audio_chunks
            ]
            video_audio_total_tokens = (
                int(video_audio_chunk_lens.sum().item()) if num_audio_chunks > 0 else 0
            )
            video_audio_features = audio_features[
                audio_token_cursor : audio_token_cursor + video_audio_total_tokens
            ]
            audio_chunk_cursor += num_audio_chunks
            audio_token_cursor += video_audio_total_tokens

            # ---- per-frame vision rotary ----
            if num_images > 0:
                video_vision_rotary = self.encoder.vision_rot_pos_emb(video_grid_thw).to(device)
                tokens_per_frame = (
                    video_grid_thw[:, 0] * video_grid_thw[:, 1] * video_grid_thw[:, 2]
                ).tolist()
                vision_feature_chunks = list(
                    video_vision_features.split(tokens_per_frame, dim=0)
                )
                vision_rotary_chunks = list(
                    video_vision_rotary.split(tokens_per_frame, dim=0)
                )
            else:
                vision_feature_chunks, vision_rotary_chunks = [], []

            # ---- per-chunk audio rotary ----
            if num_audio_chunks > 0:
                video_audio_rotary = self.encoder.audio_rot_pos_emb(
                    video_audio_chunk_lens
                ).to(device)
                tokens_per_audio_chunk = video_audio_chunk_lens.tolist()
                audio_feature_chunks = list(
                    video_audio_features.split(tokens_per_audio_chunk, dim=0)
                )
                audio_rotary_chunks = list(
                    video_audio_rotary.split(tokens_per_audio_chunk, dim=0)
                )
            else:
                audio_feature_chunks, audio_rotary_chunks = [], []

            # ---- interleave I and A in temporal order ----
            # If both modalities are present, distribute audio chunks across
            # image chunks: each image is followed by ``audios_per_image`` audio
            # chunks, the first ``extra_audio_count`` images get one extra
            # trailing audio chunk. If only one modality is present, keep its
            # natural order.
            interleaved_features = []
            interleaved_rotary = []
            interleaved_modality_mask = []

            num_image_chunks = len(vision_feature_chunks)
            num_audio_chunks_actual = len(audio_feature_chunks)

            if num_image_chunks == 0 and num_audio_chunks_actual == 0:
                continue
            elif num_image_chunks == 0:
                for audio_chunk, audio_rot in zip(audio_feature_chunks, audio_rotary_chunks):
                    interleaved_features.append(audio_chunk)
                    interleaved_rotary.append(audio_rot)
                    interleaved_modality_mask.append(
                        torch.zeros(audio_chunk.size(0), dtype=torch.bool, device=device)
                    )
            elif num_audio_chunks_actual == 0:
                for image_chunk, image_rot in zip(vision_feature_chunks, vision_rotary_chunks):
                    interleaved_features.append(image_chunk)
                    interleaved_rotary.append(image_rot)
                    interleaved_modality_mask.append(
                        torch.ones(image_chunk.size(0), dtype=torch.bool, device=device)
                    )
            else:
                audios_per_image, extra_audio_count = divmod(
                    num_audio_chunks_actual, num_image_chunks
                )
                audio_index = 0
                for image_index in range(num_image_chunks):
                    image_chunk = vision_feature_chunks[image_index]
                    image_rot = vision_rotary_chunks[image_index]
                    interleaved_features.append(image_chunk)
                    interleaved_rotary.append(image_rot)
                    interleaved_modality_mask.append(
                        torch.ones(image_chunk.size(0), dtype=torch.bool, device=device)
                    )
                    audios_to_take = audios_per_image + (
                        1 if image_index < extra_audio_count else 0
                    )
                    for _ in range(audios_to_take):
                        audio_chunk = audio_feature_chunks[audio_index]
                        audio_rot = audio_rotary_chunks[audio_index]
                        interleaved_features.append(audio_chunk)
                        interleaved_rotary.append(audio_rot)
                        interleaved_modality_mask.append(
                            torch.zeros(audio_chunk.size(0), dtype=torch.bool, device=device)
                        )
                        audio_index += 1
                assert audio_index == num_audio_chunks_actual, (
                    f"video {video_index}: distributed {audio_index} audio chunks, "
                    f"expected {num_audio_chunks_actual}"
                )

            video_segment_features = torch.cat(interleaved_features, dim=0)
            video_segment_rotary = torch.cat(interleaved_rotary, dim=0)
            video_segment_mask = torch.cat(interleaved_modality_mask, dim=0)

            segment_features.append(video_segment_features)
            segment_rotary.append(video_segment_rotary)
            segment_modality_masks.append(video_segment_mask)
            segment_lengths.append(video_segment_features.size(0))
            per_video_audio_chunk_lens.append(video_audio_chunk_lens)

        # 3. Concatenate per-video segments into one packed sequence.
        if len(segment_features) == 0:
            empty_vision = vision_features.new_zeros(
                (0, self.config.out_hidden_size)
            )
            empty_audio = vision_features.new_zeros((0, 0, self.config.out_hidden_size))
            empty_lens = torch.zeros((0,), dtype=torch.long, device=device)
            return empty_vision, empty_audio, empty_lens

        packed_features = torch.cat(segment_features, dim=0)
        packed_rotary = torch.cat(segment_rotary, dim=0)
        packed_modality_mask = torch.cat(segment_modality_masks, dim=0)

        cu_seqlens = torch.nn.functional.pad(
            torch.tensor(segment_lengths, dtype=torch.int32, device=device).cumsum(
                dim=0, dtype=torch.int32
            ),
            (1, 0),
            value=0,
        )

        # 4. Run the shared transformer once over the packed sequence.
        encoder_output = self.encoder(
            hidden_states=packed_features,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=packed_rotary,
        ).last_hidden_state

        # 5. Split encoder output back via ``packed_modality_mask``.
        vision_output = encoder_output[packed_modality_mask]
        audio_output_packed = encoder_output[~packed_modality_mask]

        # ---- Vision post-processing ----
        if vision_output.size(0) > 0:
            vision_output = self.vision_merger(vision_output)
        else:
            vision_output = encoder_output.new_zeros((0, self.config.out_hidden_size))

        # ---- Audio post-processing ----
        if audio_output_packed.size(0) > 0 and len(per_video_audio_chunk_lens) > 0:
            all_audio_chunk_lens = torch.cat(per_video_audio_chunk_lens, dim=0)
            audio_chunks_split = audio_output_packed.split(
                all_audio_chunk_lens.tolist(), dim=0
            )
            audio_output = torch.nn.utils.rnn.pad_sequence(
                audio_chunks_split, batch_first=True, padding_value=0.0
            )
            audio_output = self.audio_merger(audio_output)
            audio_lens_after_merge = -(
                -all_audio_chunk_lens // int(self.config.temporal_merge_size)
            )
        else:
            audio_output = encoder_output.new_zeros(
                (0, 0, self.config.out_hidden_size)
            )
            audio_lens_after_merge = torch.zeros((0,), dtype=torch.long, device=device)

        return vision_output, audio_output, audio_lens_after_merge

    # -------------------------------------------------------------- top-level
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        audios: Optional[Any] = None,
        modality: Optional[str] = None,
        # video_omni_fusion joint path
        video_images: Optional[torch.Tensor] = None,
        video_image_grid_thw: Optional[torch.Tensor] = None,
        video_audios: Optional[list] = None,
        video_split: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Dispatch by modality (mirrors ``MegatronOmniModel.forward``).

        * ``modality == "vision"`` or only vision inputs → ``image_embeddings``
          of shape ``[N, out_hidden_size]``.
        * ``modality == "audio"`` or only audio inputs → ``(audio_embeddings,
          audio_lengths)``.
        * ``modality == "video"`` or ``video_split`` is given → joint encode,
          returns ``(video_image_embeddings, video_audio_embeddings,
          video_audio_lens_after_merge)``.
        * Both vision and audio (legacy independent dual call) → ``{"vision":
          ..., "audio": (..., ...)}``.
        """
        has_vision = pixel_values is not None and image_grid_thw is not None
        has_audio = audios is not None
        has_video = video_split is not None and video_split.numel() > 0

        if modality == "video" or (has_video and modality is None):
            return self.forward_video(
                video_images=video_images,
                video_image_grid_thw=video_image_grid_thw,
                video_audios=video_audios,
                video_split=video_split,
            )

        if modality == "vision" or (has_vision and not has_audio):
            return self.forward_vision(pixel_values, image_grid_thw)

        if modality == "audio" or (has_audio and not has_vision):
            return self.forward_audio(audios)

        if has_vision and has_audio:
            v = self.forward_vision(pixel_values, image_grid_thw)
            a, a_len = self.forward_audio(audios)
            return {"vision": v, "audio": (a, a_len)}

        raise ValueError(
            "Qwen3VITAOmniModel.forward requires at least one of (pixel_values, image_grid_thw), "
            "(audios,), or (video_images, video_image_grid_thw, video_split)."
        )


class Qwen3VITAModel(Qwen3VITAPreTrainedModel):
    def __init__(self, config: Qwen3VITAConfig):
        super().__init__(config)

        self.omni_model = None
        self.vision_model = None
        self.audio_model = None
        if config.omni_config is not None:
            self.omni_model = Qwen3VITAOmniModel._from_config(config.omni_config)
        if config.vision_config is not None:
            self.vision_model = Qwen3VITAVisionModel._from_config(config.vision_config)
        if config.audio_config is not None:
            self.audio_model = Qwen3VITAAudioModel._from_config(config.audio_config)

        self.language_model = Qwen3VITATextModel._from_config(config.text_config)

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    # ------------------------------------------------------------------
    # Modality forward dispatch — when ``self.omni_model`` is present, the
    # vision and audio inputs go through the shared Qwen3 omni encoder
    # (mirrors ``GPTMMModel._preprocess`` in
    # ``vita_megatron/core/models/multimodal/gpt_mm_model.py``).
    # ------------------------------------------------------------------
    def _encode_vision(self, pixel_values, image_grid_thw):
        """Return image embeddings of shape ``[N, hidden]``."""
        if self.omni_model is not None:
            return self.omni_model(
                modality="vision",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        return self.vision_model(
            pixel_values=pixel_values,
            attention_mask=None,
            grid_thw=image_grid_thw,
        ).last_hidden_state

    def _encode_audio(self, audios):
        """Return ``(audio_embeddings, audio_lengths)``."""
        if self.omni_model is not None:
            return self.omni_model(modality="audio", audios=audios)
        return self.audio_model(audios)

    def _encode_video(
        self,
        video_images,
        video_image_grid_thw,
        video_audios,
        video_split,
    ):
        """Joint video encode: same-video vision/audio share an attention
        window inside the omni encoder. Mirrors
        ``GPTMMModel._preprocess`` joint-video branch in
        ``vita_megatron/core/models/multimodal/gpt_mm_model.py``.

        Returns ``(video_image_embeds, video_audio_embeds, video_audio_lens)``.
        """
        if self.omni_model is None:
            raise ValueError(
                "Qwen3VITAModel: video joint encoding requires `omni_model`, "
                "but it is not configured. Either disable `video_omni_fusion` "
                "in the processor or load a checkpoint with `omni_config`."
            )
        return self.omni_model(
            modality="video",
            video_images=video_images,
            video_image_grid_thw=video_image_grid_thw,
            video_audios=video_audios,
            video_split=video_split,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        images: torch.FloatTensor | None = None,
        image_indices: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        audios: torch.FloatTensor | None = None,
        audio_indices: torch.LongTensor | None = None,
        # video_omni_fusion joint path
        video_images: torch.FloatTensor | None = None,
        video_image_grid_thw: torch.LongTensor | None = None,
        video_image_indices: torch.LongTensor | None = None,
        video_audios: list | None = None,
        video_audio_indices: list | None = None,
        video_split: torch.LongTensor | None = None,
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
                    _image_embeds = self._encode_vision(
                        pixel_values=images[im_st:im_ed],
                        image_grid_thw=image_grid_thw[chunk_idx],
                    )
                    image_embeds.append(_image_embeds)

                    im_st = im_ed
                image_embeds = torch.cat(image_embeds, dim=0)
            else:
                image_embeds = self._encode_vision(
                    pixel_values=images,
                    image_grid_thw=image_grid_thw,
                )
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

            image_embeds = self._encode_vision(
                pixel_values=fake_images,
                image_grid_thw=image_grid_thw,
            )
            # image_embeds = image_embeds[:, 1:, :]
            # image_embeds = self.vision_projection(image_embeds)

            # torch.set_printoptions(threshold=100_000)
            # print(f"{image_embeds.size()=}")

        else:
            fake_images = None
            image_embeds = None

        if audios is not None:
            audio_embeds, audio_lengths = self._encode_audio(audios)
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
            audio_embeds, audio_lengths = self._encode_audio(fake_audios)
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

        # ------------------------------------------------------------------
        # Joint video path: same-video vision/audio go through the shared
        # omni encoder together (mirrors ``GPTMMModel._preprocess`` joint
        # branch and ``LanguageModelEmbedding.forward`` independent
        # ``video_*`` scatter in
        # ``vita_megatron/core/models/common/embeddings/language_model_embedding.py``).
        # ------------------------------------------------------------------
        if video_split is not None and video_split.numel() > 0:
            device = inputs_embeds.device
            dtype = inputs_embeds.dtype
            video_images = video_images.to(dtype).to(device)
            video_image_grid_thw = video_image_grid_thw.to(device)
            if video_audios is not None:
                video_audios = [x.to(dtype).to(device) for x in video_audios]
            video_split_dev = video_split.to(device)

            video_image_embeds, video_audio_embeds, video_audio_lens = self._encode_video(
                video_images=video_images,
                video_image_grid_thw=video_image_grid_thw,
                video_audios=video_audios,
                video_split=video_split_dev,
            )

            # Independent scatter for video vision tokens.
            if video_image_indices is not None and video_image_indices.numel() > 0 and video_image_embeds.numel() > 0:
                inputs_embeds = inputs_embeds.clone()
                video_image_embeds = video_image_embeds.to(inputs_embeds.device)
                v_idx = video_image_indices.to(inputs_embeds.device)
                v_indices_b, v_indices_s = v_idx.unbind(dim=0)
                inputs_embeds[v_indices_b.view(-1), v_indices_s.view(-1)] = video_image_embeds.view(-1, video_image_embeds.shape[-1])

            # Independent scatter for video audio tokens.
            if video_audio_indices is not None and len(video_audio_indices) > 0 and video_audio_embeds.numel() > 0:
                inputs_embeds = inputs_embeds.clone()
                for v_aud_emb, v_aud_len, v_aud_idx in zip(
                    video_audio_embeds, video_audio_lens, video_audio_indices
                ):
                    v_aud_emb = v_aud_emb[: int(v_aud_len), ...].to(inputs_embeds.device)
                    indices_b, indices_s = v_aud_idx.to(inputs_embeds.device).unbind(dim=0)
                    inputs_embeds[indices_b.view(-1), indices_s.view(-1)] = v_aud_emb.view(-1, v_aud_emb.shape[-1])

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



class Qwen3VITAForCausalLM(Qwen3VITAPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: Qwen3VITAConfig):
        super().__init__(config)
        self.model = Qwen3VITAModel(config)
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
        # video_omni_fusion joint path
        video_images: torch.FloatTensor | None = None,
        video_image_grid_thw: torch.LongTensor | None = None,
        video_image_indices: torch.LongTensor | None = None,
        video_audios: list | None = None,
        video_audio_indices: list | None = None,
        video_split: torch.LongTensor | None = None,
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
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("meta-Qwen3/Qwen3-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-Qwen3/Qwen3-2-7b-hf")

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
            video_images=video_images,
            video_image_grid_thw=video_image_grid_thw,
            video_image_indices=video_image_indices,
            video_audios=video_audios,
            video_audio_indices=video_audio_indices,
            video_split=video_split,
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
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs)

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
        position_ids: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        images: torch.FloatTensor | None = None,
        image_indices: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        audios: torch.FloatTensor | None = None,
        audio_indices: torch.LongTensor | None = None,
        video_images: torch.FloatTensor | None = None,
        video_image_grid_thw: torch.LongTensor | None = None,
        video_image_indices: torch.LongTensor | None = None,
        video_audios: list | None = None,
        video_audio_indices: list | None = None,
        video_split: torch.LongTensor | None = None,
        is_first_iteration: bool | None = False,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image/audio/video inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            images=images,
            image_indices=image_indices,
            image_grid_thw=image_grid_thw,
            audios=audios,
            audio_indices=audio_indices,
            video_images=video_images,
            video_image_grid_thw=video_image_grid_thw,
            video_image_indices=video_image_indices,
            video_audios=video_audios,
            video_audio_indices=video_audio_indices,
            video_split=video_split,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # After the prefill step the multimodal features have been written into the KV cache,
        # so during decode steps we must clear ALL multimodal tensors (raw inputs, scatter
        # indices and grid metadata) to avoid re-encoding and mismatched scatter writes.
        if not is_first_iteration and use_cache:
            for key in (
                "images",
                "image_indices",
                "image_grid_thw",
                "audios",
                "audio_indices",
                "video_images",
                "video_image_grid_thw",
                "video_image_indices",
                "video_audios",
                "video_audio_indices",
                "video_split",
            ):
                model_inputs[key] = None

        return model_inputs


from dataclasses import dataclass, fields

from ...trainer_pt_utils import LabelSmoother

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


class Qwen3_VITA_TOKEN_bus1(DEFAULT_TOKEN):

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



class Qwen3_VITA_TOKEN(DEFAULT_TOKEN):

    IM_START = "<|begin_of_text|>"
    IM_END = "<|end_of_text|>"
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

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
    
    ACTION_START_TOKEN = "<action>"
    ACTION_END_TOKEN = "</action>"
    OPERATION_START_TOKEN = "<operation>"
    OPERATION_END_TOKEN = "</operation>"
    CLICK_TOKEN = "<click>"
    MOVETO_TOKEN = "<moveTo>"
    SCROLL_TOKEN = "<scroll>"
    WRITE_TOKEN = "<write>"
    DRAGTO_TOKEN = "<dragTo>"
    KEYDOWN_TOKEN = "<keyDown>"
    KEYUP_TOKEN = "<keyUp>"

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
            [
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
            
                self.ACTION_START_TOKEN,
                self.ACTION_END_TOKEN,
                self.OPERATION_START_TOKEN,
                self.OPERATION_END_TOKEN,
                self.CLICK_TOKEN,
                self.MOVETO_TOKEN,
                self.SCROLL_TOKEN,
                self.WRITE_TOKEN,
                self.DRAGTO_TOKEN,
                self.KEYDOWN_TOKEN,
                self.KEYUP_TOKEN,

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
                
            ]
            + [getattr(self, f"{axis}_{i}_TOKEN") for i in range(2048) for axis in ["x", "y"]]
            + [getattr(self, f"custom_{i}_TOKEN") for i in range(1, 1001)]
        )


# _GLOBAL_CONSTANTS = Qwen3_VITA_TOKEN_bus1()
_GLOBAL_CONSTANTS = Qwen3_VITA_TOKEN()


def get_token():
    _ensure_var_is_initialized(_GLOBAL_CONSTANTS, "token")
    return _GLOBAL_CONSTANTS


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, "{} is not initialized.".format(name)


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

    sampling_rate = 16000

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

        self.rank = rank
        logger.info(f"{self.rank=}")

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
        self.device = "cpu"

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
        if not hasattr(self, "audio_decoder"):
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
                    audio, sampling_rate = utt
                else:
                    audio, sampling_rate = torchaudio.load(utt)
                # audio = audio.to(device)
                if sampling_rate != 16000:
                    if sampling_rate not in self._resample_buffer:
                        self._resample_buffer[sampling_rate] = torchaudio.transforms.Resample(
                            orig_freq=sampling_rate, new_freq=16000
                        ) #.to(device)
                    # self._resample_buffer[sampling_rate].to(device)
                    audio = self._resample_buffer[sampling_rate](audio)
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







def update_tokenizer_for_wav_frontend(tokenizer):

    return tokenizer


class WavFrontendTokenizer:
    def __init__(self,):

        self.sampling_rate = 16000

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
            audio, sampling_rate = audio_or_path
        elif isinstance(audio_or_path, str):
            audio, sampling_rate = torchaudio.load(audio_or_path)
        else:
            audio = torch.tensor(audio_or_path)
            sampling_rate = self.sampling_rate
        # print(f"{audio.size()=} {sampling_rate=}")
        if audio.dim() == 2:
            audio = audio.mean(0)

        if sampling_rate != self.sampling_rate:
            if sampling_rate not in self._resample_buffer:
                # print(f"torchaudio.transforms.Resample {sampling_rate=} {self.sampling_rate=} {self.device=}", flush=True)
                self._resample_buffer[sampling_rate] = torchaudio.transforms.Resample(
                    orig_freq=sampling_rate, new_freq=self.sampling_rate
                ).to(self.device)
            audio = audio.to(self.device)
            self._resample_buffer[sampling_rate].to(self.device)
            audio = self._resample_buffer[sampling_rate](audio[None, :])[0, :]
            audio = audio.cpu()
        # resampler = torchaudio.transforms.Resample(
        #     orig_freq=sampling_rate, new_freq=self.sampling_rate
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
            "audio_token_length_func": lambda x: x,
            "duration_seconds": len(audio) / self.sampling_rate,
        }

    @torch.no_grad()
    def decode(self, audio_tokens, **kwargs):
        return None


def update_tokenizer_for_melfilterbank(tokenizer):
    return tokenizer


class MelFilterBankTokenizer:
    def __init__(self, model_name_or_path, rank=None):
        self.model_name_or_path = model_name_or_path

        self.rank = rank
        logger.info(f"{self.rank=}")

        self.sampling_rate = 16000

        self.is_discrete = True
        self.is_contiguous = False

        self._resample_buffer: dict[int, torchaudio.transforms.Resample] = {}

        self.tokenizer_type = "melfilterbank"

    def load_model(self):
        if hasattr(self, "feature_extractor"):
            return

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            # num_workers = worker_info.num_workers
            self.rank = worker_id % 8
            logger.info(f"{self.rank=}")

        if self.rank is not None:
            self.device = f"cuda:{self.rank}"
            # torch.cuda.set_device(self.rank)
        else:
            self.device = "cuda"
            # self.device = "cpu"
        self.device = "cpu"

        logger.info(f"{self.device=}")

        assert isinstance(self.model_name_or_path, str)

        logger.info(
            f"⏳ {self.device=} Loading {self.tokenizer_type} from {self.model_name_or_path}"
        )
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.model_name_or_path
        )
        logger.info(f"⏳ {self.device=} Loading {self.tokenizer_type} Done")

    @torch.no_grad()
    def encode(self, audio_or_path, **kwargs):
        if not hasattr(self, "feature_extractor"):
            self.load_model()

        if isinstance(audio_or_path, tuple):
            audio, sampling_rate = audio_or_path
        elif isinstance(audio_or_path, str):
            audio, sampling_rate = torchaudio.load(audio_or_path)
        else:
            audio = torch.tensor(audio_or_path)
            sampling_rate = self.sampling_rate
        # print(f"{audio_or_path=} {audio.size()=} {sampling_rate=}")
        if audio.dim() == 2:
            audio = audio.mean(0)

        if sampling_rate != self.sampling_rate:
            if sampling_rate not in self._resample_buffer:
                # print(f"torchaudio.transforms.Resample {sampling_rate=} {self.sampling_rate=} {self.device=}", flush=True)
                self._resample_buffer[sampling_rate] = torchaudio.transforms.Resample(
                    orig_freq=sampling_rate, new_freq=self.sampling_rate
                ).to(self.device)
            audio = audio.to(self.device)
            self._resample_buffer[sampling_rate].to(self.device)
            audio = self._resample_buffer[sampling_rate](audio[None, :])[0, :]
            audio = audio.cpu()
        # resampler = torchaudio.transforms.Resample(
        #     orig_freq=sampling_rate, new_freq=self.sampling_rate
        # )
        # audio = resampler(audio[None, :])[0, :]
        # audio = audio.to(self.device)

        # print(f"{audio_or_path=} {audio.size()=} {sampling_rate=}")

        features = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
            padding=True,
            device=self.device,
        )
        input_features = features["input_features"]
        # feature_attention_mask = features["attention_mask"]

        return {
            "audio": input_features.squeeze(0).permute(1, 0),
            "audio_token_length_func": _get_feat_extract_output_lengths,
            "duration_seconds": len(audio) / self.sampling_rate,
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
            raise NotImplementedError(f"Unsupported audio tokenizer types: {audio_tokenizer_type_list}")

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
            raise NotImplementedError(f"Unsupported vision tokenizer types: {vision_tokenizer_type_list}")

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

    if isinstance(model_name_or_path_list, str):
        model_name_or_path_list = model_name_or_path_list.split()

    if isinstance(audio_tokenizer_type_list, str):
        audio_tokenizer_type_list = audio_tokenizer_type_list.split()

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
            raise NotImplementedError(f"Unsupported audio tokenizer types: {audio_tokenizer_type_list}")

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
            raise NotImplementedError(f"Unsupported vision tokenizer types: {vision_tokenizer_type_list}")

    vision_tokenizer = VisionTokenizer(tokenizer_contiguous, tokenizer_discrete)
    return vision_tokenizer


class Qwen3VITAImagesKwargs(ImagesKwargs, total=False):
    """
    """
    discrete_image_idxs: list
    contiguous_image_idxs: list

    vision_normalize_type: str
    image_min_num_tokens: int
    image_max_num_tokens: int


class Qwen3VITAAudioKwargs(AudioKwargs, total=False):
    """
    """
    discrete_audio_idxs: list
    contiguous_audio_idxs: list

    # temporal_merge_size: int

    # audio_tokenizer_type: str
    # audio_tokenizer_path: str

class Qwen3VITAVideosKwargs(VideosKwargs, total=False):
    """
    """
    video_min_num_tokens: int
    video_max_num_tokens: int
    video_image_min_num_tokens: int
    video_image_max_num_tokens: int
    video_max_num_frames: int
    temporal_patch_size: int
    spatial_merge_size: int
    patch_size: int
    video_key_frame: bool
    use_audio_in_video: bool
    use_vision_in_video: bool
    # When True, video frames+audio of the same video are jointly encoded by
    # ``Qwen3VITAOmniModel.forward_video`` so they can attend to each other.
    # When False (default), video frames/audio fall back to the standalone
    # image/audio paths.
    video_omni_fusion: bool


class Qwen3VITAProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Qwen3VITAImagesKwargs
    videos_kwargs: Qwen3VITAVideosKwargs
    audio_kwargs: Qwen3VITAAudioKwargs

    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
        "images_kwargs": {
            # "vision_normalize_type": "siglip",
            # "image_min_num_tokens": 4,
            # "image_max_num_tokens": 8192,
        },
        "videos_kwargs": {
            # "video_min_num_tokens": 64,
            # "video_max_num_tokens": 8192,
            # "video_image_min_num_tokens": 4,
            # "video_image_max_num_tokens": 256,
            # "video_max_num_frames": 64,
            # "temporal_patch_size": 1,
            # "spatial_merge_size": 2,
            # "patch_size":16,
            # "video_key_frame": False,
            # "use_audio_in_video": True,
            # "use_vision_in_video": True,
            # "video_omni_fusion": False,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": "max_length",
            "return_attention_mask": True,
            # "temporal_merge_size": 1,
        },
    }


class Qwen3VITAFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["audios", "audio_indices"]
    valid_kwargs = Qwen3VITAAudioKwargs

    def __init__(
        self,
        audio_tokenizer_path=None,
        audio_tokenizer_type=None,
        flow_path=None,
        rank=None,
        text_audio_interval_ratio=None,
        audio_chunk_min_second=2,
        audio_chunk_max_second=30,
        temporal_merge_size=1,
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
        self.audio_chunk_min_second = audio_chunk_min_second
        self.audio_chunk_max_second = audio_chunk_max_second
        self.temporal_merge_size = temporal_merge_size

        # self.load_model()

    def to_dict(self):
        output = super().to_dict()
        # Remove the non-serializable object before returning
        if "audio_tokenizer" in output:
            del output["audio_tokenizer"]
        return output


    def load_model(self):
        self.audio_tokenizer.load_model()

    def process_audio(
        self, audio_or_path, is_discrete=False, is_contiguous=False, **kwargs
    ):
        assert not (is_discrete and is_contiguous)
        assert is_discrete or is_contiguous

        if is_discrete and self.audio_tokenizer.tokenizer_discrete is not None:
            if isinstance(audio_or_path, str):
                tokenizer_type = self.audio_tokenizer.tokenizer_discrete.tokenizer_type
                cache_path = (
                    os.path.splitext(audio_or_path)[0] + f"-{tokenizer_type}.json"
                )

                audio_data = load_cache(cache_path)
                if audio_data is not None:
                    return audio_data

            audio_data = self.audio_tokenizer.encode(
                audio_or_path,
                is_discrete=is_discrete,
                is_contiguous=is_contiguous,
                **kwargs,
            )
            # print(f"{len(audio_data)=}")

            if isinstance(audio_or_path, str):
                save_cache(audio_data, cache_path)

            return audio_data

        if is_contiguous:
            audio_dict = self.audio_tokenizer.encode(
                audio_or_path,
                is_discrete=is_discrete,
                is_contiguous=is_contiguous,
                **kwargs,
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

    def add_audio_input_discrete_or_contiguous(
        self,
        input_ids,
        audio_or_paths,
        tokenizer,
        discrete_audio_idxs=[],
        contiguous_audio_idxs=[],
        targets=None,
        is_pretrain=False,
        **kwargs,
    ):
        audio_chunk_min_second = kwargs.get("audio_chunk_min_second", self.audio_chunk_min_second)
        audio_chunk_max_second = kwargs.get("audio_chunk_max_second", self.audio_chunk_max_second)

        GLOBAL_CONSTANTS = get_token()

        AUD_CONTEXT_ID = tokenizer.convert_tokens_to_ids(GLOBAL_CONSTANTS.AUD_CONTEXT_TOKEN)
        AUD_TAG_ID = tokenizer.convert_tokens_to_ids(GLOBAL_CONSTANTS.AUD_TAG_TOKEN)
        AUD_START_ID = tokenizer.convert_tokens_to_ids(GLOBAL_CONSTANTS.AUD_START_TOKEN)
        AUD_END_ID = tokenizer.convert_tokens_to_ids(GLOBAL_CONSTANTS.AUD_END_TOKEN)

        if self.audio_tokenizer.tokenizer_discrete is not None:
            AUD_FIRST_ID = tokenizer.convert_tokens_to_ids(
                self.audio_tokenizer.tokenizer_discrete.first_audio_token
            )

        aud_positions = [i for i, x in enumerate(input_ids) if x == AUD_TAG_ID]
        assert len(aud_positions) == len(audio_or_paths), (
            f"{len(aud_positions)=} {len(audio_or_paths)=} {AUD_TAG_ID=}"
        )

        if not discrete_audio_idxs and not contiguous_audio_idxs:
            contiguous_audio_idxs = list(range(len(audio_or_paths)))

        audios = []
        audio_indices = []
        new_input_ids = []
        new_targets = []
        if targets is not None and self.audio_tokenizer.num_codebook > 1:
            # additional_targets_list = [[] for _ in range(self.audio_tokenizer.num_codebook - 1)]
            additional_targets_list = [
                [] for _ in range(self.audio_tokenizer.num_codebook)
            ]
        else:
            additional_targets_list = None

        st = 0
        for aud_idx, aud_pos in enumerate(aud_positions):
            new_input_ids += input_ids[st:aud_pos]
            if targets is not None:
                new_targets += targets[st:aud_pos]
            if additional_targets_list is not None:
                additional_targets_list = [
                    x + [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID] * (aud_pos - st)
                    for x in additional_targets_list
                ]

            # --------------------------------------------------------------------------
            # add discrete
            if aud_idx in discrete_audio_idxs:
                audio_tokens = self.process_audio(
                    audio_or_paths[aud_idx], is_discrete=True
                )
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

                        audio_tokens_ = tokenizer(
                            audio_tokens_, add_special_tokens=False
                        ).input_ids
                        if i > 0:
                            CB_AUD_FIRST_ID = tokenizer.convert_tokens_to_ids(
                                f"<|audio_{i}_0|>"
                            )
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
                    additional_targets_list = [
                        x + [AUD_START_ID] for x in additional_targets_list
                    ]

                new_input_ids += audio_tokens
                if targets is not None:
                    new_targets += audio_tokens
                if additional_targets_list is not None:
                    additional_targets_list = [
                        x + y
                        for x, y in zip(
                            additional_targets_list, additional_audio_tokens
                        )
                    ]

                new_input_ids += [AUD_END_ID]
                if targets is not None:
                    new_targets += [AUD_END_ID]
                if additional_targets_list is not None:
                    additional_targets_list = [
                        x + [AUD_END_ID] for x in additional_targets_list
                    ]

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
                        audio_second_chunks.append(
                            sum([len(x) * second_per_audio for x in audio_chunks])
                        )
                        audio_chunks.append(torch.stack(audio_chunk, dim=0))

                        audio_chunk = []
                        audio_chunk.append(audio_frame)

                if len(audio_chunk) > 0:
                    audio_second_chunks.append(
                        sum([len(x) * second_per_audio for x in audio_chunks])
                    )
                    audio_chunks.append(torch.stack(audio_chunk, dim=0))

                audios.extend(audio_chunks)
                # print(f"{audios=}")

                timestamp_format = "HHMMSS"

                for audio, audio_second in zip(audio_chunks, audio_second_chunks):
                    # add timestamp
                    if len(audio_chunks) > 1:
                        if timestamp_format == "HHMMSS":
                            audio_second = round(audio_second)
                            timestamp = time.strftime(
                                "%H:%M:%S", time.gmtime(audio_second)
                            )
                        else:
                            timestamp = f"{audio_second:.2f}"
                        _input_id = tokenizer(
                            timestamp, add_special_tokens=False
                        ).input_ids
                        new_input_ids += _input_id
                        if targets is not None:
                            if is_pretrain:
                                new_targets += _input_id
                            else:
                                new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID] * len(
                                    _input_id
                                )

                    new_input_ids += [AUD_START_ID]
                    if targets is not None:
                        if is_pretrain:
                            new_targets += [AUD_START_ID]
                        else:
                            new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID]
                    if additional_targets_list is not None:
                        additional_targets_list = [
                            x + [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID]
                            for x in additional_targets_list
                        ]

                    audio_token_length = -(
                        -audio_token_length_func(len(audio)) // self.temporal_merge_size
                    )
                    assert audio_token_length > 0

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
                        new_targets += [
                            GLOBAL_CONSTANTS.IGNORE_TOKEN_ID
                        ] * audio_token_length
                    if additional_targets_list is not None:
                        additional_targets_list = [
                            x + [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID] * audio_token_length
                            for x in additional_targets_list
                        ]

                    new_input_ids += [AUD_END_ID]
                    if targets is not None:
                        if is_pretrain:
                            new_targets += [AUD_END_ID]
                        else:
                            new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID]
                    if additional_targets_list is not None:
                        additional_targets_list = [
                            x + [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID]
                            for x in additional_targets_list
                        ]

            st = aud_pos + 1

        new_input_ids += input_ids[st:]
        if targets is not None:
            new_targets += targets[st:]
        if additional_targets_list is not None:
            additional_targets_list = [
                x + [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID] * (len(targets) - st)
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


class Qwen3VITAVideoProcessor(BaseVideoProcessor):
    model_input_names = ["pixel_values", "image_grid_thw"]
    valid_kwargs = Qwen3VITAVideosKwargs


    def __init__(
        self,
        image_processor=None,
        audio_processor=None,
        video_max_num_frames=64,
        video_max_fps=1,
        video_min_num_tokens=64,
        video_max_num_tokens=8192,
        video_image_min_num_tokens=4,
        video_image_max_num_tokens=256,
        video_audio_chunk_min_second=2,
        video_audio_chunk_max_second=30,
        use_audio_in_video=True,
        use_vision_in_video=True,
        temporal_patch_size=1,
        spatial_merge_size=2,
        temporal_merge_size=1,
        patch_size=14,
        video_key_frame=False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.image_processor = image_processor
        self.audio_processor = audio_processor

        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_merge_size = temporal_merge_size
        self.patch_size = patch_size

        self.video_max_num_frames = video_max_num_frames
        self.video_max_fps = video_max_fps
        self.video_max_num_tokens = video_max_num_tokens
        self.video_min_num_tokens = video_min_num_tokens
        self.video_image_max_num_tokens = video_image_max_num_tokens
        self.video_image_min_num_tokens = video_image_min_num_tokens
        self.video_audio_chunk_min_second = video_audio_chunk_min_second
        self.video_audio_chunk_max_second = video_audio_chunk_max_second
        self.use_audio_in_video = use_audio_in_video
        self.use_vision_in_video = use_vision_in_video

        self.sampling_rate = 16000

        self.video_key_frame = video_key_frame

    def to_dict(self):
        output = super().to_dict()
        # Remove the non-serializable object before returning
        if "image_processor" in output:
            del output["image_processor"]
        if "audio_processor" in output:
            del output["audio_processor"]
        return output

    def get_video_frames(self, vid_path, video_max_fps=1, video_max_num_frames=8):
        vid = decord.VideoReader(vid_path, num_threads=1)

        fps = vid.get_avg_fps()
        fps = round(fps)

        if self.video_key_frame:
            indices = get_video_keyframe(vid_path)
            # TODO: fix this
            sample_fps = None
            sample_fps = 1
            if len(indices) > video_max_num_frames:
                random_indices = sorted(random.sample(range(len(indices)), video_max_num_frames))
                indices = [indices[i] for i in random_indices]
        else:
            # step_size = len(vid) / (video_max_num_frames + 1)
            step_size = len(vid) / video_max_num_frames
            step_size = max(fps / video_max_fps, step_size)

            # indices = [int(i * step_size) for i in range(1, video_max_num_frames + 1)]
            indices = [int(i * step_size) for i in range(0, video_max_num_frames)]
            sample_fps = 1 / (step_size / fps)

        indices = [i for i in indices if i < len(vid)]

        images = [vid[i].asnumpy() for i in indices]
        images = [PIL.Image.fromarray(arr) for arr in images]

        timestamps = [1.0 / fps * i for i in indices]

        duration_seconds = len(vid) / vid.get_avg_fps()

        # print(f"get_video_frames vid_path {vid_path} fps {fps} len(vid) {len(vid)} frame_paths {frame_paths}")
        return images, sample_fps, timestamps, duration_seconds

    def get_image_and_audio(self, video_file_or_dir, video_max_num_frames=8, video_max_fps=1):

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
            img_or_path_list, fps, timestamps, duration_seconds = self.get_video_frames(
                video_file_or_dir, video_max_num_frames=video_max_num_frames, video_max_fps=video_max_fps
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
            target_frame = int(min(total_frames / fps * video_max_fps, video_max_num_frames))
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
        try:
            audio, sampling_rate = torchaudio.load(video_file_or_dir)
            # print(f"{audio.size()=} {sampling_rate=}")
            if audio.dim() == 2:
                audio = audio.mean(0)

            if sampling_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sampling_rate, new_freq=self.sampling_rate
                )
                audio = resampler(audio[None, :])[0, :]
        except Exception as e:
            pass

        return img_or_path_list, fps, timestamps, (audio, self.sampling_rate), duration_seconds

    def process_video(self, video_file_or_dir, video_max_num_frames=8, video_max_fps=1):

        images, fps, timestamps, (audio, sampling_rate), duration_seconds = self.get_image_and_audio(
            video_file_or_dir,
            video_max_num_frames=video_max_num_frames,
            video_max_fps=video_max_fps,
        )

        min_pixels = (
            (self.patch_size * self.spatial_merge_size) ** 2
            * self.video_min_num_tokens
            // len(images)
        )
        max_pixels = (
            (self.patch_size * self.spatial_merge_size) ** 2
            * self.video_max_num_tokens
            // len(images)
        )

        image_min_pixels = (
            self.patch_size * self.spatial_merge_size
        ) ** 2 * self.video_image_min_num_tokens
        image_max_pixels = (
            self.patch_size * self.spatial_merge_size
        ) ** 2 * self.video_image_max_num_tokens

        min_pixels = max(min_pixels, image_min_pixels)
        max_pixels = min(max_pixels, image_max_pixels)

        # print(f"{len(images)=} {min_pixels=} {max_pixels=}")
        image_data = self.image_processor.process_images(
            images,
            is_contiguous=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        image_frames = image_data["images"]
        best_height = image_data["image_height"]
        best_width = image_data["image_width"]

        if audio is not None:
            total_time = len(audio) / sampling_rate
            # print(f"{duration_seconds=} {total_time=}", flush=True)

            audio_dict = self.audio_processor.process_audio(
                (audio, sampling_rate), is_discrete=False, is_contiguous=True
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

        # print(f"{image_frames.size()=} {video_grid_thw=} {video_max_fps=} {video_max_num_frames=} {fps=} {second_per_grids=} ")

        return (
            image_frames,
            audio_frames,
            audio_token_length_func,
            video_grid_thw,
            second_per_grids,
            timestamps,
            duration_seconds,
        )

    @staticmethod
    def _chunk_audio_video_frames(
        image_frames,
        audio_frames,
        second_frames,
        duration_seconds,
        video_audio_chunk_min_second,
        video_audio_chunk_max_second,
    ):
        """Split image/audio frames into time-aligned chunks based on audio
        duration limits. Mirrors :meth:`VideoProcessor._chunk_audio_video_frames`
        in ``cognitron_mm/processor/video_processor.py``.

        Returns:
            ``(image_chunks, image_second_chunks, audio_chunks, audio_second_chunks)``
        """
        if audio_frames is None:
            return [image_frames], [second_frames], [[]], [[]]

        second_per_audio = 1.0 * duration_seconds / sum(len(x) for x in audio_frames)

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
            audio_chunk_second = sum(len(x) * second_per_audio for x in audio_chunk)

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

        return image_chunks, image_second_chunks, audio_chunks, audio_second_chunks

    def add_video_input_discrete_or_contiguous(
        self,
        input_ids,
        video_paths,
        tokenizer,
        targets=None,
        discrete_video_idxs=[],
        contiguous_video_idxs=[],
        is_pretrain=False,
        **kwargs,
    ):
        video_max_num_frames = kwargs.get("video_max_num_frames", self.video_max_num_frames)
        video_max_fps = kwargs.get("video_max_fps", self.video_max_fps)
        use_audio_in_video = kwargs.get("use_audio_in_video", self.use_audio_in_video)
        use_vision_in_video = kwargs.get("use_vision_in_video", self.use_vision_in_video)
        video_audio_chunk_min_second = kwargs.get("video_audio_chunk_min_second", self.video_audio_chunk_min_second)
        video_audio_chunk_max_second = kwargs.get("video_audio_chunk_max_second", self.video_audio_chunk_max_second)

        GLOBAL_CONSTANTS = get_token()

        IMG_CONTEXT_ID = tokenizer.convert_tokens_to_ids(GLOBAL_CONSTANTS.IMG_CONTEXT_TOKEN)
        IMG_START_ID = tokenizer.convert_tokens_to_ids(GLOBAL_CONSTANTS.IMG_START_TOKEN)
        IMG_END_ID = tokenizer.convert_tokens_to_ids(GLOBAL_CONSTANTS.IMG_END_TOKEN)

        AUD_CONTEXT_ID = tokenizer.convert_tokens_to_ids(GLOBAL_CONSTANTS.AUD_CONTEXT_TOKEN)
        AUD_START_ID = tokenizer.convert_tokens_to_ids(GLOBAL_CONSTANTS.AUD_START_TOKEN)
        AUD_END_ID = tokenizer.convert_tokens_to_ids(GLOBAL_CONSTANTS.AUD_END_TOKEN)

        VID_CONTEXT_ID = tokenizer.convert_tokens_to_ids(GLOBAL_CONSTANTS.VID_CONTEXT_TOKEN)
        VID_START_ID = tokenizer.convert_tokens_to_ids(GLOBAL_CONSTANTS.VID_START_TOKEN)
        VID_END_ID = tokenizer.convert_tokens_to_ids(GLOBAL_CONSTANTS.VID_END_TOKEN)

        IMG_TAG_ID = tokenizer.convert_tokens_to_ids(GLOBAL_CONSTANTS.IMG_TAG_TOKEN)
        AUD_TAG_ID = tokenizer.convert_tokens_to_ids(GLOBAL_CONSTANTS.AUD_TAG_TOKEN)
        VID_TAG_ID = tokenizer.convert_tokens_to_ids(GLOBAL_CONSTANTS.VID_TAG_TOKEN)

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
        # Per-video splits ``(num_images, num_audios)``, consumed downstream
        # by the ``video_omni_fusion`` joint-encode path. Mirrors
        # :meth:`VideoProcessor.add_video_input_contiguous` in
        # ``cognitron_mm/processor/video_processor.py``.
        video_split = []

        new_input_ids = []
        new_targets = []
        st = 0
        for vid_idx, vid_pos in enumerate(vid_positions):
            # Snapshot per-video accounting before processing this video.
            num_images_before = len(video_grid_thw)
            num_audios_before = len(audios)
            (
                image_frames,
                audio_frames,
                audio_token_length_func,
                _video_grid_thw,
                _second_per_grids,
                second_frames,
                duration_seconds,
            ) = self.process_video(video_paths[vid_idx], video_max_num_frames, video_max_fps)

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

            (
                image_chunks,
                image_second_chunks,
                audio_chunks,
                audio_second_chunks,
            ) = self._chunk_audio_video_frames(
                image_frames,
                audio_frames,
                second_frames,
                duration_seconds,
                video_audio_chunk_min_second,
                video_audio_chunk_max_second,
            )

            assert len(image_frames) == sum([len(x) for x in image_chunks])
            # assert len(audio_frames) == sum([len(x) for x in audio_chunks])

            if use_vision_in_video:
                images.append(
                    torch.cat(
                        [
                            self.image_processor.convert_image_to_patches_with_pixel_shuffle(x)
                            for x in image_frames
                        ],
                        dim=0,
                    )
                )

            if use_audio_in_video and audio_frames is not None:
                # audios.extend(audio_frames)
                audios.extend([xx for x in audio_chunks for xx in x])

            new_input_ids += [VID_START_ID]
            if targets is not None:
                new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID]

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
                            new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID] * len(_input_id)

                    if vid_idx in contiguous_video_idxs:
                        new_input_ids += [IMG_START_ID]
                        if targets is not None:
                            if is_pretrain:
                                new_targets += [IMG_START_ID]
                            else:
                                new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID]

                        resolution = f"{_video_grid_thw[0][1] * self.patch_size}*{_video_grid_thw[0][2] * self.patch_size}"
                        _input_id = tokenizer(resolution, add_special_tokens=False).input_ids
                        new_input_ids += _input_id
                        if targets is not None:
                            if is_pretrain:
                                # new_targets += _input_id
                                new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID] * len(_input_id)
                            else:
                                new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID] * len(_input_id)

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
                                    GLOBAL_CONSTANTS.IGNORE_TOKEN_ID
                                ] * image_token_length

                            new_input_ids += nl_tokens
                            if targets is not None:
                                if is_pretrain:
                                    new_targets += nl_tokens
                                else:
                                    new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID] * len(
                                        nl_tokens
                                    )

                        new_input_ids += [IMG_END_ID]
                        if targets is not None:
                            if is_pretrain:
                                new_targets += [IMG_END_ID]
                            else:
                                new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID]

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
                            new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID] * len(_input_id)

                    if vid_idx in contiguous_video_idxs:
                        new_input_ids += [AUD_START_ID]
                        if targets is not None:
                            if is_pretrain:
                                new_targets += [AUD_START_ID]
                            else:
                                new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID]

                        audio_token_length = -(-audio_token_length_func(len(audio_chunk_frame)) // self.temporal_merge_size)
                        assert audio_token_length > 0
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
                            new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID] * audio_token_length

                        new_input_ids += [AUD_END_ID]
                        if targets is not None:
                            if is_pretrain:
                                new_targets += [AUD_END_ID]
                            else:
                                new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID]

                    if vid_idx in discrete_video_idxs:
                        raise NotImplementedError

            new_input_ids += [VID_END_ID]
            if targets is not None:
                if is_pretrain:
                    new_targets += [VID_END_ID]
                else:
                    new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID]

            video_grid_thw.extend(_video_grid_thw)
            second_per_grids.extend(_second_per_grids)

            # Record per-video split delta after processing this video.
            video_split.append(
                (
                    len(video_grid_thw) - num_images_before,
                    len(audios) - num_audios_before,
                )
            )

            st = vid_pos + 1

        new_input_ids += input_ids[st:]
        if targets is not None:
            new_targets += targets[st:]

        input_ids = new_input_ids
        if targets is not None:
            targets = new_targets

        video_grid_thw = torch.tensor(video_grid_thw, dtype=torch.long)
        second_per_grids = torch.tensor(second_per_grids, dtype=torch.long)

        # Per-video split metadata, consumed by the joint-video encoder when
        # ``video_omni_fusion`` is enabled. ``None`` when no videos.
        video_split_out = video_split if len(video_split) > 0 else None

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
                video_split_out,
            )

        if len(images) == 0:
            images = None
            image_indices = None
        else:
            images = torch.cat(images, dim=0).contiguous()
            image_indices = torch.cat(image_indices, dim=1).contiguous()

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
            video_split_out,
        )


class Qwen3VITAImageProcessor(BaseImageProcessor):
    model_input_names = ["images", "image_indices", "image_grid_thw"]
    valid_kwargs = Qwen3VITAImagesKwargs

    def __init__(
        self,
        image_size=448,
        image_size_discrete=None,
        vision_normalize_type="imagenet",
        min_tile_grid=1,
        max_tile_grid=6,
        image_min_num_tokens=4,
        image_max_num_tokens=256,
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
        self.min_tile_grid = min_tile_grid
        self.max_tile_grid = max_tile_grid
        self.tile_image_size = image_size
        self.image_max_num_tokens = image_max_num_tokens
        self.image_min_num_tokens = image_min_num_tokens

        GLOBAL_CONSTANTS = get_token()
        if vision_normalize_type == "imagenet":
            MEAN, STD = GLOBAL_CONSTANTS.IMAGENET_DEFAULT_MEAN, GLOBAL_CONSTANTS.IMAGENET_DEFAULT_STD
        elif vision_normalize_type == "clip":
            MEAN, STD = GLOBAL_CONSTANTS.OPENAI_CLIP_MEAN, GLOBAL_CONSTANTS.OPENAI_CLIP_STD
        elif vision_normalize_type == "siglip":
            MEAN, STD = GLOBAL_CONSTANTS.IMAGENET_STANDARD_MEAN, GLOBAL_CONSTANTS.IMAGENET_STANDARD_STD
        else:
            raise NotImplementedError(vision_normalize_type)
        self.mean = MEAN
        self.std = STD

        self.min_pixels = (patch_size * spatial_merge_size) ** 2 * image_min_num_tokens
        self.max_pixels = (patch_size * spatial_merge_size) ** 2 * image_max_num_tokens
        logger.info(f"{self.min_pixels=} {self.max_pixels=}")

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
            return self.process_native(image_or_path, **kwargs)

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
        #     max_num_patches=self.image_max_num_tokens,
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

        GLOBAL_CONSTANTS = get_token()

        IMG_CONTEXT_ID = tokenizer.convert_tokens_to_ids(GLOBAL_CONSTANTS.IMG_CONTEXT_TOKEN)
        IMG_START_ID = tokenizer.convert_tokens_to_ids(GLOBAL_CONSTANTS.IMG_START_TOKEN)
        IMG_END_ID = tokenizer.convert_tokens_to_ids(GLOBAL_CONSTANTS.IMG_END_TOKEN)
        IMG_TAG_ID = tokenizer.convert_tokens_to_ids(GLOBAL_CONSTANTS.IMG_TAG_TOKEN)

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
                image_data = self.process_image(
                    image_or_paths[img_idx],
                    is_discrete=True,
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

                images.append(
                    torch.cat(
                        [
                            self.convert_image_to_patches_with_pixel_shuffle(x)
                            for x in image_patches
                        ],
                        dim=0,
                    )
                )

                new_input_ids += [IMG_START_ID]
                if targets is not None:
                    if is_pretrain:
                        new_targets += [IMG_START_ID]
                    else:
                        new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID]

                resolution = f"{_image_grid_thw[0][1] * self.patch_size}*{_image_grid_thw[0][2] * self.patch_size}"
                size_input_id = tokenizer(resolution, add_special_tokens=False).input_ids
                new_input_ids += size_input_id
                if targets is not None:
                    if is_pretrain:
                        # new_targets += size_input_id
                        new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID] * len(size_input_id)
                    else:
                        new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID] * len(size_input_id)

                new_input_ids += nl_tokens
                if targets is not None:
                    new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID] * len(nl_tokens)

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
                        new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID] * image_token_length

                    new_input_ids += nl_tokens
                    if targets is not None:
                        if is_pretrain:
                            new_targets += nl_tokens
                        else:
                            new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID] * len(nl_tokens)

                new_input_ids += [IMG_END_ID]
                if targets is not None:
                    if is_pretrain:
                        new_targets += [IMG_END_ID]
                    else:
                        new_targets += [GLOBAL_CONSTANTS.IGNORE_TOKEN_ID]

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
                torch.tensor(images, dtype=torch.float32)
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



class Qwen3VITAProcessor(ProcessorMixin):
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
        images: ImageInput | None = None,
        videos: VideoInput | None = None,
        audio: AudioInput | None = None,
        **kwargs: Unpack[Qwen3VITAProcessorKwargs],
    ) -> BatchFeature:
        audios = audio
        logger.debug(f"{text=}")
        logger.debug(f"{images=}")
        logger.debug(f"{videos=}")
        logger.debug(f"{audios=}")
        logger.debug(f"{kwargs=}")

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        output_kwargs = self._merge_kwargs(
            Qwen3VITAProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        logger.debug(f"{output_kwargs=}")

        output_kwargs["text_kwargs"].pop("return_tensors", None)
        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        input_ids = texts_inputs["input_ids"]

        images_inputs = {}
        videos_inputs = {}
        audio_inputs = {}

        if audios:
            input_ids, _audios, audio_indices = self.audio_processor.add_audio_input_discrete_or_contiguous(
                input_ids,
                audios,
                self.tokenizer,
                discrete_audio_idxs=kwargs.get("discrete_audio_idxs", []),
                **output_kwargs["audio_kwargs"],
            )

            audio_seqlens = [len(x) for x in _audios]

            logger.debug(f"{audios=} {len(input_ids)=} {len(_audios)=} {sum(x.abs().sum() for x in _audios)=} {len(audio_indices)=}")

            audio_inputs["audios"] = _audios
            audio_inputs["audio_indices"] = audio_indices
            # audio_inputs["audio_feature_lengths"] = audio_seqlens

        if images:
            if (isinstance(images, (list, tuple)) and all(isinstance(images_i, (list, tuple)) for images_i in images)):
                images = [img for img_list in images for img in img_list]
            input_ids, _images, image_indices, image_grid_thw = self.image_processor.add_image_input_discrete_or_contiguous(
                input_ids,
                images,
                self.tokenizer,
                **output_kwargs["images_kwargs"],
            )
            logger.debug(f"{images=} {len(input_ids)=} {_images.size()=} {image_indices.size()=} {image_grid_thw=}")

            images_inputs["images"] = _images
            images_inputs["image_indices"] = image_indices
            images_inputs["image_grid_thw"] = image_grid_thw

        if videos:
            if (isinstance(videos, (list, tuple)) and all(isinstance(videos_i, (list, tuple)) for videos_i in videos)):
                videos = [vid for vid_list in videos for vid in vid_list]

            # Joint-encode switch: when True, same-video vision/audio go
            # through ``Qwen3VITAOmniModel.forward_video`` together. When
            # False (default), video frames/audio are fed into the
            # standalone image/audio paths (current behavior).
            video_omni_fusion = output_kwargs["videos_kwargs"].pop(
                "video_omni_fusion", False
            )

            (
                input_ids,
                _images,
                image_indices,
                _audios,
                audio_indices,
                image_grid_thw,
                second_per_grids,
                video_split,
            ) = self.video_processor.add_video_input_discrete_or_contiguous(
                input_ids,
                videos,
                self.tokenizer,
                **output_kwargs["videos_kwargs"],
            )
            if _images is not None:
                logger.debug(f"{len(input_ids)=} {_images.size()=} {image_indices.size()=} {image_grid_thw.size()=}")
            if _audios is not None:
                logger.debug(f"{len(input_ids)=} {len(_audios)=} {[x.size() for x in _audios]=} {len(audio_indices)=}")

            if _audios is None:
                audio_seqlens = None
            else:
                audio_seqlens = [len(x) for x in _audios]

            if video_omni_fusion and _images is not None:
                # Joint-encode mode: emit dedicated ``video_*`` keys so the
                # model's joint forward path picks them up. ``video_audios``
                # / ``video_audio_indices`` keep the list form used by the
                # standalone audio path.
                videos_inputs["video_images"] = _images
                videos_inputs["video_image_grid_thw"] = image_grid_thw
                videos_inputs["video_image_indices"] = image_indices
                if _audios is not None:
                    videos_inputs["video_audios"] = _audios
                    videos_inputs["video_audio_indices"] = audio_indices
                videos_inputs["video_split"] = torch.tensor(video_split, dtype=torch.long)
            else:
                # Independent mode (default, current behavior unchanged):
                # video frames/audio are fed into the standalone image/audio
                # paths via the ``images`` / ``audios`` keys.
                videos_inputs["images"] = _images
                videos_inputs["image_indices"] = image_indices
                videos_inputs["audios"] = _audios
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
    "Qwen3VITAConfig",
    "Qwen3VITAOmniConfig",
    "Qwen3VITAPreTrainedModel",
    "Qwen3VITAModel",
    "Qwen3VITAForCausalLM",
    "Qwen3VITAOmniPreTrainedModel",
    "Qwen3VITAOmniEncoder",
    "Qwen3VITAOmniModel",
    "Qwen3VITAProcessor",
    "Qwen3VITAImageProcessor",
    "Qwen3VITAVideoProcessor",
    "Qwen3VITAFeatureExtractor",
]

