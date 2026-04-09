
"""PyTorch Mamba3-VITA model."""

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
from ...utils import TransformersKwargs, is_decord_available, is_flash_attn_2_available, is_torchaudio_available, logging
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

from ...utils.import_utils import is_mamba_ssm_available

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb

if is_decord_available():
    import decord

if is_torchaudio_available():
    import torchaudio

if is_mamba_ssm_available():
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo as mamba3_mimo_combined
    from mamba_ssm import Mamba3

import ffmpeg
from funasr.frontends.wav_frontend import WavFrontend
from funasr.utils.load_utils import extract_fbank

is_aiter_available = False

logger = logging.get_logger(__name__)



class Mamba3VITAAudioConfig(PreTrainedConfig):
    r"""
    Mamba3VITAAudioConfig
    """
    model_type = "mamba3_vita_audio"
    base_config_key = "audio_config"

    def __init__(
        self,
        hidden_size=512,
        temporal_merge_size=1,
        out_hidden_size=4608,
        merger_hidden_size=4608,
        downsample_hidden_size=512,
        num_mel_bins=128,
        n_window=50,
        n_window_infer=800,
        conv_chunksize=500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.temporal_merge_size = temporal_merge_size
        self.out_hidden_size = out_hidden_size
        self.merger_hidden_size = merger_hidden_size

        self.downsample_hidden_size = downsample_hidden_size
        self.num_mel_bins = num_mel_bins
        self.n_window = n_window
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.num_hidden_layers = 0


class Mamba3VITAVisionConfig(PreTrainedConfig):
    r"""
    Mamba3VITAVisionConfig
    """
    model_type = "mamba3_vita_vision"
    base_config_key = "vision_config"
        
    def __init__(
        self,
        hidden_size=1024,
        d_state=128,
        expand=2,
        headdim=64,
        is_mimo=False,
        mimo_rank=4,
        chunk_size=64,
        is_outproj_norm=False,
        intermediate_size=4096,
        num_hidden_layers=12,
        num_channels=3,
        num_patches=256,
        patch_size=16,
        hidden_act="silu",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        spatial_merge_size=2,
        out_hidden_size=4608,
        merger_hidden_size=4608,
        layer_type_list=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.d_model = hidden_size
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank
        self.chunk_size = chunk_size
        self.is_outproj_norm = is_outproj_norm

        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.spatial_merge_size = spatial_merge_size
        self.out_hidden_size = out_hidden_size
        self.merger_hidden_size = merger_hidden_size

        
        self.layer_type_list = layer_type_list


class Mamba3VITATextConfig(PreTrainedConfig):
    r"""
    Mamba3VITATextConfig
    """
    model_type = "mamba3_vita_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    def __init__(
        self,
        vocab_size: int | None = 151936,
        hidden_size=1024,
        d_state=128,
        expand=2,
        headdim=64,
        is_mimo=False,
        mimo_rank=4,
        chunk_size=64,
        is_outproj_norm=False,
        intermediate_size=4096,
        num_hidden_layers=24,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 32768,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-6,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        
        attention_dropout: float | None = 0.0,
        
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,

        layer_type_list=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings

        self.hidden_size = hidden_size
        self.d_model = hidden_size
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank
        self.chunk_size = chunk_size
        self.is_outproj_norm = is_outproj_norm

        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout

        self.layer_type_list = layer_type_list

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

class Mamba3VITAConfig(PreTrainedConfig):
    r"""
    Mamba3VITAConfig
    """

    model_type = "mamba3_vita"
    sub_configs = {"audio_config": Mamba3VITAAudioConfig, "vision_config": Mamba3VITAVisionConfig, "text_config": Mamba3VITATextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        vision_config=None,
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


class Mamba3VITACNNAudioEmbeddings(nn.Module):
    def __init__(self, config: Mamba3VITAAudioConfig):
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


class Mamba3VITACNNAudioEncoderLayer(nn.Module):

    def __init__(self, config: Mamba3VITAAudioConfig):
        super().__init__()

    def forward(
            self,
            hidden_states: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        return hidden_states


class Mamba3VITACNNAudioEncoder(nn.Module):

    def __init__(self, config: Mamba3VITAAudioConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            Mamba3VITACNNAudioEncoderLayer(config) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = True

    def forward(self, x):
        for idx, encoder_layer in enumerate(self.layers):
            x = encoder_layer(x)
        return x


class Mamba3VITACNNAudio(nn.Module):

    def __init__(self, config: Mamba3VITAAudioConfig):
        super().__init__()
        self.config = config

        self.embeddings = Mamba3VITACNNAudioEmbeddings(config)
        self.encoder = Mamba3VITACNNAudioEncoder(config)

    def forward(self, audios):

        audio_lengths = torch.as_tensor([len(x) for x in audios])
        # audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=0.0)
        audios = torch.cat(audios, dim=0).transpose(1, 0)

        features, feature_lengths = self.embeddings(audios, audio_lengths)

        features = self.encoder(features)

        features = features.split(feature_lengths.tolist(), dim=0)
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)

        return features, feature_lengths


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


class Mamba3VITAAudioPatchMerger(nn.Module):
    def __init__(self, config: Mamba3VITAAudioConfig) -> None:
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


class Mamba3VITAVisionPatchMerger(nn.Module):
    def __init__(self, config: Mamba3VITAVisionConfig) -> None:
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


class Mamba3VITATextRMSNorm(Qwen3RMSNorm):
    pass


class Mamba3VITATextRotaryEmbedding(Qwen3RotaryEmbedding):
    pass


class Mamba3VITATextMLP(Qwen3MLP):
    pass


class Mamba3VITATextMixer(nn.Module):
    
    def __init__(self, config: Mamba3VITATextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.is_causal = False

        self.d_model = config.d_model
        self.d_state = config.d_state
        self.expand = config.expand
        self.headdim = config.headdim

        self.is_mimo = config.is_mimo
        self.mimo_rank = config.mimo_rank
        self.chunk_size = config.chunk_size
        self.is_outproj_norm = config.is_outproj_norm

        self.d_inner = int(self.expand * self.d_model)

        self.layer = Mamba3(
            # This module uses roughly 6 * d_model^2 parameters
            d_model=self.d_model, # Model dimension d_model
            d_state=self.d_state,  # SSM state size
            headdim=self.headdim, # SSM headdim
            is_mimo=self.is_mimo, # Use MIMO mode
            mimo_rank=self.mimo_rank, # MIMO rank when is_mimo=True
            chunk_size=self.chunk_size, # 64/mimo_rank if x is in bf16, else 32/mimo_rank.
            is_outproj_norm=self.is_outproj_norm, # Additional post SSM norm
            dtype=torch.bfloat16,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Input shape: Batch x Time x Channel"""

        input_ndim = hidden_states.dim()
        if input_ndim == 3:
            pass
        elif input_ndim == 2:
            # [seq_len, hidden_size] -> [1, seq_len, hidden_size]
            hidden_states = hidden_states.unsqueeze(0)

        y = self.layer(hidden_states, cu_seqlens=kwargs.get("cu_seqlens", None))

        if input_ndim == 3:
            pass
        elif input_ndim == 2:
            # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
            y = y.squeeze(0)
        
        attn_weights = None

        return y, attn_weights


class Mamba3VITATextDecoderLayer(GradientCheckpointingLayer):

    def __init__(self, config: Mamba3VITATextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.layer_idx = layer_idx
        self.layer_type = config.layer_type_list[layer_idx]

        if self.layer_type == "M":
            self.input_layernorm =  Mamba3VITATextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mixer = Mamba3VITATextMixer(config=config, layer_idx=layer_idx)
        elif self.layer_type == "-":
            self.post_attention_layernorm = Mamba3VITATextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp = Mamba3VITATextMLP(config)
        else:
            raise ValueError(f"Invalid layer type: {self.layer_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
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
        print(f"{self.layer_idx=} {self.layer_type=} {hidden_states.shape=} {hidden_states.max()=} {hidden_states.min()=} {hidden_states.mean()=}")
        if self.layer_type == "M":
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _ = self.mixer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + hidden_states
        
        elif self.layer_type == "-":
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        else:
            raise ValueError(f"Invalid layer type: {self.layer_type}")
        print(f"{self.layer_idx=} {self.layer_type=} {hidden_states.shape=} {hidden_states.max()=} {hidden_states.min()=} {hidden_states.mean()=}")

        return hidden_states


class Mamba3VITAAudioPreTrainedModel(PreTrainedModel):
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    pass


class Mamba3VITAVisionPreTrainedModel(PreTrainedModel):
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    pass


class Mamba3VITAPreTrainedModel(PreTrainedModel):
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


class Mamba3VITAAudioModel(Mamba3VITAAudioPreTrainedModel):
    config: Mamba3VITAAudioConfig

    def __init__(
        self,
        config: Mamba3VITAAudioConfig,
        *inputs,
        **kwargs,
    ):
        super().__init__(config, *inputs, **kwargs)

        self.model = Mamba3VITACNNAudio(config)
        self.merger = Mamba3VITAAudioPatchMerger(config)
    
    def forward(
        self,
        audios,
    ):
        encoder_out, encoder_out_lens = self.model(audios)
        encoder_out = self.merger(encoder_out)

        encoder_out_lens = -(-encoder_out_lens // self.config.temporal_merge_size)

        return encoder_out, encoder_out_lens


class Mamba3VITAVisionRotaryEmbedding(nn.Module):
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


class Mamba3VITAVisionEmbeddings(nn.Module):
    def __init__(self, config: Mamba3VITAVisionConfig):
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


class Mamba3VITAVisionMixer(nn.Module):

    def __init__(self, config: Mamba3VITAVisionConfig):
        super().__init__()
        
        self.config = config
        self.is_causal = False

        self.d_model = config.d_model
        self.d_state = config.d_state
        self.expand = config.expand
        self.headdim = config.headdim

        self.is_mimo = config.is_mimo
        self.mimo_rank = config.mimo_rank
        self.chunk_size = config.chunk_size
        self.is_outproj_norm = config.is_outproj_norm

        self.d_inner = int(self.expand * self.d_model)

        self.layer = Mamba3(
            # This module uses roughly 6 * d_model^2 parameters
            d_model=self.d_model, # Model dimension d_model
            d_state=self.d_state,  # SSM state size
            headdim=self.headdim, # SSM headdim
            is_mimo=self.is_mimo, # Use MIMO mode
            mimo_rank=self.mimo_rank, # MIMO rank when is_mimo=True
            chunk_size=self.chunk_size, # 64/mimo_rank if x is in bf16, else 32/mimo_rank.
            is_outproj_norm=self.is_outproj_norm, # Additional post SSM norm
            dtype=torch.bfloat16,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        y = self.layer(hidden_states.unsqueeze(0), cu_seqlens=cu_seqlens).squeeze(0)
        attn_weights = None

        return y, attn_weights

class Mamba3VITAVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Mamba3VITAVisionEncoderLayer(nn.Module):
    def __init__(self, config: Mamba3VITAVisionConfig, layer_idx: int):
        super().__init__()
        self.embed_dim = config.hidden_size
        
        self.layer_idx = layer_idx
        self.layer_type = config.layer_type_list[layer_idx]
        if self.layer_type == "M":
            self.layer_norm1 = nn.RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.mixer = Mamba3VITAVisionMixer(config)
        elif self.layer_type == "-":
            self.layer_norm2 = nn.RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.mlp = Mamba3VITAVisionMLP(config)
        else:
            raise ValueError(f"Invalid layer type: {self.layer_type}")

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
        if self.layer_type == "M":
            residual = hidden_states
            hidden_states = self.layer_norm1(hidden_states)
            hidden_states, attn_weights = self.mixer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            hidden_states = residual + hidden_states

        elif self.layer_type == "-":
            residual = hidden_states
            hidden_states = self.layer_norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        else:
            raise ValueError(f"Invalid layer type: {self.layer_type}")

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class Mamba3VITAVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Mamba3VITAVisionEncoderLayer`].

    Args:
        config: Mamba3VITAVisionConfig
    """

    def __init__(self, config: Mamba3VITAVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Mamba3VITAVisionEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        self.spatial_merge_size = 2
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_size = config.patch_size
        # self.window_size = self.patch_size * 2 * 8

        # self.rotary_pos_emb = Mamba3VITAVisionRotaryEmbedding(config.hidden_size // config.num_attention_heads // 2)

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

        # rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        # position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

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
                    # position_embeddings,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                    cu_seqlens=cu_seqlens, 
                    # position_embeddings=position_embeddings
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


class Mamba3VITAVisionModel(Mamba3VITAVisionPreTrainedModel):
    _input_embed_layer = "patch_embedding"
    config: Mamba3VITAVisionConfig

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Mamba3VITAVisionEmbeddings(config)
        self.encoder = Mamba3VITAVisionEncoder(config)

        self.merger = Mamba3VITAVisionPatchMerger(config)

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


class Mamba3VITATextModel(Mamba3VITAPreTrainedModel):
    config: Mamba3VITATextConfig

    def __init__(self, config: Mamba3VITATextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Mamba3VITATextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Mamba3VITATextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.rotary_emb = Mamba3VITATextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # # It may already have been prepared by e.g. `generate`
        # if not isinstance(causal_mask_mapping := attention_mask, dict):
        #     # Prepare mask arguments
        #     mask_kwargs = {
        #         "config": self.config,
        #         "input_embeds": inputs_embeds,
        #         "attention_mask": attention_mask,
        #         "cache_position": cache_position,
        #         "past_key_values": past_key_values,
        #         "position_ids": position_ids,
        #     }
        #     # Create the masks
        #     causal_mask_mapping = {
        #         "full_attention": create_causal_mask(**mask_kwargs),
        #     }
        #     # The sliding window alternating layers are not always activated depending on the config
        #     if self.has_sliding_layers:
        #         causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        # position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                # attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                # position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Mamba3VITAModel(Mamba3VITAPreTrainedModel):
    def __init__(self, config: Mamba3VITAConfig):
        super().__init__(config)
        self.audio_model = Mamba3VITAAudioModel._from_config(config.audio_config)
        self.vision_model = Mamba3VITAVisionModel._from_config(config.vision_config)
        self.language_model = Mamba3VITATextModel._from_config(config.text_config)

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



class Mamba3VITAForCausalLM(Mamba3VITAPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: Mamba3VITAConfig):
        super().__init__(config)
        self.model = Mamba3VITAModel(config)
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



class Mamba3_VITA_TOKEN(DEFAULT_TOKEN):

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


_GLOBAL_TOKEN = Mamba3_VITA_TOKEN()


def get_token():
    _ensure_var_is_initialized(_GLOBAL_TOKEN, "token")
    return _GLOBAL_TOKEN


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
        else:
            audio, sampling_rate = torchaudio.load(audio_or_path)
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


class Mamba3VITAImagesKwargs(ImagesKwargs, total=False):
    """
    """
    discrete_image_idxs: list
    contiguous_image_idxs: list

    vision_resolution_type: str
    vision_normalize_type: str
    image_min_num_tokens: int
    image_max_num_tokens: int


class Mamba3VITAAudioKwargs(AudioKwargs, total=False):
    """
    """
    discrete_audio_idxs: list
    contiguous_audio_idxs: list

    # temporal_merge_size: int

    # audio_tokenizer_type: str
    # audio_tokenizer_path: str

class Mamba3VITAVideosKwargs(VideosKwargs, total=False):
    """
    """
    vision_resolution_type: str
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


class Mamba3VITAProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Mamba3VITAImagesKwargs
    videos_kwargs: Mamba3VITAVideosKwargs
    audio_kwargs: Mamba3VITAAudioKwargs

    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
        "images_kwargs": {
            "vision_resolution_type": "native",
            "vision_normalize_type": "siglip",
            "image_min_num_tokens": 4,
            "image_max_num_tokens": 8192,
        },
        "videos_kwargs": {
            "vision_resolution_type": "native",
            "video_min_num_tokens": 64,
            "video_max_num_tokens": 8192,
            "video_image_min_num_tokens": 4,
            "video_image_max_num_tokens": 256,
            "video_max_num_frames": 64,
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
            # "temporal_merge_size": 1,
        },
    }


class Mamba3VITAFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["pixel_values", "image_grid_thw"]
    valid_kwargs = Mamba3VITAAudioKwargs

    def __init__(
        self,
        audio_tokenizer_path=None,
        audio_tokenizer_type=None,
        flow_path=None,
        rank=None,
        text_audio_interval_ratio=None,
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
        audio_chunk_min_second=30,
        audio_chunk_max_second=30,
        **kwargs,
    ):
        GLOBAL_TOKEN = get_token()

        AUD_CONTEXT_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.AUD_CONTEXT_TOKEN)
        AUD_TAG_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.AUD_TAG_TOKEN)
        AUD_START_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.AUD_START_TOKEN)
        AUD_END_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.AUD_END_TOKEN)

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
                    x + [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * (aud_pos - st)
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
                                new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID] * len(
                                    _input_id
                                )

                    new_input_ids += [AUD_START_ID]
                    if targets is not None:
                        if is_pretrain:
                            new_targets += [AUD_START_ID]
                        else:
                            new_targets += [GLOBAL_TOKEN.IGNORE_TOKEN_ID]
                    if additional_targets_list is not None:
                        additional_targets_list = [
                            x + [GLOBAL_TOKEN.IGNORE_TOKEN_ID]
                            for x in additional_targets_list
                        ]

                    audio_token_length = -(
                        -audio_token_length_func(len(audio)) // self.temporal_merge_size
                    )
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
                            GLOBAL_TOKEN.IGNORE_TOKEN_ID
                        ] * audio_token_length
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
                            x + [GLOBAL_TOKEN.IGNORE_TOKEN_ID]
                            for x in additional_targets_list
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
    except Exception as e:
        logger.error(f"Error probing video: {e}")
        return False


class Mamba3VITAVideoProcessor(BaseVideoProcessor):
    model_input_names = ["pixel_values", "image_grid_thw"]
    valid_kwargs = Mamba3VITAVideosKwargs


    def __init__(
        self,
        image_processor=None,
        audio_processor=None,
        vision_resolution_type=None,
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

        self.vision_resolution_type = vision_resolution_type
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
        # if has_audio(video_file_or_dir):
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

        if self.vision_resolution_type == "native":
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
                vision_resolution_type=self.vision_resolution_type,
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
        video_audio_chunk_min_second = kwargs.get("video_audio_chuk_min_second", self.video_audio_chunk_min_second)
        video_audio_chunk_max_second = kwargs.get("video_audio_chuk_max_second", self.video_audio_chunk_max_second)

        GLOBAL_TOKEN = get_token()

        IMG_CONTEXT_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.IMG_CONTEXT_TOKEN)
        IMG_START_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.IMG_START_TOKEN)
        IMG_END_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.IMG_END_TOKEN)

        AUD_CONTEXT_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.AUD_CONTEXT_TOKEN)
        AUD_START_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.AUD_START_TOKEN)
        AUD_END_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.AUD_END_TOKEN)

        VID_CONTEXT_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.VID_CONTEXT_TOKEN)
        VID_START_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.VID_START_TOKEN)
        VID_END_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.VID_END_TOKEN)

        IMG_TAG_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.IMG_TAG_TOKEN)
        AUD_TAG_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.AUD_TAG_TOKEN)
        VID_TAG_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.VID_TAG_TOKEN)

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
                if self.image_processor.vision_resolution_type == "native":
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

                        if self.image_processor.vision_resolution_type == "native":
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

                        audio_token_length = -(-audio_token_length_func(len(audio_chunk_frame)) // self.temporal_merge_size)
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




class Mamba3VITAImageProcessor(BaseImageProcessor):
    model_input_names = ["images", "image_indices", "image_grid_thw"]
    valid_kwargs = Mamba3VITAImagesKwargs

    def __init__(
        self,
        image_size=448,
        image_size_discrete=None,
        vision_normalize_type="imagenet",
        vision_resolution_type="dynamic",
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
        self.vision_resolution_type = vision_resolution_type
        self.min_tile_grid = min_tile_grid
        self.max_tile_grid = max_tile_grid
        self.tile_image_size = image_size
        self.image_max_num_tokens = image_max_num_tokens
        self.image_min_num_tokens = image_min_num_tokens

        GLOBAL_TOKEN = get_token()
        if vision_normalize_type == "imagenet":
            MEAN, STD = GLOBAL_TOKEN.IMAGENET_DEFAULT_MEAN, GLOBAL_TOKEN.IMAGENET_DEFAULT_STD
        elif vision_normalize_type == "clip":
            MEAN, STD = GLOBAL_TOKEN.OPENAI_CLIP_MEAN, GLOBAL_TOKEN.OPENAI_CLIP_STD
        elif vision_normalize_type == "siglip":
            MEAN, STD = GLOBAL_TOKEN.IMAGENET_STANDARD_MEAN, GLOBAL_TOKEN.IMAGENET_STANDARD_STD
        else:
            raise NotImplementedError(vision_normalize_type)
        self.mean = MEAN
        self.std = STD

        if self.vision_resolution_type == "anyres":
            raise NotImplemented
            self.grid_pinpoints = [
                (i, j)
                for i in range(min_tile_grid, max_tile_grid + 1)
                for j in range(min_tile_grid, max_tile_grid + 1)
            ]
            self.possible_resolutions = [
                [dim * self.tile_image_size for dim in pair] for pair in self.grid_pinpoints
            ]
            logger.info(f"{self.grid_pinpoints=}")
            logger.info(f"{self.possible_resolutions=}")

        if self.vision_resolution_type == "dynamic":
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
            logger.info(f"{self.target_ratios=}")
            logger.info(f"{self.possible_resolutions=}")

        if self.vision_resolution_type == "native":
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

    def process_image_to_tiles(self, image_or_path, **kwargs):
        if self.vision_resolution_type == "anyres":
            return self.process_anyres(image_or_path)
        if self.vision_resolution_type == "dynamic":
            return self.process_dynamic(image_or_path)
        if self.vision_resolution_type == "native":
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
            vision_resolution_type = kwargs.get("vision_resolution_type", self.vision_resolution_type)
            if self.vision_resolution_type == "anyres":
                return self.process_anyres(image_or_path)
            if self.vision_resolution_type == "dynamic":
                return self.process_dynamic(image_or_path)
            if self.vision_resolution_type == "native":
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

        GLOBAL_TOKEN = get_token()

        IMG_CONTEXT_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.IMG_CONTEXT_TOKEN)
        IMG_START_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.IMG_START_TOKEN)
        IMG_END_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.IMG_END_TOKEN)
        IMG_TAG_ID = tokenizer.convert_tokens_to_ids(GLOBAL_TOKEN.IMG_TAG_TOKEN)

        if self.vision_resolution_type == "native":
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
                assert self.vision_resolution_type == "native"
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

                if self.vision_resolution_type == "native":
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

                if self.vision_resolution_type == "native":
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



class Mamba3VITAProcessor(ProcessorMixin):
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
        **kwargs: Unpack[Mamba3VITAProcessorKwargs],
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
            Mamba3VITAProcessorKwargs,
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
            (
                input_ids,
                _images,
                image_indices,
                _audios,
                audio_indices,
                image_grid_thw,
                second_per_grids,
                # ) = self.video_processor.add_video_input_contiguous(
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
    "Mamba3VITAConfig",
    "Mamba3VITAPreTrainedModel",
    "Mamba3VITAModel",
    "Mamba3VITAForCausalLM",
    "Mamba3VITAProcessor",
    "Mamba3VITAImageProcessor",
    "Mamba3VITAVideoProcessor",
    "Mamba3VITAFeatureExtractor",
]

