# VITA 系列模型（`qwen3_vita` / `youtu_vita`）`omni_encoder + llm_decoder` 范式模型结构与推理流程说明

> 涉及代码：
> - `src/transformers/models/qwen3_vita/`（核心：`modular_qwen3_vita.py`，自动生成 `modeling_qwen3_vita.py`）
> - `src/transformers/models/youtu_vita/`（核心：`modular_youtu_vita.py`，自动生成 `modeling_youtu_vita.py`）
>
> 文档定位：**专门介绍上述两个模型采用的 `omni_encoder + llm_decoder` 范式**，包括其模型结构、各模态预处理与 forward 推理流程。
> - 仅描述 **omni 路径**（即 `config.omni_config is not None` 的主线 checkpoint）；早期独立 vision/audio encoder 后备分支（`Qwen3VITAVisionModel` / `Qwen3VITAAudioModel`，由 `vision_config` / `audio_config` 触发）不在本文范围。
> - 不涉及训练、对齐验收、引擎适配细节。
>
> **代码对齐说明**：本次说明对齐 `modular_*.py` 的当前结构。`qwen3_vita` 与 `youtu_vita` 两份 `modular` 文件在 omni 范式部分**结构完全一致（逐行对应）**，`youtu_vita` 是带前缀重命名的**独立完整拷贝**（并非 `pass` 继承 `qwen3_vita`），差异仅集中在内嵌 tokenizer 工具与特殊 token 集合（见 §5）。下文以 `qwen3_vita` 类名为例，`youtu_vita` 把 `Qwen3VITA*` 前缀替换为 `YoutuVITA*` 即可一一对应。

---

## 符号约定

| 符号 | 含义 |
|---|---|
| `B`, `S`, `H` | LM 端 batch size、序列长度、隐维 |
| `H_enc` | OmniEncoder 隐维（合并 + 投影前） |
| `N_img_tokens` | merger 后视觉 token 数（≤ patch 总数） |
| `S_a` | merger 后单段音频 token 数 |
| `N_video` | 一个 batch 中的视频条数（范式 B） |

---

## 1. 模型整体架构

### 1.1 数据流图

```
┌─ 模态前端（OmniEncoder 内） ──────────────────────────────────────────────┐
│                                                                          │
│ [pixel_values, image_grid_thw]   ─► vision_embeddings (linear patch)     │
│ [audios (list of mel)]           ─► audio_embeddings   (3×Conv2d s=2)    │
│ [video_images, video_image_grid_thw, video_audios, video_split]          │
│                                  ─► (范式 B 才用，见 §2.4)               │
│                                                                          │
│         共享 Qwen3 Transformer Body（Qwen3VITAOmniEncoder）             │
│         双向 + packed varlen flash-attn；RoPE 默认 2D(视觉)/1D(音频)，    │
│         可选 4D (M|T|H|W) 见 §1.3.4                                       │
│                       │                              │                   │
│  vision_merger (2×2 空间合并 + 2 层 MLP) ──► image_embeds  [N_img, H]   │
│  audio_merger  (2× 时间合并   + 2 层 MLP) ──► audio_embeds [B, S_a, H]  │
└──────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
                          ┌─── scatter 替换占位 token 的 embedding ───┐
                          │  inputs_embeds[batch_idx, seq_idx] = X    │
                          └────────────────────┬──────────────────────┘
                                               │
   input_ids ──► language_model.embed_tokens ──┘
                                               │
                                               ▼
                          ┌────────────────────────────────────────────┐
                          │  LanguageModel（Qwen3 Decoder）            │
                          │  qk‑norm + GQA + 标准 RoPE + (可选 SWA)    │
                          │  自回归 / causal mask / KV cache           │
                          └────────────────────┬───────────────────────┘
                                               ▼
                                       lm_head（tied）→ logits
```

### 1.2 主要类与角色

`qwen3_vita`（以 modular 文件为权威源，行号对齐 `modular_qwen3_vita.py`）：

| 类 | 角色 |
|---|---|
| `Qwen3VITAConfig`（`:283`） | 顶层 config：`text_config` + `omni_config`（本范式下二者均必填）；顶层还持有 `tie_word_embeddings`、`video_omni_fusion`（视频范式开关，见 §2.4）。 |
| `Qwen3VITAOmniConfig`（`:210`，继承 `Qwen3VITATextConfig`） | OmniEncoder 的 Transformer 配置（继承 Qwen3 结构），同时含视觉/音频前端超参（patch、mel、merger）与视频融合 / 4D RoPE 相关字段。 |
| `Qwen3VITATextConfig`（`:201`，继承 `Qwen3Config`） | LanguageModel 的 Qwen3 配置。 |
| `Qwen3VITAOmniVisionEmbeddings`（`:2143`，继承 `Qwen3VITAVisionEmbeddings`） | 视觉前端：`nn.Linear(in=C·patch²·temporal_patch, out=hidden_size)`，把已 patchify 的像素投到 Encoder 隐维。 |
| `Qwen3VITAOmniAudioEmbeddings`（`:2156`，继承 `Qwen3VITACNNAudioEmbeddings`） | 音频前端：3 层 stride‑2 Conv2d（**对 mel 时间轴 8× 降采样**）+ 末尾 `linear_proj` 投影到 `hidden_size`。 |
| `Qwen3VITAOmniEncoder`（`:2280`） | 共享 Transformer body：双向、无 causal mask、packed varlen flash-attn；视觉 2D RoPE、音频 1D RoPE，可选 4D MTHW RoPE；`forward` 支持 **per-layer 的 `cu_seqlens` 与 rotary**（供视频分层融合用）。 |
| `Qwen3VITAOmniVisionPatchMerger`（`:2589`）/ `Qwen3VITAOmniAudioPatchMerger`（`:2601`） | merger + projector：视觉 `spatial_merge_size=2`（2×2）、音频 `temporal_merge_size=2`；其后接 2 层 MLP 投到 LM 隐维 `out_hidden_size`。 |
| `Qwen3VITAOmniRotaryEmbedding` / `Qwen3VITAOmniChunkedMTHWRotaryEmbedding`（`:1905`）/ `Qwen3VITAOmniInterleavedMTHWRotaryEmbedding`（`:2040`） | OmniEncoder 的 RoPE：默认旋转 + 两种可选 4D (M\|T\|H\|W) 变体（见 §1.3.4）。 |
| `Qwen3VITAOmniModel`（`:2657`） | 上述前端 + body + merger 的封装；`forward(modality=...)` 分派 `forward_vision` / `forward_audio` / `forward_video`。 |
| `Qwen3VITATextModel`（`:1794`） | 自回归 LM 主体（Qwen3：qk‑norm、GQA、SwiGLU、RMSNorm、标准 RoPE、可选 sliding window）。 |
| `Qwen3VITAModel`（`:3592`） | 顶层组合：`omni_model` + `language_model`（按 config 还可含后备 `vision_model` / `audio_model`，本范式不走）。 |
| `Qwen3VITAForCausalLM`（`:3916`） | `GenerationMixin` 入口，含 `lm_head`（与 `embed_tokens` tied）。 |

`youtu_vita`：在 omni 范式部分与 `qwen3_vita` **结构逐行一致**（独立拷贝、前缀替换为 `YoutuVITA*`，`YoutuVITAOmniModel` 继承自 `YoutuVITAOmniPreTrainedModel` 而非 `qwen3_vita` 的类）；差异集中在 **内嵌 tokenizer 工具** 与 **特殊 token 集合 / video processor** 三处（见 §5）。

### 1.3 架构特点

1. **OmniEncoder 是共享的多模态前融合 Encoder**：vision/audio 前端各自投到同一隐维后，**复用同一份 Transformer 权重**做双向编码，仅在最后由各自 patch merger + 2 层 MLP 分别投到 LM 隐维。
2. **Encoder 完全双向 + packed varlen flash-attn**：
   - 视觉用 2D RoPE，按 `image_grid_thw` 拼成 `cu_seqlens`；
   - 音频用 1D RoPE，按音频特征长度拼成 `cu_seqlens`；
   - attention 走 `flash_attn_varlen_func`（`Qwen3VITAOmniFlashAttention2`，`is_causal=False`，复用 Qwen3 的 qk‑norm + GQA 投影）；
   - **不会进入 LM 的 KV cache**，每条 prompt 在 prefill 阶段一次性算完。
3. **LanguageModel 是标准 Qwen3 Decoder**：
   - causal、自回归、有 KV cache；
   - **LM 端用标准 RoPE，不是 M‑RoPE**（与 Qwen2‑VL/Qwen2.5‑Omni 不同）；§1.3.4 的 4D MTHW RoPE 只作用于 OmniEncoder 内部，不影响 LM；
   - 可选 sliding window attention（由 `text_config.sliding_window` 决定）。
4. **多模态嵌入注入方式：显式索引 scatter（非 placeholder id 比对）**：
   - processor 输出 `image_indices` / `audio_indices` / `video_image_indices` / `video_audio_indices`；
   - 模型在 `forward` 中直接按这些 `(batch_idx, seq_idx)` 对 `inputs_embeds` 做赋值替换；
   - 这一点是该范式区别于多数 VLM「找 `image_token_id` 出现位置再 scatter」做法的关键差异。
5. **音频输出走主词表的离散 token**：
   - `<\|audio_0\|>` ~ `<\|audio_16383\|>` 共 16384 个 token 直接挤进 LM 词表；
   - 不存在独立 codec / TTS head；
   - 主 LM 自回归产出这些 token 即视为音频输出，waveform 合成在外部完成（不在本文范围）。
6. **权重共享**：`lm_head.weight` 与 `model.embed_tokens.weight` tied（`_tied_weights_keys`）。
7. **TP/PP 提示**：`_tp_plan = {"lm_head": "colwise_gather_output"}`、`_pp_plan = {"lm_head": (["hidden_states"], ["logits"])}`。

#### 1.3.4 可选：OmniEncoder 4D MTHW RoPE 与视频分层融合（默认关闭）

以下三项均为 **OmniEncoder 内部** 的可选增强，由 `omni_config` 控制，**默认全关时退化为上文的 2D/1D RoPE + 整段双向注意力**，不影响 LM：

1. **4D (M\|T\|H\|W) RoPE**：每个 token 用 `(modality, t, h, w)` 四维坐标，取代默认 2D/1D 路径。两种互斥变体：
   - `video_omni_chunked_mthw_rope`（`Qwen3VITAOmniChunkedMTHWRotaryEmbedding`）：head_dim 切成 M/T/H/W 四段分别旋转；
   - `video_omni_interleaved_mthw_rope`（`Qwen3VITAOmniInterleavedMTHWRotaryEmbedding`）：M 段高频独占 + T/H/W 以 stride=3 交错（Qwen3‑VL 风格 `apply_interleaved_mrope`）；两者同开时 **interleaved 优先**。
   - M 轴模态 id：`M_IMAGE=0 / M_AUDIO=1 / M_VIDEO_FRAME=2 / M_VIDEO_AUDIO=3`；`rope_m_dim=0` 时退化为纯 3D (T\|H\|W)。
   - 视频融合模式下视觉帧与音频在 **同一 t 轴** 推进（每帧推进 `_VIDEO_FRAME_T_STEP = 100/8 = 12.5`，对齐音频 token 速率），使同一时刻的视/听 token 共享 t 坐标。
2. **`video_fusion_layer_freq`**：逐层 fusion 掩码（`None`→全 fusion；`int N`→`i%N==0` 为 fusion；`list[int]` 显式 0/1）。fusion 层让同一视频的视/音 token 共享 attention 窗口；非 fusion 层则每个 image/audio chunk 各自独立成段。由此 `Qwen3VITAOmniEncoder.forward` 支持 **每层传入不同的 `cu_seqlens` 与 rotary**。
3. **`video_group_attention`**：fusion 层内进一步把单个视频切成 `(I + A*)` 组，使注意力限制在同组的图/音之间。

---

## 2. 各模态预处理

> 预处理由 `Qwen3VITAProcessor` / `YoutuVITAProcessor`（`processing_*.py`） + `*ImageProcessor` / `*VideoProcessor` / `*FeatureExtractor` 组合完成，并由 tokenizer 在文本侧插入相应数量的占位 token。

### 2.1 文本与特殊 token

LM 词表包括：

- 普通文本 token（Qwen3 tokenizer）；
- 多模态占位 / 边界 token（字面量见下表，引用自 `youtu_vita/modular_youtu_vita.py:Youtu_VITA_TOKEN`，`qwen3_vita` 同套）；
- 16384 个离散音频输出 token `<\|audio_*\|>`（仅 S2S 场景下出现在输出端）；
- 控制 token：`Youtu_VITA_TOKEN_bus1`（`modular_youtu_vita.py:4112`，bus1 精简版，仅含 image）与 `Youtu_VITA_TOKEN`（`:4219`，含 image/audio/video/think/code/tool_call/action 等完整集，默认启用见 `:4357`），按 config 选择。

| 类别 | tag | start | end | context（占位） |
|---|---|---|---|---|
| 图像 | `<\|image\|>` | `<\|vision_start\|>` | `<\|vision_end\|>` | `<\|image_pad\|>` |
| 音频 | `<\|audio\|>` | `<\|audio_start\|>` | `<\|audio_end\|>` | `<\|audio_pad\|>` |
| 视频 | `<\|video\|>` | `<\|video_start\|>` | `<\|video_end\|>` | `<\|video_pad\|>` |

**最小 prompt 示例**（伪 chat template）：

```
<|begin_of_text|>user
<|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>请描述这张图。
assistant
```

- `<|image_pad|>` 的重复次数等于该图 **merger 后** 的视觉 token 数（详见 §2.2 公式）；
- 音频/视频段以 `<|*_start|>...<|*_pad|>×N...<|*_end|>` 同构包裹；
- 文本流中所有 `*_pad` 的位置由 processor 同步写入对应的 `*_indices`，供模型 scatter 使用。

输出：`input_ids: LongTensor[B, S]` + `attention_mask: LongTensor[B, S]`。

### 2.2 图像

1. 输入图像按 `patch_size`（默认 16，见 `Qwen3VITAOmniConfig.patch_size`）做 patchify，得到 `pixel_values: FloatTensor[N_patches_all, C·patch²·temporal_patch]`，`N_patches_all` 为 batch 内所有图像 patch 数之和。
2. 同时输出 `image_grid_thw: LongTensor[N_images, 3]`，每张图的 `(T, H_patches, W_patches)`（H/W 已是 patch 数；单图 `T=1`）。
3. tokenizer 按 **merger 后** 的视觉 token 数插入 `<|image_pad|>` 占位：
   ```
   image_tokens = T * (H_patches / spatial_merge_size) * (W_patches / spatial_merge_size)
                = T * (H_patches / 2) * (W_patches / 2)        # 默认 spatial_merge_size=2
   ```
4. 输出 `image_indices: LongTensor[2, N_img_tokens]`：第 0 行是 `batch_idx`、第 1 行是 `seq_idx`，对齐到 `inputs_embeds` 中所有 `<|image_pad|>` 的位置。

### 2.3 音频

1. 原始波形经 `WavFrontendTokenizer` / `MelFilterBankTokenizer`（`youtu_vita/modular_youtu_vita.py:4596/4689`）得到 mel 特征 `[T_i, num_mel_bins]`（`num_mel_bins=128`），按样本组成 list。
2. 模型音频前端（`Qwen3VITAOmniAudioEmbeddings`）：
   - 把所有样本 mel 在时间轴拼接 `[num_mel_bins, sum(T_i)]`；
   - 3 层 stride‑2 Conv2d → **时间轴 8× 降采样**；
   - 线性投影到 `H_enc`；
   - 内部按 `n_window`（默认 50）做分块处理，长音频走 `conv_chunksize`（默认 500）分块卷积以控显存。
3. tokenizer 按 **音频前端 + merger 后** 的音频 token 数插入 `<|audio_pad|>`：
   ```
   mel_T_after_frontend = ceil(T_i_after_padding / 8)       # 8× 降采样（3 个 stride‑2）
   audio_tokens_i       = ceil(mel_T_after_frontend / temporal_merge_size)
                        = ceil(mel_T_after_frontend / 2)     # 默认 temporal_merge_size=2
   ```
4. 输出：
   - `audios` / `input_features`：`list[FloatTensor[T_i, num_mel_bins]]`，模型前端内部会做 `torch.cat(..., dim=0).transpose(1, 0)` 拼接；
   - `audio_indices: list[LongTensor[2, n_i]]` —— **per‑sample list**（与图像的全局二维索引不同，这里是逐样本张量）。

### 2.4 视频

视频可走 **两种处理范式**，二者互斥；**范式选择由顶层 `Qwen3VITAConfig.video_omni_fusion` 决定（默认 `False`），由模型在 `forward` 中分派，而非 processor**。重要更正：processor 只要存在视频就**始终**填充 `video_*` 缓冲（含非空 `video_split`）；模型侧据 `self.config.video_omni_fusion` 决定把这批 `video_*` 数据送入范式 A 还是范式 B。源码判定（`Qwen3VITAModel.forward`，`:3847`）：

```python
if video_split is not None and video_split.numel() > 0:   # 有视频
    if self.config.video_omni_fusion:                      # True → 范式 B（联合）
        ... = self._encode_video(...)                      # forward_video
    else:                                                  # False → 范式 A（独立）
        video_image_embeds = self._encode_vision(...)
        video_audio_embeds, ... = self._encode_audio(...)
```

> 注意：`video_split` 在两种范式下都会被填充，**不能用「`video_split is None`」区分范式**；它在范式 B 中用于 `forward_video` 切分每个视频的图/音 chunk，在范式 A 中不参与 OmniEncoder（仅作为「存在视频」的触发标记）。

#### 范式 A：拆分模态独立处理（`video_omni_fusion=False`，默认）

视频被拆为 **图像帧 + 对齐音频** 两路，分别复用单图/单音频通路：

1. **视觉路径**：与单图相同，但 `image_grid_thw` 的 `T` 维体现帧数；同一视频的多帧打包成连续 patch；占位 token 由 `video_image_indices: LongTensor[2, N]` 索引。
2. **音频路径**：与单段音频相同，由 `video_audio_indices: list[LongTensor[2, n_i]]` 索引。
3. 两路在 OmniEncoder 中 **互不感知**（视觉与音频各自独立做 packed varlen attention），scatter 后落入同一段 `inputs_embeds`；视频内的音视频时间对齐由 processor 在拼 prompt 时通过占位 token 顺序保证。
4. 适用场景：多数视频理解 / VQA / 字幕等任务，结构简单、可与图像/音频混批。

#### 范式 B：联合视频编码（`video_omni_fusion=True`，`Qwen3VITAOmniModel.forward_video`）

同一视频的视觉帧与音频块在 OmniEncoder 内 **共享一个 attention 窗口**，并在 token 维 **音视频交错（interleaved）**，不同视频之间仍通过 `cu_seqlens` 保持注意力隔离。

##### 入参（来自 processor）

- `video_images: [N_patch_rows, 3·patch_dim²]` —— batch 内全部视频帧的视觉 patch 拼接；
- `video_image_grid_thw: [N_grid_rows, 3]` —— 每帧 `(T, H, W)`，多行可属同一视频；
- `video_audios: list[FloatTensor[T_i, num_mel_bins]]` —— 每个 **音频块**（chunk）的 mel 张量；processor 端按 `video_audio_chunk_max_second` 把整段音频切成等时长的 chunk（不足部分末尾保留尾巴）；
- `video_split: LongTensor[N_video, 2]` —— 每个视频的 `(num_images, num_audio_chunks)`，由 processor 在写完每段视频后追加一行（`Qwen3VITAVideoProcessor.add_video_input_discrete_or_contiguous`，`modular_qwen3_vita.py:6290` 附近）；
- `video_image_indices: [2, total_image_tokens]` / `video_audio_indices: list[[2, 1, n_i]]` —— scatter 索引，**同时被 `forward_video` 用来按 `seq_pos` 还原真实音视频交错顺序**（缺失时回退 `divmod`）。

##### 关键中间量

`forward_video` 内部对每个视频 `v` 单独建立两组 chunk：

- 视觉 chunk 列表 `vision_feature_chunks[v]`：**按帧**切分，第 `k` 个 chunk 形状 `[tokens_per_frame[k], H_enc]`，其中 `tokens_per_frame[k] = T_k · H_p_k · W_p_k`（`merger 之前` 的视觉 token 数）；
- 音频 chunk 列表 `audio_feature_chunks[v]`：**按音频块**切分，第 `j` 个 chunk 形状 `[audio_token_lens[j], H_enc]`，`audio_token_lens[j]` 为该 chunk 在 `audio_embeddings` 之后（8× 降采）但 **merger 之前** 的 token 数；
- `vision_rotary_chunks[v]` / `audio_rotary_chunks[v]`：与上面一一对应的 RoPE 张量（视觉 2D，音频 1D），shape 与对应 feature chunk 沿第 0 维一致。

##### 交错规则（核心）

`forward_video`（`:2896`）在每个视频内把视觉帧 chunk（`I`）与音频块 chunk（`A`）按 **真实时间顺序** 交错成一个 segment。当前实现有两条路径：

**优先：按 processor 写入顺序恢复（`video_image_indices` / `video_audio_indices` 均提供时）**

processor 在拼 prompt 时已把每帧/每音频块的占位 token 按真实时间顺序写入 `input_ids`；`forward_video` 读取它们在文本中的首个 `seq_pos`（图像取每帧首个 `IMG_CONTEXT`，音频取每段 `seg[1,0,0]`），按 `seq_pos` 升序排出真实交错次序 `ordered_events`。此路径**不依赖任何固定比例假设**，任意的 `I/A` 交错都能正确还原。并强制校验音频段 `seq_pos` 单调递增，否则报错（避免后续 split 错位）。

**回退：`divmod` 启发式（缺少 indices 时）**

```python
num_image_chunks         = len(vision_feature_chunks)
num_audio_chunks_actual  = len(audio_feature_chunks)
audios_per_image, extra_audio_count = divmod(num_audio_chunks_actual, num_image_chunks)
```

- 仅有视觉 → 直接按帧顺序排；
- 仅有音频 → 直接按 chunk 顺序排；
- 二者皆有 → **每张图后挂 `audios_per_image` 个音频块；前 `extra_audio_count` 张图各多挂 1 个**（仅当「每个 chunk 恰为 1 图 + 1 音」时与真实顺序一致）。

> 下面的可视化以回退路径的 `divmod` 排布为例说明 segment 结构；走优先路径时排布顺序由 processor 的真实写入次序决定，segment 的拼接 / RoPE / mask 构造方式完全相同。

视频内排序总是 **`I A* I A* I A* …`**，永不出现"两个图相邻而无音频在中间分隔"的情况（除非 `audios_per_image == 0` 且 `extra_audio_count` 已耗尽）。

##### `interleaved_features` / `interleaved_rotary` 可视化

> 设视频 `v` 有 `num_images = 3` 帧（记为 I0/I1/I2，每帧 visual token 数依次为 `t0/t1/t2`），`num_audio_chunks = 7` 个音频块（A0…A6，每块 token 数 `a0…a6`）。
> 则 `divmod(7, 3) = (2, 1)`，`audios_per_image = 2`、`extra_audio_count = 1` —— 第 0 张图后多挂 1 个音频块。

排布顺序为：

```
chunk 序：   I0   A0   A1   A2   I1   A3   A4   I2   A5   A6
说明：      img  aud  aud  aud  img  aud  aud  img  aud  aud
                ↑─── 头 1 张图多挂 1 块 ───┘
```

对应到 `forward_video` 的三个 list：

```
interleaved_features  = [ I0_feat , A0_feat , A1_feat , A2_feat , I1_feat , A3_feat , A4_feat , I2_feat , A5_feat , A6_feat ]
                         (t0,Henc) (a0,Henc) (a1,Henc) (a2,Henc) (t1,Henc) (a3,Henc) (a4,Henc) (t2,Henc) (a5,Henc) (a6,Henc)

interleaved_rotary    = [ I0_rot  , A0_rot  , A1_rot  , A2_rot  , I1_rot  , A3_rot  , A4_rot  , I2_rot  , A5_rot  , A6_rot   ]
                         （视觉 2D RoPE，音频 1D RoPE，每段独立计算后按上面顺序原样排列）

interleaved_modality_mask
                      = [ 1×t0   , 0×a0   , 0×a1   , 0×a2   , 1×t1   , 0×a3   , 0×a4   , 1×t2   , 0×a5   , 0×a6    ]
                         (1=vision, 0=audio；每段长度等于对应 chunk 的 token 数)
```

`torch.cat(interleaved_features, dim=0)` 后得到该视频的 segment：

```
video v 的 segment_features:
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ I0 │ A0 │ A1 │ A2 │ I1 │ A3 │ A4 │ I2 │ A5 │ A6 │   shape = [t0+a0+a1+a2+t1+a3+a4+t2+a5+a6, Henc]
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
  ↑    ↑                    ↑              ↑
  modality_mask: 1   0  0  0   1   0  0   1   0  0
  rotary       : 视觉2D  音频1D  视觉2D  音频1D  视觉2D  音频1D
```

多视频时，每个视频按上述方式各自拼一个 segment，所有 segment 再 `torch.cat(..., dim=0)` 成 packed sequence；用 `cu_seqlens = [0, len(seg_0), len(seg_0)+len(seg_1), ...]` 投入 `self.encoder`，**视频之间的 attention 互不可见，视频内 visual+audio 全互可见**。

> **分层融合（`video_fusion_layer_freq` / `video_group_attention`，见 §1.3.4）**：上述「视频内全互可见」是 **fusion 层** 的行为。当配置了非全 fusion 的逐层掩码时，`forward_video` 会**同时构造两套切分**——fusion 层用上面的「整段视频」`cu_seqlens`（可再被 `video_group_attention` 切成 `(I+A*)` 组），非 fusion 层用「每个 image/audio chunk 各自独立」的 `cu_seqlens`——再以 `list` 形式逐层传入 `Qwen3VITAOmniEncoder.forward`，由各层选用对应切分（4D RoPE 开启时还会并行准备每层不同的 rotary）。默认 `video_fusion_layer_freq=None` 时所有层都是 fusion 层，退化为上述单一 `cu_seqlens`。

##### 边界情形

| 情形 | 行为 |
|---|---|
| `num_images == 0 && num_audio_chunks == 0` | `continue`，不为该视频建 segment |
| `num_images == 0` | 仅按 `A0,A1,...` 顺序排，全 mask=0 |
| `num_audio_chunks == 0` | 仅按 `I0,I1,...` 顺序排，全 mask=1 |
| `num_audio_chunks_actual % num_image_chunks != 0` | 余数由前 `extra_audio_count` 张图各分 1 个 |
| `audios_per_image == 0 && extra_audio_count > 0` | 前 `extra_audio_count` 张图各挂 1 个音频，其余图后无音频 |

##### 输出

- `video_image_embeddings: [N_v_after_merge, H]` —— 全 batch 视觉部分经 `vision_merger` 合并 + 投到 LM 隐维后的结果；
- `video_audio_embeddings: [N_audios, max_S_after_merge, H]` + `video_audio_lens_after_merge: [N_audios]` —— 全 batch 音频块经 `audio_merger` 合并后按 chunk pad 成定长；
- 后续仍按 `video_image_indices` / `video_audio_indices` 分别 scatter 回 `inputs_embeds`，**LM 端看到的 prompt 仍是「先一段视频图占位、再一段视频音占位」的常规布局**——交错只发生在 OmniEncoder 内部的 attention 窗口里。

##### 适用场景

强音视频耦合的任务（如对口型、视听问答、视频中"谁在说话"），需要在编码阶段就让视觉与音频互看。

> 两种范式 **共用同一份 OmniEncoder 权重**；区别只在「视频内是否构造交错 segment + 共享 attention 窗口」。

### 2.5 预处理输出契约（送入 `forward`）

| 字段 | dtype/shape | 说明 |
|---|---|---|
| `input_ids` | `LongTensor[B, S]` | 含 `<\|*_pad\|>` 占位 token 的文本 id。 |
| `attention_mask` | `LongTensor[B, S]` | 标准 mask。 |
| `images` | `FloatTensor[N_patches_all, C·patch²·temporal_patch]` | patchify 后的图像（`Qwen3VITAModel.forward` 的形参名为 `images`；内部 `_encode_vision` 再传给 `forward_vision(pixel_values=...)`）。 |
| `image_grid_thw` | `LongTensor[N_images, 3]` | 每张图的 `(T, H_patches, W_patches)`。 |
| `image_indices` | `LongTensor[2, N_img_tokens]` | 全局 `(batch_idx, seq_idx)`；个数 = merger 后视觉 token 数。 |
| `audios` / `input_features` | `list[FloatTensor[T_i, num_mel_bins]]` | 经 mel + WavFrontend 处理；模型前端内部 `cat+transpose` 拼接。 |
| `audio_indices` | `list[LongTensor[2, n_i]]` | per‑sample `(batch_idx, seq_idx)`。 |
| `video_images` / `video_image_grid_thw` | 同 `images` / `image_grid_thw` | 视频帧的视觉 patch（专用缓冲）。 |
| `video_audios` | `list[FloatTensor[T_i, num_mel_bins]]` | 视频音频 chunk 的 mel（专用缓冲）。 |
| `video_image_indices` | `LongTensor[2, N]` | 视频帧对应的占位索引；范式 B 下还用于恢复真实音视频交错顺序。 |
| `video_audio_indices` | `list[LongTensor[2, 1, n_i]]` | 视频音轨对应的占位索引（per‑segment）；范式 B 下与上者配合恢复交错顺序。 |
| `video_split` | `LongTensor[N_video, 2]` 或 `None` | **有视频时由 processor 始终填充**：每个视频的 `(num_images, num_audios)`。非空即触发视频分支；**走范式 A 还是 B 由顶层 `config.video_omni_fusion` 决定，与本字段是否为 None 无关**。 |
| `position_ids`、`past_key_values`、`use_cache`、`cache_position`、`inputs_embeds` | — | LM 标准字段。 |

---

## 3. Forward 流程

下面给出 **推理阶段**（生成或一次性 logits）`Qwen3VITAModel.forward` / `YoutuVITAModel.forward` 的执行序列，所有步骤同样适用于 `youtu_vita`。

### 3.0 调用分派速查

| 输入存在的字段 | 触发调用 | 输出 |
|---|---|---|
| `images` + `image_grid_thw` | `_encode_vision → omni_model(modality="vision")` → `forward_vision` | `image_embeds: [N_img_tokens, H]` |
| `audios`（非视频上下文） | `_encode_audio → omni_model(modality="audio")` → `forward_audio` | `(audio_embeds: [B, S_a, H], audio_lengths)` |
| `video_split` 非空 **且** `config.video_omni_fusion=False` | 范式 A：`_encode_vision` + `_encode_audio`（复用单图/单音通路），写回用 `video_*_indices` | 同上 |
| `video_split` 非空 **且** `config.video_omni_fusion=True` | 范式 B：`_encode_video → omni_model.forward_video(...)` | `(video_image_embeddings, video_audio_embeddings, lens)` |

> 视觉路径在 `len(image_grid_thw) > 64` 时会按 64 张一组分块编码再 `cat`（`:3713`，控显存）。训练态在缺图/缺音时还会用 `fake_images` / `fake_audios` 走一次前向并以 `*.mean()*0.0` 接回梯度（推理不涉及）。

### 3.1 Step‑by‑step

1. **文本 embedding 初始化**
   `inputs_embeds = language_model.embed_tokens(input_ids)`，形状 `[B, S, H]`。

2. **视觉编码**（当 `pixel_values is not None`）
   - 调 `self._encode_vision(pixel_values, image_grid_thw)` → `omni_model(modality="vision", ...)` → `forward_vision`：
     1. `vision_embeddings(pixel_values, image_grid_thw)` → `[N_patch_all, H_enc]`；
     2. 由 `image_grid_thw` 计算 2D RoPE 与 `cu_seqlens`；
     3. 共享 Encoder 双向 packed varlen 前向；
     4. `vision_merger`：2×2 空间合并 + 2 层 MLP → `image_embeds: [N_img_tokens, H]`。

3. **音频编码**（当 `audios is not None`）
   - 调 `self._encode_audio(audios)` → `omni_model(modality="audio", ...)` → `forward_audio`：
     1. 拼接 mel `[num_mel, sum(T_i)]`，记录 `audio_lengths`；
     2. `audio_embeddings`：3×Conv2d 8× 时间降采样 + 线性投影；
     3. 1D RoPE + `cu_seqlens` + 共享 Encoder；
     4. `audio_merger`：2× 时间合并 + 2 层 MLP → `(audio_embeds: [B, S_a, H], audio_lengths)`。

4. **视频编码**（当 `video_split` 非空）—— **两种范式互斥，由顶层 `config.video_omni_fusion` 决定，与 `video_split` 是否为 None 无关**：
   - **范式 A**（`config.video_omni_fusion=False`，默认）：视觉部分复用 §3.1.2（输入是该视频的全部帧 patch）、音频部分复用 §3.1.3（输入是该视频的全部音频块）；两路在 OmniEncoder 内 **互不感知**，分别输出 `video_image_embeds`、`video_audio_embeds`。
   - **范式 B**（`config.video_omni_fusion=True`）：调 `_encode_video → omni_model.forward_video(video_images, video_image_grid_thw, video_audios, video_split, video_image_indices, video_audio_indices)`：
     1. 分别过 `vision_embeddings` / `audio_embeddings` 拿到 visual/audio features；
     2. 按 `video_split` 把同一视频的 visual chunk 与 audio chunk 取出，**按 processor 写入 `input_ids` 的真实时间顺序（读 `video_image_indices` / `video_audio_indices` 的 `seq_pos` 排序）交错拼成 segment**（缺 indices 时回退 `divmod` 启发式），同时拼好 RoPE（默认 2D/1D，或 4D MTHW）与 `modality_mask`；
     3. 多视频 segment 通过 `cu_seqlens` 在 OmniEncoder 内 **共享 attention 窗口但相互隔离**；若开启 `video_fusion_layer_freq` 则按层切换 fusion / 非 fusion 的 `cu_seqlens`（与 rotary）；
     4. Encoder 输出按 `modality_mask` 拆回视觉/音频，分别经 `vision_merger` / `audio_merger` → `video_image_embeddings`、`video_audio_embeddings` (+ `video_audio_lens_after_merge`)。

5. **占位 scatter（核心步骤）**
   - 图像（**全局一次性**）：
     ```python
     indices_b, indices_s = image_indices.unbind(dim=0)
     inputs_embeds[indices_b, indices_s] = image_embeds
     ```
   - 音频（**逐样本**）：
     ```python
     for audio_embeds_, audio_lengths_, audio_indices_ in zip(audio_embeds, audio_lengths, audio_indices):
         audio_embeds_ = audio_embeds_[:audio_lengths_]  # 去 padding
         indices_b, indices_s = audio_indices_.unbind(dim=0)
         inputs_embeds[indices_b, indices_s] = audio_embeds_
     ```
   - 视频：同上，分别对 `video_image_indices`、`video_audio_indices` 做 scatter。

6. **LanguageModel 前向**
   ```python
   return self.language_model(
       input_ids=None,                # 已用 inputs_embeds
       attention_mask=attention_mask,
       position_ids=position_ids,
       past_key_values=past_key_values,
       inputs_embeds=inputs_embeds,
       use_cache=use_cache,
       cache_position=cache_position,
       **kwargs,
   )
   ```
   - causal、KV cache、可选 sliding window；
   - 输出 `BaseModelOutputWithPast`。

7. **LM head → logits**
   - `Qwen3VITAForCausalLM` 在 LM 输出之上接 `lm_head`（与 `embed_tokens` tied）→ `logits: [B, S, V]`。
   - generation 阶段进入 `GenerationMixin` 通用 sampling 循环；**LM 输出后没有任何自定义后处理**——即便输出包含 `<\|audio_*\|>` 也是常规 token，模型链路本身不再做声学解码。

### 3.2 关键不变量

| # | 不变量 |
|---|---|
| I1 | 占位替换 **不通过 token id 比对**，而是直接按 `*_indices` 二维下标 scatter。 |
| I2 | `image_indices` 是 **batch 内全局** 的 `[2, N]`；`audio_indices` / `video_*_indices` 是 **per‑sample list**。 |
| I3 | OmniEncoder **不进入 KV cache**，prefill 一次性算完，后续 decode 步只跑 LanguageModel。 |
| I4 | OmniEncoder 用 packed varlen attention，`cu_seqlens` 由 `image_grid_thw` / 音频特征长度 / `video_split` 决定；多图/多段音频/多视频混批的边界必须严格按它来切。 |
| I5 | merger 下采样比例（视觉 `spatial_merge_size`，默认 2×2；音频 `temporal_merge_size`，默认 2）+ 音频前端 8× 降采样共同决定占位 token 数量，processor 与模型必须严格对齐，否则 scatter 越界或错位。 |
| I6 | `lm_head.weight` 与 `model.embed_tokens.weight` tied；权重加载完成后两者须指向同一块存储。 |
| I7 | LM 是 **标准 RoPE，不是 M‑RoPE**。 |
| I8 | 音频输出 token `<\|audio_*\|>` 由主 LM 自回归产出，**不需要任何额外 head**；LM 输出后链路无自定义后处理。 |
| I9 | 视频走两种范式之一，**由顶层 `config.video_omni_fusion` 决定，而非 processor，也与 `video_split` 是否为 None 无关**：有视频时 processor 始终填非空 `video_split`；`video_omni_fusion=False`（默认）→ 范式 A（拆分模态独立）；`True` → 范式 B（联合视频编码，OmniEncoder 内音视频 chunk 按真实时间顺序交错并按视频隔离 attention，可选分层融合 / 4D RoPE）。两者共享同一份 OmniEncoder 权重，最终都用 `video_image_indices` / `video_audio_indices` scatter 回 `inputs_embeds`。 |
| I10 | OmniEncoder **双向 + 无 causal mask**；LM **严格 causal**；两者之间通过 scatter 把 OmniEncoder 输出当作 prompt token embedding 注入，**不引入 cross‑attention**——这是"前融合"的核心。 |
| I11 | 4D MTHW RoPE 与视频分层融合（`video_omni_*_mthw_rope` / `video_fusion_layer_freq` / `video_group_attention`）只作用于 **OmniEncoder 内部**，默认全关时等价于 2D/1D RoPE + 整段双向注意力；**任何配置下都不改变 LM 端的标准 RoPE / causal 行为**。 |

---

## 4. Config 关键字段速查

| 字段 | 说明 |
|---|---|
| `text_config` (`Qwen3VITATextConfig`) | LM 主体：`hidden_size`、`num_hidden_layers`、`num_attention_heads`、`num_key_value_heads`、`head_dim`、`rope_theta`、`max_position_embeddings`、可选 `sliding_window`、`vocab_size`。 |
| `omni_config` (`Qwen3VITAOmniConfig`) | 共享 OmniEncoder 配置；除继承 `Qwen3VITATextConfig` 的 Transformer 字段外，关键多模态超参（默认值见源码 `modular_qwen3_vita.py:215-280`）： |
| └ `num_channels` | 视觉通道数，默认 `3`。 |
| └ `patch_size` | 视觉 patch 大小，默认 `16`。 |
| └ `spatial_merge_size` | 视觉 merger 空间合并因子，默认 `2`（即 2×2）。 |
| └ `num_mel_bins` | 音频 mel 维数，默认 `128`。 |
| └ `downsample_hidden_size` | 音频前端 Conv2d 通道数（也是 3 层 8× 降采的隐维），默认 `512`。 |
| └ `temporal_merge_size` | 音频 merger 时间合并因子，默认 `2`。 |
| └ `n_window` / `n_window_infer` | 音频前端的 attention 窗口长度（train/infer 各一份），默认 `50` / `800`。 |
| └ `conv_chunksize` | 音频 Conv2d 分块大小（控显存），默认 `500`。 |
| └ `merger_hidden_size` / `out_hidden_size` | merger MLP 隐维 / 投到 LM 的输出隐维，默认均 `4608`。 |
| └ `video_group_attention` | fusion 层内是否按 `(I+A*)` 分组隔离注意力，默认 `False`（见 §1.3.4）。 |
| └ `video_fusion_layer_freq` | 逐层 fusion 掩码（`None` / `int N` / `list[int]`），默认 `None`（全 fusion）。 |
| └ `video_omni_chunked_mthw_rope` / `video_omni_interleaved_mthw_rope` | 两种互斥的 4D (M\|T\|H\|W) RoPE 开关，默认均 `False`（用 2D/1D 传统 RoPE）；同开时 interleaved 优先。 |
| └ `video_omni_interleaved_thw_section` | interleaved 变体的显式 `(t,h,w)` 切分，默认 `None`（自动）。 |
| └ `rope_m_dim` / `rope_theta_m` / `rope_theta` | 4D RoPE 的 M 段维数 / M 段 theta / T·H·W 段 theta，默认 `4` / `100.0` / `10000.0`；`rope_m_dim=0` 退化为纯 3D。 |
| 顶层 `Qwen3VITAConfig` 字段 | |
| └ `video_omni_fusion` | **视频范式总开关**，默认 `False`（范式 A）；`True` 走范式 B 联合视频编码（见 §2.4）。 |
| └ `tie_word_embeddings` | 顶层默认 `False`，但 `Qwen3VITAForCausalLM._tied_weights_keys` 仍把 `lm_head.weight` 绑到 `embed_tokens.weight`。 |
| `_tied_weights_keys` | `{"lm_head.weight": "model.embed_tokens.weight"}`。 |
| `_tp_plan` | `{"lm_head": "colwise_gather_output"}`。 |
| `_pp_plan` | `{"lm_head": (["hidden_states"], ["logits"])}`。 |

---

## 5. `qwen3_vita` vs `youtu_vita` 的差异

| 维度 | `qwen3_vita` | `youtu_vita` |
|---|---|---|
| 与对方的关系 | 范式定义方 | **独立完整拷贝**：omni 范式部分与 `qwen3_vita` 逐行一致，仅前缀 `Qwen3VITA*`→`YoutuVITA*`；**不通过 `import` 继承 `qwen3_vita`**。 |
| `omni_config` 默认启用 | 是（本范式主线） | 是（本范式主线） |
| OmniEncoder | `Qwen3VITAOmniModel(Qwen3VITAOmniPreTrainedModel)` | `YoutuVITAOmniModel(YoutuVITAOmniPreTrainedModel)`（结构一致） |
| LanguageModel | `Qwen3VITATextModel`（基于 Qwen3） | `YoutuVITATextModel`（基于 `Youtu*`，同 Qwen3 结构） |
| 顶层 `Model` / `ForCausalLM` | `Qwen3VITAModel` / `Qwen3VITAForCausalLM` | `YoutuVITAModel` / `YoutuVITAForCausalLM`（forward 签名一致） |
| 特殊 token 集 | `processing_qwen3_vita.py` 内 | 两套：`Youtu_VITA_TOKEN_bus1`（`modular_youtu_vita.py:4112`，仅含 image）与 `Youtu_VITA_TOKEN`（`:4219`，含 image/audio/video + think/code/tool_call/action 等），默认用后者（`:4357`） |
| Tokenizer 内嵌工具 | `tokenization_qwen3_vita.py` | 额外内嵌 `GLM4VoiceTokenizer`（`:4391`）、`WavFrontendTokenizer`（`:4596`，依赖 `funasr.frontends.wav_frontend.WavFrontend`）、`MelFilterBankTokenizer` 等 |
| Video processor | `Qwen3VITAVideoProcessor` | `YoutuVITAVideoProcessor`（更复杂的视频分桶；范式 A/B 仍由顶层 `config.video_omni_fusion` 决定） |
