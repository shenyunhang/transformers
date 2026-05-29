# VITA 系列模型（`qwen3_vita` / `youtu_vita`）`omni_encoder + llm_decoder` 范式模型结构与推理流程说明

> 涉及代码：
> - `src/transformers/models/qwen3_vita/`（核心：`modular_qwen3_vita.py`，自动生成 `modeling_qwen3_vita.py`）
> - `src/transformers/models/youtu_vita/`（核心：`modular_youtu_vita.py`，自动生成 `modeling_youtu_vita.py`）
>
> 文档定位：**专门介绍上述两个模型采用的 `omni_encoder + llm_decoder` 范式**，包括其模型结构、各模态预处理与 forward 推理流程。
> - 仅描述 **omni 路径**（即 `config.omni_config is not None` 的主线 checkpoint）；早期独立 vision/audio encoder 后备分支不在本文范围。
> - 不涉及训练、对齐验收、引擎适配细节。

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
│                共享 Qwen3 Transformer Body (双向 + packed varlen)        │
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

`qwen3_vita`（以 modular 文件为权威源）：

| 类 | 角色 |
|---|---|
| `Qwen3VITAConfig` | 顶层 config：`text_config` + `omni_config`（本范式下二者均必填）。 |
| `Qwen3VITAOmniConfig`（继承 `Qwen3VITATextConfig`） | OmniEncoder 的 Transformer 配置（继承 Qwen3 结构），同时含视觉/音频前端的超参（patch、mel、merger 等）。 |
| `Qwen3VITATextConfig`（继承 `Qwen3Config`） | LanguageModel 的 Qwen3 配置。 |
| `Qwen3VITAOmniVisionEmbeddings` | 视觉前端：`nn.Linear(in=C·patch²·temporal_patch, out=H_enc)`，把已 patchify 的像素投到 Encoder 隐维。 |
| `Qwen3VITAOmniAudioEmbeddings` | 音频前端：3 层 stride‑2 Conv2d（**对 mel 时间轴 8× 降采样**）+ 线性投影到 `H_enc`。 |
| `Qwen3VITAOmniEncoder` | 共享 Transformer body：双向、无 causal mask、packed varlen 注意力；视觉走 2D RoPE，音频走 1D RoPE。 |
| `Qwen3VITAVisionPatchMerger` / `Qwen3VITAAudioPatchMerger` | merger + projector：视觉 `spatial_merge_size=2`（即 2×2）、音频 `temporal_merge_size=2`；其后接 2 层 MLP 投到 LM 隐维 `out_hidden_size = H`。 |
| `Qwen3VITAOmniModel` | 上述前端 + body + merger 的封装；`forward(modality=...)` 分派 `forward_vision` / `forward_audio` / `forward_video`。 |
| `Qwen3VITATextModel` | 自回归 LM 主体（Qwen3：qk‑norm、GQA、SwiGLU、RMSNorm、标准 RoPE、可选 sliding window）。 |
| `Qwen3VITAModel` | 顶层组合：`omni_model` + `language_model`。 |
| `Qwen3VITAForCausalLM` | `GenerationMixin` 入口，含 `lm_head`（与 `embed_tokens` tied）。 |

`youtu_vita`：在结构上几乎完全继承自 `qwen3_vita`（`YoutuVITAOmniModel(Qwen3VITAOmniModel): pass`），差异集中在 **特殊 token 集合** 与 **video processor** 两处（见 §5）。

### 1.3 架构特点

1. **OmniEncoder 是共享的多模态前融合 Encoder**：vision/audio 前端各自投到同一隐维后，**复用同一份 Transformer 权重**做双向编码，仅在最后由各自 patch merger + 2 层 MLP 分别投到 LM 隐维。
2. **Encoder 完全双向 + packed varlen**：
   - 视觉用 2D RoPE，按 `image_grid_thw` 拼成 `cu_seqlens`；
   - 音频用 1D RoPE，按音频特征长度拼成 `cu_seqlens`；
   - **不会进入 LM 的 KV cache**，每条 prompt 在 prefill 阶段一次性算完。
3. **LanguageModel 是标准 Qwen3 Decoder**：
   - causal、自回归、有 KV cache；
   - **标准 RoPE，不是 M‑RoPE**（与 Qwen2‑VL/Qwen2.5‑Omni 不同）；
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

---

## 2. 各模态预处理

> 预处理由 `Qwen3VITAProcessor` / `YoutuVITAProcessor`（`processing_*.py`） + `*ImageProcessor` / `*VideoProcessor` / `*FeatureExtractor` 组合完成，并由 tokenizer 在文本侧插入相应数量的占位 token。

### 2.1 文本与特殊 token

LM 词表包括：

- 普通文本 token（Qwen3 tokenizer）；
- 多模态占位 / 边界 token（字面量见下表，引用自 `youtu_vita/modular_youtu_vita.py:Youtu_VITA_TOKEN`，`qwen3_vita` 同套）；
- 16384 个离散音频输出 token `<\|audio_*\|>`（仅 S2S 场景下出现在输出端）；
- 控制 token：`Youtu_VITA_TOKEN_bus1`（`modular_youtu_vita.py:2056`，bus1 精简版，仅含 image）与 `Youtu_VITA_TOKEN`（`:2164`，含 image/audio/video/think/code/tool_call/action 等完整集），按 config 选择。

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

1. 原始波形经 `WavFrontendTokenizer` / `MelFilterBankTokenizer`（`youtu_vita/modular_youtu_vita.py:2541/2634`）得到 mel 特征 `[T_i, num_mel_bins]`（`num_mel_bins=128`），按样本组成 list。
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

视频可走 **两种处理范式**，二者互斥；**范式选择由 processor 的 `video_omni_fusion` 开关决定**（默认 `False` → 范式 A；`True` 时 processor 会输出非空 `video_split` 张量，模型侧路由到 `forward_video`）。源码判定：`Qwen3VITAOmniModel.forward` 中 `has_video = video_split is not None and video_split.numel() > 0`。

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
- `video_split: LongTensor[N_video, 2]` —— 每个视频的 `(num_images, num_audio_chunks)`，由 processor 在写完每段视频后追加一行（`modular_qwen3_vita.py:5250`）。

##### 关键中间量

`forward_video` 内部对每个视频 `v` 单独建立两组 chunk：

- 视觉 chunk 列表 `vision_feature_chunks[v]`：**按帧**切分，第 `k` 个 chunk 形状 `[tokens_per_frame[k], H_enc]`，其中 `tokens_per_frame[k] = T_k · H_p_k · W_p_k`（`merger 之前` 的视觉 token 数）；
- 音频 chunk 列表 `audio_feature_chunks[v]`：**按音频块**切分，第 `j` 个 chunk 形状 `[audio_token_lens[j], H_enc]`，`audio_token_lens[j]` 为该 chunk 在 `audio_embeddings` 之后（8× 降采）但 **merger 之前** 的 token 数；
- `vision_rotary_chunks[v]` / `audio_rotary_chunks[v]`：与上面一一对应的 RoPE 张量（视觉 2D，音频 1D），shape 与对应 feature chunk 沿第 0 维一致。

##### 交错规则（核心）

```python
num_image_chunks         = len(vision_feature_chunks)
num_audio_chunks_actual  = len(audio_feature_chunks)
audios_per_image, extra_audio_count = divmod(num_audio_chunks_actual, num_image_chunks)
```

- 仅有视觉 → 直接按帧顺序排；
- 仅有音频 → 直接按 chunk 顺序排；
- 二者皆有 → **每张图后挂 `audios_per_image` 个音频块；前 `extra_audio_count` 张图各多挂 1 个**（保证总数严格对齐）。

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
| `pixel_values` / `images` | `FloatTensor[N_patches_all, C·patch²·temporal_patch]` | patchify 后的图像。 |
| `image_grid_thw` | `LongTensor[N_images, 3]` | 每张图的 `(T, H_patches, W_patches)`。 |
| `image_indices` | `LongTensor[2, N_img_tokens]` | 全局 `(batch_idx, seq_idx)`；个数 = merger 后视觉 token 数。 |
| `audios` / `input_features` | `list[FloatTensor[T_i, num_mel_bins]]` | 经 mel + WavFrontend 处理；模型前端内部 `cat+transpose` 拼接。 |
| `audio_indices` | `list[LongTensor[2, n_i]]` | per‑sample `(batch_idx, seq_idx)`。 |
| `video_image_indices` | `LongTensor[2, N]` | 视频帧对应的占位索引。 |
| `video_audio_indices` | `list[LongTensor[2, n_i]]` | 视频音轨对应的占位索引。 |
| `video_split` | `LongTensor[N_video, 2]` 或 `None` | **范式 B 必填**：每个视频的 `(num_images, num_audios)`，用于 `forward_video` 内构造交错 segment 与 OmniEncoder 注意力隔离。`None` 时走范式 A。 |
| `position_ids`、`past_key_values`、`use_cache`、`cache_position` | — | LM 标准字段。 |

---

## 3. Forward 流程

下面给出 **推理阶段**（生成或一次性 logits）`Qwen3VITAModel.forward` / `YoutuVITAModel.forward` 的执行序列，所有步骤同样适用于 `youtu_vita`。

### 3.0 调用分派速查

| 输入存在的字段 | 触发调用 | 输出 |
|---|---|---|
| `pixel_values` + `image_grid_thw`（无 `video_split`） | `_encode_vision → omni_model(modality="vision")` → `forward_vision` | `image_embeds: [N_img_tokens, H]` |
| `audios`（非视频上下文） | `_encode_audio → omni_model(modality="audio")` → `forward_audio` | `(audio_embeds: [B, S_a, H], audio_lengths)` |
| `video_images` + `video_image_grid_thw`，`video_split=None` | 范式 A：与上面两条相同，仅用 `video_image_indices` / `video_audio_indices` 写回 | 同上 |
| `video_images` + `video_audios` + `video_split` 非空 | 范式 B：`omni_model.forward_video(...)` | `(video_image_embeddings, video_audio_embeddings, lens)` |

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

4. **视频编码**（当存在视频帧/视频音轨）—— **两种范式互斥，由 `video_split` 是否非空决定**：
   - **范式 A**：`video_split is None`。视觉部分复用 §3.1.2（输入是该视频的全部帧 patch）、音频部分复用 §3.1.3（输入是该视频的全部音频块）；两路在 OmniEncoder 内 **互不感知**，分别输出 `video_image_embeds`、`video_audio_embeds`。
   - **范式 B**：`video_split is not None`。调 `omni_model.forward_video(video_images, video_image_grid_thw, video_audios, video_split)`：
     1. 分别过 `vision_embeddings` / `audio_embeddings` 拿到 visual/audio features；
     2. 按 `video_split` 把同一视频的 visual chunk 与 audio chunk 取出，按"每张图后挂 `audios_per_image` 个音频块、前 `extra_audio_count` 张图多挂 1 个"做 **交错拼接**，形成一个 segment，同时拼好 1D/2D 混合 RoPE 与 `modality_mask`；
     3. 多视频的 segment 通过 `cu_seqlens` 在 OmniEncoder 内 **共享 attention 窗口但相互隔离**；
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
| I9 | 视频走两种范式之一，由 processor 的 `video_omni_fusion` 开关决定：`False`（默认）→ `video_split=None` → 范式 A（拆分模态独立）；`True` → `video_split` 非空 → 范式 B（联合视频编码，OmniEncoder 内音视频 chunk 交错并按视频隔离 attention）。两者共享同一份 OmniEncoder 权重，最终都用 `video_image_indices` / `video_audio_indices` scatter 回 `inputs_embeds`。 |
| I10 | OmniEncoder **双向 + 无 causal mask**；LM **严格 causal**；两者之间通过 scatter 把 OmniEncoder 输出当作 prompt token embedding 注入，**不引入 cross‑attention**——这是"前融合"的核心。 |

---

## 4. Config 关键字段速查

| 字段 | 说明 |
|---|---|
| `text_config` (`Qwen3VITATextConfig`) | LM 主体：`hidden_size`、`num_hidden_layers`、`num_attention_heads`、`num_key_value_heads`、`head_dim`、`rope_theta`、`max_position_embeddings`、可选 `sliding_window`、`vocab_size`。 |
| `omni_config` (`Qwen3VITAOmniConfig`) | 共享 OmniEncoder 配置；除继承 `Qwen3VITATextConfig` 的 Transformer 字段外，关键多模态超参（默认值见源码 `modular_qwen3_vita.py:213-229`）： |
| └ `patch_size` | 视觉 patch 大小，默认 `16`。 |
| └ `spatial_merge_size` | 视觉 merger 空间合并因子，默认 `2`（即 2×2）。 |
| └ `num_mel_bins` | 音频 mel 维数，默认 `128`。 |
| └ `downsample_hidden_size` | 音频前端 Conv2d 通道数（也是 3 层 8× 降采的隐维），默认 `512`。 |
| └ `temporal_merge_size` | 音频 merger 时间合并因子，默认 `2`。 |
| └ `n_window` / `n_window_infer` | 音频前端的 attention 窗口长度（train/infer 各一份），默认 `50` / `800`。 |
| └ `conv_chunksize` | 音频 Conv2d 分块大小（控显存），默认 `500`。 |
| `image_token_id` / `audio_token_id` / `video_token_id` | 仅用于 processor 端占位标记；模型侧靠 `*_indices` 直接定位，不强依赖 id。 |
| `_tied_weights_keys` | `{"lm_head.weight": "model.embed_tokens.weight"}`。 |
| `_tp_plan` | `{"lm_head": "colwise_gather_output"}`。 |
| `_pp_plan` | `{"lm_head": (["hidden_states"], ["logits"])}`。 |

---

## 5. `qwen3_vita` vs `youtu_vita` 的差异

| 维度 | `qwen3_vita` | `youtu_vita` |
|---|---|---|
| `omni_config` 默认启用 | 是（本范式主线） | 是（本范式主线） |
| OmniEncoder | `Qwen3VITAOmniModel` | `YoutuVITAOmniModel(Qwen3VITAOmniModel): pass`（结构无差异） |
| LanguageModel | `Qwen3VITATextModel` | `YoutuVITATextModel`（同样基于 Qwen3） |
| 顶层 `Model` / `ForCausalLM` | `Qwen3VITAModel` / `Qwen3VITAForCausalLM` | `YoutuVITAModel` / `YoutuVITAForCausalLM`（forward 签名一致） |
| 特殊 token 集 | `processing_qwen3_vita.py` 内 | 两套：`Youtu_VITA_TOKEN_bus1`（`:2056`，仅含 image）与 `Youtu_VITA_TOKEN`（`:2164`，含 image/audio/video + think/code/tool_call/action 等），按 config 选择 |
| Tokenizer 内嵌工具 | `tokenization_qwen3_vita.py` | 额外内嵌 `GLM4VoiceTokenizer`、`WavFrontendTokenizer`、`MelFilterBankTokenizer`、`VisionTokenizer`、`AudioTokenizer` |
| Video processor | `Qwen3VITAVideoProcessor` | `YoutuVITAVideoProcessor`（更复杂的视频分桶；范式 A/B 仍由 `video_split` 是否非空决定） |
