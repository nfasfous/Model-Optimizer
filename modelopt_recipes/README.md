# ModelOpt Recipes

This folder is the library of **ModelOpt optimization recipes** — declarative
YAML files that describe a complete model-optimization workflow (post-training
quantization, speculative-decoding training, diffusion distillation).

**Purpose:** a recipe is the single, version-controlled source of truth for *how*
a model is optimized — algorithm, per-layer numeric formats, and calibration —
expressed as data instead of code. That makes an optimization run reproducible,
diffable, and shareable without hand-writing Python config, and lets a tuned
configuration be looked up by name. The same YAML drives the Python API
(`load_recipe`), the example CLIs (`--recipe`), and — for the presets under
`configs/` — the built-in `*_CFG` constants.

Recipes are composed from small, reusable building blocks via an `$import`
system, then loaded by path relative to this folder, e.g.:

```python
# PTQ recipe -> mtq.quantize()
from modelopt.recipe import load_recipe
cfg = load_recipe("general/ptq/nvfp4_default-kv_fp8_cast")

# distillation recipe -> DMDConfig
from modelopt.torch.fastgen import load_dmd_config
cfg = load_dmd_config("general/distillation/dmd2_qwen_image")
```

or selected from a script/CLI flag, e.g. `hf_ptq.py --recipe
huggingface/qwen3_5/ptq/w4a16_nvfp4-fp8_attn-kv_fp8_cast`.

> 📖 **Must-read for PTQ recipe tuning → [`ptq.md`](ptq.md).** It is the
> guide to every PTQ scheme — body scopes (NVFP4/FP8, experts-only / mlp-only /
> weight-only), KV-cache modes, and calibration variants — with concrete guidance
> on **choosing and tuning a recipe** for your model and deployment. Start there
> before picking a recipe.
>
> This README is the **catalog** across all recipe families; `ptq.md` is the
> how-to for PTQ.

## Layout

| Directory | What lives here |
|-----------|-----------------|
| `general/` | **Model-agnostic** recipes — a good starting point for any model. PTQ combos, speculative-decoding training, and distillation. |
| `huggingface/<model_type>/` | **Model-specific** recipes keyed by a HF `model_type`, optionally nested by released checkpoint. Use these first if your model has an entry. |
| `models/<model_name>/` | **Instance-specific** recipes that mirror a particular published checkpoint's quantization config. |
| `configs/` | Shared building blocks (`numerics/`, `ptq/units/`, `ptq/presets/`) that recipes compose from via `$import`. Not run directly. |

**Choosing where to look:** check `huggingface/<model_type>/` (then any nested
`<checkpoint>/`) for your model first; if there's no entry, fall back to
`general/`. The presence of a model folder signals a recommended, tuned recipe.

---

## The composition model (how the combos are built)

Most PTQ recipes are **not** one-off configs — they are a mix-and-match of four
independent axes, which is why `general/ptq/` looks like a combinatorial matrix.
The file name encodes the choices: `<weight-scope>-<kv-mode>[-<algorithm>].yaml`.

| Axis | What it controls | Common choices |
|------|------------------|----------------|
| **Numeric format** | The precision of the quantized tensors | `fp8` (per-tensor E4M3, W8A8), `nvfp4` (E2M1 block W4A4 w/ FP8 scales), `int4`/`int8`, and `mxfp4`/`mxfp6`/`mxfp8`/`mxint8` (available as building blocks/presets) |
| **Scope** | *Which* layers get quantized | `default` (whole model), `mlp_only` (MLP/MoE blocks), `experts_only` (MoE routed experts), `omlp_only` (MLP/MoE + attention output proj), `weight_only` (weights only, W4A16 — activations stay BF16) |
| **KV-cache mode** | How (or whether) the attention KV cache is quantized | `kv_fp8` (calibrated), `kv_fp8_cast` (FP8 with constant amax — **no KV calibration**), `kv_nvfp4_cast`, or `kv_fp16`/`none` (KV left unquantized) |
| **Calibration algorithm** | How scales are searched during calibration | `max` (default, fast), `mse` (often with `fp8_scale_sweep`), `gptq` (layerwise), and the AWQ/SmoothQuant/SVDQuant families (mostly via presets) |

Reading a name: `nvfp4_experts_only_mse-kv_fp8_cast` = NVFP4 numerics, applied to
MoE experts only, MSE weight calibration, with an FP8 cast KV cache.

### `general/ptq/` — the available combos

These are the model-agnostic PTQ recipes shipped today. Rather than memorize
each file, read them along the axes above:

| Numeric × Scope | KV-cache variants shipped | Calibration variants |
|-----------------|---------------------------|----------------------|
| **FP8, whole model** (`fp8_default`) | `kv_fp8`, `kv_fp8_cast` | max |
| **NVFP4, whole model** (`nvfp4_default`) | `kv_fp8`, `kv_fp8_cast`, `kv_nvfp4_cast`, `kv_none` | max, and `gptq` (layerwise) for `kv_none` |
| **NVFP4, MLP/MoE only** (`nvfp4_mlp_only`) | `kv_fp8`, `kv_fp8_cast` | max, and `_mse` (FP8-scale sweep) |
| **NVFP4, experts only** (`nvfp4_experts_only`) | `kv_fp8`, `kv_fp8_cast`, `kv_fp8_layerwise` | max, layerwise, and `_mse` |
| **NVFP4, output-proj + MLP/MoE** (`nvfp4_omlp_only`) | `kv_fp8`, `kv_fp8_cast` | max |
| **NVFP4, weight-only / W4A16** (`nvfp4_weight_only`) | `kv_fp16` (none), `kv_fp8_cast` | max (no calibration forward needed) |
| **INT4 blockwise weight-only** (`int4_blockwise_weight_only`) | — (weight-only) | max |

The narrower scopes (`experts_only`, `mlp_only`, `weight_only`) exist as
accuracy/perf trade-offs: quantize the heavy MoE/MLP weights for the memory and
throughput win while leaving the sensitive attention path in higher precision.

### `general/speculative_decoding/` — draft-model training recipes

Not quantization — these bundle model/data/training/algorithm arguments for
training a speculative-decoding draft head. CLI-overridable via OmegaConf dotlists.

- **`eagle3.yaml`** — EAGLE3 draft-head training (TTT steps, self-logit
  distillation, YaRN rope injected at export for long context).
- **`dflash.yaml`** — DFlash draft-head training (block/anchor config, longer
  default schedule, answer-only loss with a chat template).

### `general/distillation/` — diffusion distillation

- **`dmd2_qwen_image.yaml`** — DMD2 few-step (4-step) distillation for the
  Qwen-Image rectified-flow text-to-image model; maps to `DMDConfig`.

---

## `configs/` — shared building blocks

Recipes don't usually redefine numerics; they `$import` from here. You normally
edit/compose these rather than run them.

- **`numerics/`** — the lowest-level *format* definitions (a single quantizer's
  attributes): `fp8`, `nvfp4`, `nvfp4_static`, `nvfp4_bs32`, `int4_per_block`,
  `int8`/`int8_per_channel`, `mxfp4`/`mxfp6`/`mxfp8`/`mxint8`.
- **`ptq/units/`** — reusable `quant_cfg` fragments (not standalone): the
  `base_disable_all` deny-all prefix, `default_disabled_quantizers` (LM head,
  routers, BatchNorm, etc.), weight+activation blocks (`w8a8_fp8_fp8`,
  `w4a4_nvfp4_nvfp4`, `w4_nvfp4`), scope blocks (`experts_nvfp4`,
  `block_sparse_moe_nvfp4`, `attention_qkv_fp8`), and the KV-cache units
  (`kv_fp8`, `kv_fp8_cast`, `kv_fp8_affine`, `kv_nvfp4`, `kv_nvfp4_cast`,
  `kv_nvfp4_affine`, `kv_nvfp4_rotate`).
- **`ptq/presets/`** — complete, ready-to-pass configs that are the YAML source
  of truth for the hardcoded `*_CFG` constants (e.g. `FP8_DEFAULT_CFG`):
  - `model/` — full presets (fp8, int8/SmoothQuant, INT4-AWQ, the NVFP4-AWQ
    family, `w4a8_*`, `mamba_moe_*`, mx-formats, weight-only variants, …).
  - `kv/` — KV-cache-only fragments meant to be merged onto a `model/` preset.
  - `diffusers/` — Diffusers/Flux full presets (FP8, INT8, NVFP4, NVFP4+FP8-MHA
    with SVDQuant) that differ from the generic presets in attention/softmax
    handling.

> Presets exist mainly for backward-compat with `hf_ptq.py`'s `--qformat` /
> `--kv_cache_qformat` flags; new work should prefer `load_recipe` on the
> `general/` or `models/` recipes.

---

## `huggingface/` — model-specific recipes

Each lives under its HF `model_type`. The point of a model folder is to capture
**what differs from the generic preset** — usually an algorithm tweak or a
disabled-quantizer pattern for non-text branches. The numerics and standard
exclusions are still inherited from `configs/`.

| Model (`model_type`) | What's model-specific |
|----------------------|------------------------|
| **`gemma`** | Algorithm overrides for stability: `w4a8_awq` uses `awq_lite` with `alpha_step: 1` (default AWQ search overflows TRT-LLM kernels on Gemma); `int8_sq` uses SmoothQuant `alpha: 0.5` (Gemma 7B regresses at the default `1.0`). |
| **`mpt`** | Same `awq_lite alpha_step: 1` override as Gemma — the default AWQ search overflows TRT-LLM kernels on MPT. |
| **`nemotron_vl`** | Vision-language family (incl. Nemotron-Parse). Ships a `disabled_quantizers.yaml` unit that adds `*vision*`/`*image*`/`*radio*`/`*visual*`/`*encoder*`/`*model_encoder*` to the standard exclusions so only the text decoder is quantized; the NVFP4 recipe imports it. |
| **`phi4mm`** (Phi-4-Multimodal) | Same idea — a `disabled_quantizers.yaml` that excludes `*speech*`/`*audio*`/`*image*`/`*vision*` so only the language model is quantized. |
| **`qwen3_5` / `qwen3_5_moe`** | Hybrid linear-attention + softmax-attention architecture. A shared `quant_cfg` snippet (W4A16 NVFP4 on MLP/expert projections + lm_head, FP8 on self-attention and the large linear-attention projections, FP8 KV cast) is imported by both the dense and MoE recipes; includes Qwen-specific disables for `in_proj_a/b`, visual, and MTP siblings. |
| **`step3p5`** (Step3.5-Flash) | Instance-tuned recipe enabling dynamic NVFP4 on MoE/MLP weights+inputs and FP8 KV cache, with shared-expert gate / router / conv1d / mamba branches left disabled. |

> Convention: model recipes are named `<qformat>-...-kv_<mode>.yaml`; when a body
> is shared (e.g. across dense/MoE variants) it's extracted into a sibling
> `<recipe>.<field>.yaml` snippet that each recipe `$import`s. Most folders carry
> a `README.md` spelling out the exact delta from the generic preset.

## `models/` — checkpoint-specific recipes

These mirror a single **published checkpoint's** quantization config exactly.

- **`Nemotron-3-Super-120B-A12B/`** — reproduces the
  `NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` release: a **mixed-precision** scheme
  (MoE routed experts NVFP4 W4A4 `group_size 16`; shared experts and Mamba
  in/out projections FP8 per-tensor; FP8 KV cache; attention, MTP head, lm_head,
  latent-MoE, and Mamba conv1d kept in BF16). Two variants differ only in
  calibration: `super-nvfp4.yaml` (weight MSE with FP8-scale sweep — matches the
  release) and `super-nvfp4-max-calib.yaml` (plain amax/max calibration, for
  comparison).

---

## Adding a recipe

- **New combo for any model** → add to `general/ptq/` by composing existing
  `configs/` units; follow the `<scope>-<kv>[-<algo>]` naming.
- **Tuned for a HF architecture** → `huggingface/<model_type>/<task>/`, with a
  `README.md` documenting the delta from the generic preset. Verify the exact
  `model_type` against the checkpoint's `config.json` before placing it.
- **Mirrors a specific released checkpoint** → `models/<model_name>/`.
- Share reused bodies via a `# modelopt-schema:`-tagged snippet and `$import`
  it; keep recipe wrappers thin.
