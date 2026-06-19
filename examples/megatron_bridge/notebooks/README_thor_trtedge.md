# Running Notebook 07 on the Edge — Cosmos-Reason 2 with **TensorRT Edge-LLM** (NVFP4) on Thor

[`07_cosmos_reason_thor_trtedge_llm.ipynb`](07_cosmos_reason_thor_trtedge_llm.ipynb) is the
**robotics-native twin** of [`06_cosmos_reason_thor_nvfp4_edge.ipynb`](06_cosmos_reason_thor_nvfp4_edge.ipynb).
nb06 served the model with **vLLM** (a datacenter server, convenient for a uniform eval harness). nb07
deploys it the way an embodied agent actually does — as a **compiled TensorRT engine** driven by
**[NVIDIA TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)**, the light-weight C++ LLM/VLM
inference SDK for **NVIDIA Jetson Thor / DRIVE Thor** (JetPack 7.x, Blackwell `sm_110`).

The pipeline: **ModelOpt NVFP4 PTQ** (unchanged from nb06) → `tensorrt-edgellm-export` (ONNX for the LM
**and** the vision tower) → `llm_build` + `visual_build` (TensorRT engines) → in-process `LLM` Python API.
Quality is measured with Edge-LLM's **native MMStar/MMMU accuracy harness** and batch-1 robotics
performance with **`llm_bench`** — both run on the device, no server.

| Platform | GPU | Compute capability | Power telemetry (§7a) |
|---|---|---|---|
| **NVIDIA Thor AGX** | Tegra Blackwell iGPU | `sm_110` (11.0) | INA3221 rails via `/sys/class/hwmon` |

> Edge-LLM targets the Tegra `sm_110` runtime. (nb06's DGX-Spark `sm_121` path stays on vLLM; the §7a
> power cell for Spark is still included in nb07 for completeness, but the Edge-LLM build flags below are
> Thor-specific.)

---

## Why a separate toolchain (and not nb06's vLLM image)

nb06 used a prebuilt L4T **vLLM** wheel. TensorRT Edge-LLM is a **C++ SDK with no PyPI wheel** — it is
built from source on the device (cmake + make). It produces the C++ binaries (`llm_build`,
`llm_inference`, `llm_bench` under `build/examples/llm/`; `visual_build` under
`build/examples/multimodal/`) and, with `-DBUILD_PYTHON_BINDINGS=ON`, the experimental Python layer
(`experimental.server` — the high-level `LLM` class + an OpenAI-compatible server — and
`tensorrt_edgellm`, the exporter). nb07 needs all of these, plus **ModelOpt** (from this repo) for §4.

---

## Prerequisites

- An **NVIDIA Jetson Thor / DRIVE Thor** with **JetPack 7.x** (CUDA 13.0, TensorRT 10.x installed under
  `/usr`) and Docker + the NVIDIA container toolkit.
- The model checkpoints on the host, mounted to `/hf`:
  `Cosmos-Reason2-2B`, `Cosmos-Reason2-distilled`, `Cosmos-Reason2-fp8`, `Cosmos-Reason2-distilled-fp8`
  (the two NVFP4 variants are produced by §4 of the notebook).
- A HuggingFace token (read scope) — §6's `prepare_dataset.py` downloads MMStar / MMMU.

> Adjust the two host paths below (the repo checkout and the `…/hf` model directory) to wherever they
> live on your machine.

---

## Steps

### 1. Build the TensorRT Edge-LLM SDK

**Option A — the Edge-LLM container (recommended).** Build the blessed SDK image from the Edge-LLM repo:
```bash
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
./experimental/docker/build_container.sh      # -> tags tensorrt-edge-llm:experimental
```

**Option B — from source on the device (no docker).** Builds the C++ binaries + Python bindings in place:
```bash
sudo apt update && sudo apt install -y cmake build-essential git
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM && git submodule update --init --recursive
pip install -r requirements-server.txt        # fastapi / pybind11 / uvicorn — BEFORE cmake
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-thor \
  -DCUDA_CTK_VERSION=13.0 \
  -DENABLE_CUTE_DSL=ALL \
  -DBUILD_PYTHON_BINDINGS=ON
make -j$(nproc)                                # ~1-2 min; tens of GB of build space
cd .. && export PYTHONPATH=$PWD:$PYTHONPATH     # makes experimental.server importable
```
**Build the example CLIs.** The Edge-LLM container/`build.sh` compiles only the plugin +
pybind runtime — **not** the example CLIs (`llm_build` / `llm_inference` / `llm_bench` /
`visual_build`) that §5–§7 use. The `build/` is already configured, so build those four targets
(quick incremental compile against the runtime):
```bash
# Option A (container): run inside it once the container is up (see step 3); Option B (host):
cmake --build build --target llm_build llm_inference llm_bench visual_build --parallel "$(nproc)"
```
Verify:
```bash
./build/examples/llm/llm_build --help
python -c "from experimental.server import LLM, SamplingParams; print('edge-llm OK')"
```
If the import raises on `_edgellm_runtime`:
`TRT_PACKAGE_DIR=/usr python experimental/server/setup_pybind.py build_ext --inplace`.

### 2. Build the notebook image
Layer ModelOpt + the notebook tooling on top of the Edge-LLM SDK image
([`Dockerfile.thor.trtedge`](Dockerfile.thor.trtedge)):
```bash
cd <your-repo-root>/Model-Optimizer
docker build -f examples/megatron_bridge/notebooks/Dockerfile.thor.trtedge \
  --build-arg BASE_IMAGE=tensorrt-edge-llm:experimental \
  --build-arg EDGE_LLM_HOME=/workspace/TensorRT-Edge-LLM \
  -t cosmos-edge-trt:latest examples/megatron_bridge/notebooks
# If you built the SDK from source (Option B) instead of a container, skip this and run
# JupyterLab directly from a shell where PYTHONPATH includes the Edge-LLM checkout, with
# EDGE_LLM_HOME exported (the notebook reads it in §1).
```

### 3. Start the container (Thor — iGPU via the NVIDIA runtime)
```bash
docker run -d --name cosmos-trt --runtime nvidia --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e EDGE_LLM_HOME=/workspace/TensorRT-Edge-LLM \
  -v <repo>/Model-Optimizer:/opt/Model-Optimizer \
  -v <host>/hf:/hf \
  cosmos-edge-trt:latest sleep infinity
```

### 4. Point modelopt at the mounted local repo
The Edge-LLM image ships a **physical** `nvidia-modelopt` under `dist-packages` that shadows the
editable install — remove the directory itself (not just the `.dist-info`), else `import modelopt`
resolves to the stale copy (you'll see `ImportError: cannot import name 'NVFP4StaticQuantizer'`):
```bash
docker exec cosmos-trt bash -lc \
  'rm -rf /usr/local/lib/python3.12/dist-packages/modelopt; \
   find / -name "nvidia_modelopt-*.dist-info" -exec rm -rf {} + 2>/dev/null; \
   pip install -e /opt/Model-Optimizer --no-deps -q'
```

### 5. Add your HuggingFace token
This image sets `HF_HOME=/data/models/huggingface`, so `huggingface_hub` looks for the token there
(not `~/.cache/huggingface`). Place it at `$HF_HOME/token`:
```bash
docker exec cosmos-trt bash -lc \
  'mkdir -p "$HF_HOME" && printf %s "hf_YOURTOKEN" > "$HF_HOME/token" && \
   python -c "from huggingface_hub import whoami; print(whoami()[\"name\"])"'
```

### 6. Launch JupyterLab (host port 8888, no token)
```bash
docker exec -d cosmos-trt bash -lc \
  'cd /opt/Model-Optimizer/examples/megatron_bridge/notebooks && \
   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
     --ServerApp.token="" --ServerApp.password=""'
```

### 7. Tunnel from your laptop and open the notebook
```bash
ssh -L 8888:localhost:8888 <you>@<thor-host>
# then browse to:
http://localhost:8888/lab/tree/07_cosmos_reason_thor_trtedge_llm.ipynb
```

---

## Running the notebook

| Section | What it does | Notes |
|---|---|---|
| §1 | Toolchain check (Edge-LLM binaries + `experimental.server` import + ModelOpt) | reads `EDGE_LLM_HOME` |
| §2 | Artifacts + `model_type==qwen3_vl` assert | the four checkpoints under `/hf` |
| §3 | NVFP4 theory + live FP8-vs-NVFP4 weight figure | **unchanged from nb06** |
| §4 | NVFP4 PTQ for original + distilled (`hf_ptq.py`) | **unchanged from nb06**; ~1 min each |
| §5 | export → `llm_build` + `visual_build` → in-process `LLM` caption | the multi-stage "compile" |
| §6 | Quality 4-way on **MMStar / MMMU** (native harness) | `prepare_dataset` needs the HF token (step 5) |
| §7a | Power telemetry — **Thor INA3221** cell | run before the §7 driver |
| §7 | Batch-1 robotics via **`llm_bench`** (prefill/decode sweeps) + power → 3 charts | no server |
| §8 | Wrap-up + the C++ / EAGLE-3 / FP8-KV production path | read |

---

## ✅ Confirm on your device

These were verified from the Edge-LLM source but depend on your exact SDK version / checkpoint — the
notebook guards each, but check them on first run:

1. **`config.json["model_type"] == "qwen3_vl"`** for the checkpoints (§2 asserts; true for stock
   Cosmos-Reason2-2B). Both the export auto-detect and the VLM runtime path gate on this string.
2. **`tensorrt-edgellm-export` accepts the ModelOpt NVFP4 checkpoint directly** (no re-quantize). Dry-run
   §5 on one variant first. If the console script isn't on `PATH`, use
   `python -m tensorrt_edgellm.scripts.export <ckpt> <onnx>`. The vision tower exports at **FP16** (there
   is no NVFP4 vision encoder — same as nb06's vLLM path).
3. **Engine filenames.** §5's `load_llm()` tries `LLM(engine_dir=…, visual_engine_dir=…)` and **falls back
   to `LLM(model=…)`** (which exports+builds its own engines) if the high-level API expects different engine
   filenames than `llm_build`/`visual_build` produce.
4. **`prepare_dataset.py` writes images as local `.jpg` files** referenced by path — the runtime loads
   images by file path only (no http/base64). The §5 demo and §6 showdown download images locally first.
5. **§6 sample-limit:** MMStar/MMMU have no native `--max_samples`, so the notebook truncates the request
   list in Python (`LIMIT`). Confirm `calculate_correctness.py` zips responses to requests positionally.
6. **`llm_bench` is a synthetic micro-benchmark** (fixed-shape prefill/decode, no live server) — absolute
   numbers differ from nb06's aiperf, but the 4-variant comparison is valid. Keep `--profile` OFF (it
   skips the `e2e_*.csv` the §7 charts parse).

---

## Troubleshooting

- **`experimental.server` import fails** — `BUILD_PYTHON_BINDINGS` was off, or `EDGE_LLM_HOME` isn't on
  `PYTHONPATH`. Rebuild with `-DBUILD_PYTHON_BINDINGS=ON`, or run
  `TRT_PACKAGE_DIR=/usr python experimental/server/setup_pybind.py build_ext --inplace`.
- **`tensorrt-edgellm-export: command not found`** — use `python -m tensorrt_edgellm.scripts.export`.
- **`visual_build` not found** — it lives under `build/examples/multimodal/`, not `build/examples/llm/`.
- **OOM building four engines** — engines are large; build/evaluate variants one at a time, or lower
  `--maxKVCacheCapacity`. `compile_variant()` skips variants whose engines already exist.
- **Harmless `UnicodeDecodeError` after a §4 quantize cell** — an interpreter-shutdown quirk in this torch
  build; it prints *after* `Quantized model exported to: …` and does not affect the checkpoint.
- **Restarting Jupyter** — `docker restart cosmos-trt`, then re-run step 6. Remove with
  `docker rm -f cosmos-trt`.
