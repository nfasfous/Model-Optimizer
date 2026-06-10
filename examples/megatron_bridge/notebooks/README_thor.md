# Running Notebook 06 on Thor AGX — Cosmos-Reason 2 Edge Compilation (NVFP4)

[`06_cosmos_reason_thor_nvfp4_edge.ipynb`](06_cosmos_reason_thor_nvfp4_edge.ipynb) takes the
pruned + distilled Cosmos-Reason 2 produced on the DGX in
[`05_cosmos_reason_end_to_end.ipynb`](05_cosmos_reason_end_to_end.ipynb) and turns it into an
edge-ready artifact on an **NVIDIA Thor AGX** (`aarch64` / Tegra / Blackwell, compute capability
11.0): quantize to **NVFP4** → compile + serve with vLLM → benchmark **quality** and **batch-1
robotics performance** (energy / reasoning-budget / responsiveness).

This guide gets the notebook running on a Thor. Everything runs inside a container built from
[`Dockerfile.thor`](Dockerfile.thor); the host only needs Docker with the NVIDIA runtime.

---

## Why a Thor-specific image (and not the NeMo container from notebook 05)

`nvcr.io/nvidia/nemo:26.04` publishes an `arm64` manifest, but it targets **SBSA / datacenter ARM**
(GH200-class), **not Tegra/Thor** — it will pull and then fail at CUDA init on the iGPU. We also
don't need NeMo here: prune + distill already happened on the DGX; on Thor we only
**quantize → compile → serve → benchmark**, which needs modelopt + vLLM + eval tooling.

`Dockerfile.thor` therefore builds on an **L4T/Thor vLLM engineering image** (ships the painful
stack prebuilt and GPU-verified: torch 2.13, vLLM 0.22.1, flash-attn, flashinfer, xformers, triton,
CUDA 13.3, transformers 5.6) and layers only thin, pure-Python additions: modelopt (from source),
`datasets`/`accelerate`, `lmms-eval`, `aiperf`, `matplotlib`, and JupyterLab.

---

## Prerequisites

- An NVIDIA Thor AGX with Docker + the NVIDIA container runtime.
- The base vLLM image present locally. Check and note the exact tag:
  ```bash
  docker images | grep core-models/vllm
  ```
  `Dockerfile.thor` defaults to `…/core-models/vllm:main_54155305`. If that tag is gone, pass a
  newer one with `--build-arg BASE_IMAGE=<tag>` in step 1.
- The model checkpoints on the host at `/home/scratch.nfasfous_wwfo/hf` (mounted to `/hf`):
  `Cosmos-Reason2-2B`, `Cosmos-Reason2-distilled`, `Cosmos-Reason2-fp8`,
  `Cosmos-Reason2-distilled-fp8` (the two NVFP4 variants are produced by §4 of the notebook).
- A HuggingFace token (read scope) for the §6 benchmark dataset downloads (BLINK / RealWorldQA).

> Adjust the two host paths below (`…/GitRepos/Model-Optimizer` and `…/hf`) if your checkout or
> model directory live elsewhere.

---

## Steps (run on the Thor)

### 1. Build the image (~5–10 min)
```bash
cd /home/scratch.nfasfous_wwfo/GitRepos/Model-Optimizer
docker build -f examples/megatron_bridge/notebooks/Dockerfile.thor \
  -t cosmos-thor:latest examples/megatron_bridge/notebooks
# If the default base tag is gone:
#   docker build --build-arg BASE_IMAGE=<tag-from-`docker images`> -f … -t cosmos-thor:latest …
```

### 2. Start the container
Tegra exposes the iGPU through the NVIDIA **runtime** (not `--gpus`). `--network host` puts Jupyter
and vLLM directly on the Thor's ports. The mounts make your **local** repo (notebook + any local
modelopt edits) and the model directory available inside.
```bash
docker run -d --name thorwork --runtime nvidia --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/scratch.nfasfous_wwfo/GitRepos/Model-Optimizer:/opt/Model-Optimizer \
  -v /home/scratch.nfasfous_wwfo/hf:/hf \
  cosmos-thor:latest sleep infinity
```

### 3. Point modelopt at the mounted local repo
Re-registers the editable install against your working tree so `modelopt` **and**
`modelopt_recipes` resolve to the mounted local source (not the image's build-time clone).
```bash
docker exec thorwork bash -lc \
  'rm -rf /usr/local/lib/python3.12/dist-packages/nvidia_modelopt-*.dist-info; \
   pip install -e /opt/Model-Optimizer --no-deps -q'
```

### 4. Add your HuggingFace token
```bash
docker exec thorwork bash -lc \
  'mkdir -p ~/.cache/huggingface && printf %s "hf_YOURTOKEN" > ~/.cache/huggingface/token'
```

### 5. Launch JupyterLab (host port 8888, no token)
```bash
docker exec -d thorwork bash -lc \
  'cd /opt/Model-Optimizer/examples/megatron_bridge/notebooks && \
   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
     --ServerApp.token="" --ServerApp.password=""'
```

### 6. Tunnel from your laptop and open the notebook
```bash
ssh -L 8888:localhost:8888 <you>@<thor-host>
```
Then browse to:
```
http://localhost:8888/lab/tree/06_cosmos_reason_thor_nvfp4_edge.ipynb
```

### Sanity check (optional)
```bash
docker exec thorwork bash -lc \
  'python -c "import modelopt.torch.quantization, lmms_eval, aiperf, matplotlib; print(\"env OK\")"; \
   nvidia-smi --query-gpu=name --format=csv,noheader'
```

---

## Running the notebook

Run top-to-bottom, or jump straight to a section — `/hf` already has the checkpoints, so §4's
quantize cells are quick re-runs.

| Section | What it does | Notes |
|---|---|---|
| §1–§3 | Toolchain, artifacts, NVFP4 best-practice layer selection | read + env check |
| §4 | NVFP4 PTQ for original + distilled (`hf_ptq.py`) | ~1 min each |
| §5 | Compile + serve with vLLM (`torch.compile` + CUDA graphs) | serve cmd in a terminal |
| §6 | Quality: judge-free BLINK + RealWorldQA, 4-way | needs the HF token (step 4) |
| §7 | Robotics batch-1: energy / reasoning-budget / responsiveness | **cell 20 driver → cell 22 plot**, ~10–12 min |
| §8 | Wrap-up + TensorRT-LLM reference | read |

§7 is self-contained: the driver cell serves each variant, samples the Thor INA3221 power rails via
`/sys/class/hwmon`, and runs the aiperf workloads; the plot cell renders the three charts inline.

---

## Troubleshooting

- **vLLM "Error in memory profiling … free memory changed"** — a shared-GPU artifact. The image
  already relaxes this assert, and the §7 driver picks a memory-adaptive `--gpu-memory-utilization`.
  If you still hit OOM on a busy box, lower the utilization in the serve command.
- **§6 `LocalTokenNotFoundError`** — the HF token (step 4) is missing or not in *this* container.
- **Base image tag not found at build** — list tags with `docker images | grep core-models/vllm`
  and pass the newest via `--build-arg BASE_IMAGE=<tag>`.
- **Harmless `UnicodeDecodeError` traceback after a quantize cell** — an interpreter-shutdown quirk in
  this Thor torch build; it prints *after* `Quantized model exported to: …` and does not affect the
  checkpoint.
- **Restarting Jupyter / clean slate** — `docker restart thorwork` (preserves the installed env),
  then re-run step 5. Remove entirely with `docker rm -f thorwork`.
