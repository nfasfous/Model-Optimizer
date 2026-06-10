# Running Notebook 06 on the Edge — Cosmos-Reason 2 Compilation (NVFP4) on Thor AGX **or** DGX Spark

[`06_cosmos_reason_thor_nvfp4_edge.ipynb`](06_cosmos_reason_thor_nvfp4_edge.ipynb) takes the
pruned + distilled Cosmos-Reason 2 produced on the DGX in
[`05_cosmos_reason_end_to_end.ipynb`](05_cosmos_reason_end_to_end.ipynb) and turns it into an
edge-ready artifact: quantize to **NVFP4** → compile + serve with vLLM → benchmark **quality** and
**batch-1 robotics performance** (energy / reasoning-budget / responsiveness).

It runs on **two aarch64 Blackwell edge targets** from a single container image:

| Platform | GPU | Compute capability | Power telemetry |
|---|---|---|---|
| **NVIDIA Thor AGX** | Tegra iGPU | `sm_110` (11.0) | INA3221 rails via `/sys/class/hwmon` |
| **NVIDIA DGX Spark** | GB10 Grace-Blackwell | `sm_121` (12.1) | NVML (`nvmlDeviceGetPowerUsage`) |

Everything runs inside a container built from [`Dockerfile.thor`](Dockerfile.thor); the host only
needs Docker with the NVIDIA container toolkit. **The image is identical for both platforms** — see
[Why one image covers both](#why-one-image-covers-both-thor--spark). There are exactly **two**
cross-platform differences, both called out below:

1. **GPU flag** — `--runtime nvidia` on Thor, **`--gpus all`** on Spark.
2. **§7 power** — INA3221 sysfs on Thor, **NVML** on Spark. The notebook has a dedicated platform
   cell for each (§7a); run the one for your hardware.

---

## Why a Blackwell-specific image (and not the NeMo container from notebook 05)

`nvcr.io/nvidia/nemo:26.04` publishes an `arm64` manifest, but it targets **SBSA / datacenter ARM**
(GH200-class), **not Tegra/Thor or GB10** — it will pull and then fail at CUDA init on the iGPU. We
also don't need NeMo here: prune + distill already happened on the DGX; on the edge we only
**quantize → compile → serve → benchmark**, which needs modelopt + vLLM + eval tooling.

[`Dockerfile.thor`](Dockerfile.thor) therefore builds on an **L4T vLLM engineering image** (ships the
painful stack prebuilt and GPU-verified: torch 2.13, vLLM 0.22.1, flash-attn, flashinfer, xformers,
triton, CUDA 13.3, transformers 5.6) and layers only thin, pure-Python additions: modelopt (from
source), `datasets`/`accelerate`, `lmms-eval`, `aiperf`, `matplotlib`, and JupyterLab.

### Why one image covers both Thor & Spark

The base ships `TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0 10.0 11.0 12.0+PTX"`. Thor's `sm_110` has native
compiled SASS; Spark's `sm_121` has no dedicated cubin, but the embedded **`12.0` PTX
forward-JIT-compiles to `sm_121`** at load time (CUDA forward-compatibility). This was verified
end-to-end on a GB10: torch matmul, a **flash-attn** forward, **flashinfer** import, and a full
**vLLM serve + chat completion** all run. So no Spark-specific build is required — the same
`docker build` and the same base tag work for both.

---

## Prerequisites

- An NVIDIA **Thor AGX** *or* **DGX Spark** with Docker + the NVIDIA container toolkit.
- The base vLLM image present locally. Check and note the exact tag:
  ```bash
  docker images | grep core-models/vllm
  ```
  `Dockerfile.thor` defaults to `…/core-models/vllm:main_54155305`. If that tag is gone, pass a
  newer one with `--build-arg BASE_IMAGE=<tag>` in step 1. (All recent tags carry the same
  `…12.0+PTX` arch list, so any of them covers both Thor and Spark.)
- The model checkpoints on the host, mounted to `/hf`:
  `Cosmos-Reason2-2B`, `Cosmos-Reason2-distilled`, `Cosmos-Reason2-fp8`,
  `Cosmos-Reason2-distilled-fp8` (the two NVFP4 variants are produced by §4 of the notebook).
- A HuggingFace token (read scope) for the §6 benchmark dataset downloads (BLINK / RealWorldQA).

> Adjust the two host paths below (the repo checkout and the `…/hf` model directory) to wherever
> they live on your machine.

---

## Steps

### 1. Build the image (~5–10 min)
```bash
cd <your-repo-root>/Model-Optimizer
docker build -f examples/megatron_bridge/notebooks/Dockerfile.thor \
  -t cosmos-edge:latest examples/megatron_bridge/notebooks
# If the default base tag is gone:
#   docker build --build-arg BASE_IMAGE=<tag-from-`docker images`> -f … -t cosmos-edge:latest …
```

### 2. Start the container

> ⚠️ **This is the GPU-flag delta.** Pick the line for your platform.

**Thor AGX** — Tegra exposes the iGPU through the NVIDIA **runtime**:
```bash
docker run -d --name cosmos --runtime nvidia --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v <repo>/Model-Optimizer:/opt/Model-Optimizer \
  -v <host>/hf:/hf \
  cosmos-edge:latest sleep infinity
```

**DGX Spark (GB10)** — use **`--gpus all`** (the container toolkit is installed, but no named
`nvidia` runtime is registered):
```bash
docker run -d --name cosmos --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v <repo>/Model-Optimizer:/opt/Model-Optimizer \
  -v <host>/hf:/hf \
  cosmos-edge:latest sleep infinity
```
*(On Spark you can alternatively register the runtime once with
`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`, after which
`--runtime nvidia` also works — but `--gpus all` needs no host change.)*

### 3. Point modelopt at the mounted local repo
Re-registers the editable install against your working tree so `modelopt` **and**
`modelopt_recipes` resolve to the mounted local source (not the image's build-time clone).
```bash
docker exec cosmos bash -lc \
  'rm -rf /usr/local/lib/python3.12/dist-packages/nvidia_modelopt-*.dist-info; \
   pip install -e /opt/Model-Optimizer --no-deps -q'
```

### 4. Add your HuggingFace token
```bash
docker exec cosmos bash -lc \
  'mkdir -p ~/.cache/huggingface && printf %s "hf_YOURTOKEN" > ~/.cache/huggingface/token'
```

### 5. Launch JupyterLab (host port 8888, no token)
```bash
docker exec -d cosmos bash -lc \
  'cd /opt/Model-Optimizer/examples/megatron_bridge/notebooks && \
   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
     --ServerApp.token="" --ServerApp.password=""'
```

### 6. Tunnel from your laptop and open the notebook
```bash
ssh -L 8888:localhost:8888 <you>@<edge-host>
```
Then browse to:
```
http://localhost:8888/lab/tree/06_cosmos_reason_thor_nvfp4_edge.ipynb
```

### Sanity check (optional)
```bash
docker exec cosmos bash -lc \
  'python -c "import modelopt.torch.quantization, lmms_eval, aiperf, matplotlib; print(\"env OK\")"; \
   nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader'
```
Thor reports `NVIDIA Thor, 11.0`; Spark reports `NVIDIA GB10, 12.1`.

---

## One-liner (DGX Spark): build, spin up, and serve the notebook

From the repo root — builds, starts the container with `--gpus all`, registers the editable
modelopt, and launches JupyterLab in one shot:

```bash
docker build -f examples/megatron_bridge/notebooks/Dockerfile.thor -t cosmos-edge:latest examples/megatron_bridge/notebooks && \
docker run -d --name cosmos --gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/opt/Model-Optimizer -v /home/nfasfous/GitRepos/hf:/hf cosmos-edge:latest \
  bash -lc 'rm -rf /usr/local/lib/python3.12/dist-packages/nvidia_modelopt-*.dist-info; \
            pip install -e /opt/Model-Optimizer --no-deps -q; \
            cd /opt/Model-Optimizer/examples/megatron_bridge/notebooks && \
            jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token="" --ServerApp.password=""'
```
Then open `http://localhost:8888/lab/tree/06_cosmos_reason_thor_nvfp4_edge.ipynb` (add your HF token
first — step 4 — for §6). For Thor, swap `--gpus all` → `--runtime nvidia`.

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
| §7a | **Power telemetry — pick your platform cell** | **Thor INA3221 OR Spark NVML** |
| §7 | Robotics batch-1: energy / reasoning-budget / responsiveness | driver → plot, ~10–12 min |
| §8 | Wrap-up + TensorRT-LLM reference | read |

### §7 power telemetry — the one notebook delta

§7's **energy per decision** needs a live power reading, and the two platforms expose power
completely differently. §7a has **two clearly-headed cells** — run exactly the one for your
hardware before the driver:

- 🟠 **THOR AGX cell** — reads the INA3221 rails from `/sys/class/hwmon/hwmon4`
  (`VDD_GPU + CPU_SOC + VIN_SYS`) → **total-board** Joules.
- 🟢 **DGX SPARK cell** — GB10 does **not** expose those rails (only `acpitz`/`nvme`/wifi on
  `/sys/class/hwmon`); it reads **GPU-module** power via NVML → **GPU** Joules.

Both define an identical `PowerSampler` interface, so the driver and plot cells are the same on both
platforms; the plot labels which scope is in play. **Don't compare a Thor *board* number against a
Spark *GPU* number directly** — different scopes.

---

## Troubleshooting

- **`docker: unknown or invalid runtime name: nvidia` (Spark)** — use `--gpus all` (step 2), or
  register the runtime with `nvidia-ctk runtime configure --runtime=docker`.
- **`no kernel image is available for execution on the device`** — would mean a prebuilt kernel lib
  lacks both SASS for your arch and a JIT-able PTX. Not observed on GB10 with `main_54155305`; if it
  appears on a newer GPU, try a newer base tag via `--build-arg BASE_IMAGE=`.
- **First vLLM start is slow on Spark** — expected: the `12.0` PTX JIT-compiles to `sm_121` on first
  launch. Set a persistent `CUDA_CACHE_PATH` to keep the JIT cache across runs.
- **vLLM "Error in memory profiling … free memory changed"** — a shared-GPU artifact. The image
  already relaxes this assert, and the §7 driver picks a memory-adaptive `--gpu-memory-utilization`.
  If you still hit OOM on a busy box, lower the utilization in the serve command.
- **§6 `LocalTokenNotFoundError`** — the HF token (step 4) is missing or not in *this* container.
- **Base image tag not found at build** — list tags with `docker images | grep core-models/vllm`
  and pass the newest via `--build-arg BASE_IMAGE=<tag>`.
- **Harmless `UnicodeDecodeError` traceback after a quantize cell** — an interpreter-shutdown quirk in
  this torch build; it prints *after* `Quantized model exported to: …` and does not affect the
  checkpoint.
- **Restarting Jupyter / clean slate** — `docker restart cosmos` (preserves the installed env),
  then re-run step 5. Remove entirely with `docker rm -f cosmos`.
