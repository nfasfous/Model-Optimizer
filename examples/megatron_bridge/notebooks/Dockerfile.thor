# aarch64 / Blackwell edge image for the Cosmos-Reason 2 *edge compilation*
# notebook (06_cosmos_reason_thor_nvfp4_edge.ipynb).
#
# Builds the SAME image for BOTH edge targets:
#   * Thor AGX          — Tegra / Blackwell sm_110
#   * DGX Spark (GB10)  — Grace-Blackwell sm_121
# The base ships TORCH_CUDA_ARCH_LIST="... 12.0+PTX": sm_110 has native SASS and
# sm_121 is reached by forward JIT of the 12.0 PTX at load time (verified — torch,
# flash-attn and flashinfer all run on GB10). No Spark-specific build is needed.
# The only runtime difference is the GPU flag: `--runtime nvidia` on Thor vs
# `--gpus all` on Spark (see README_thor_spark.md).

# The internal core-models/vLLM image below is an L4T engineering build that
# ships the entire painful-to-build stack prebuilt and GPU-verified:
#   torch 2.13, vLLM 0.22.1, flash-attn, flashinfer, xformers, triton, CUDA 13.3,
#   transformers 5.6.  We only layer thin, pure-Python additions on top.
ARG BASE_IMAGE=gitlab-master.nvidia.com/dl/core-models/core-models/vllm:main_54155305
FROM ${BASE_IMAGE}

# ---------------------------------------------------------------------------
# Notebook + git-lfs
# ---------------------------------------------------------------------------
RUN pip install --root-user-action ignore \
        jupyterlab==4.4.10 \
        ipywidgets==8.1.7 && \
    jupyter labextension disable "@jupyterlab/apputils-extension:announcements"
ENV SHELL=/bin/bash

RUN apt-get update && \
    apt-get install -y --no-install-recommends git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Quieter student experience
ENV PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning,ignore::SyntaxWarning"

# ---------------------------------------------------------------------------
# Protect the prebuilt heavy stack.  The pure-Python tools below otherwise try
# to drag in incompatible pins (aiperf -> apache-tvm-ffi 0.1.7; lmms-eval ->
# antlr 4.7.2) that would break vLLM / omegaconf.  A pip constraint file pins
# the versions we validated on Thor so nothing downgrades them.
# ---------------------------------------------------------------------------
RUN cp /etc/pip/constraint.txt /etc/pip/thor-constraint.txt 2>/dev/null || touch /etc/pip/thor-constraint.txt && \
    printf '%s\n' \
        'transformers==5.6.0' \
        'apache-tvm-ffi==0.1.9' \
        >> /etc/pip/thor-constraint.txt
ENV PIP_CONSTRAINT=/etc/pip/thor-constraint.txt

# ---------------------------------------------------------------------------
# ModelOpt from source (branch lmikaelyan/compress-vlms).  --no-deps keeps pip
# from touching the container's torch build; we then add only the pure-Python
# runtime deps the HF-PTQ / NVFP4 export path actually imports.
# ---------------------------------------------------------------------------
ARG MODELOPT_BRANCH=lmikaelyan/compress-vlms
RUN git clone https://github.com/NVIDIA/Model-Optimizer.git /opt/Model-Optimizer && \
    cd /opt/Model-Optimizer && git checkout ${MODELOPT_BRANCH} && \
    rm -rf /usr/local/lib/python3.12/dist-packages/nvidia_modelopt-*.dist-info && \
    python -m pip install --root-user-action ignore -e . --no-deps && \
    python -m pip install --root-user-action ignore \
        omegaconf pydantic rich pulp cppimport scipy ml_dtypes \
        "nvidia-ml-py>=12" regex "accelerate>=1.0.0" "datasets>=3.0.0" \
        matplotlib   # for the robotics benchmark charts (§7)

# ---------------------------------------------------------------------------
# lmms-eval (VLM quality benchmarks: BLINK / RealWorldQA)
# ---------------------------------------------------------------------------
RUN git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git /opt/lmms-eval && \
    cd /opt/lmms-eval && git checkout v0.7.1 && \
    python -m pip install --root-user-action ignore -e . && \
    python -m pip install --root-user-action ignore python-Levenshtein

# ---------------------------------------------------------------------------
# aiperf (inference perf: TTFT / ITL / throughput)
# ---------------------------------------------------------------------------
RUN python -m pip install --root-user-action ignore aiperf

# ---------------------------------------------------------------------------
# Final pin: lmms-eval's latex2sympy2 hard-requests antlr 4.7.2, but omegaconf
# (used by modelopt config loading) needs 4.9.*.  4.9.3 satisfies omegaconf and
# latex2sympy2 still parses correctly (only emits a harmless version-disagree
# warning), so we force it last and let it win.
# ---------------------------------------------------------------------------
RUN python -m pip install --root-user-action ignore --no-deps "antlr4-python3-runtime==4.9.3"

# ---------------------------------------------------------------------------
# Shared-GPU robustness: vLLM's startup memory profiler asserts that free GPU
# memory does not change during profiling. On a *shared* Thor (other jobs
# allocating/freeing concurrently) this trips intermittently. Relax it with an
# 80 GiB slack so vLLM starts reliably. On a dedicated Thor it never triggers,
# so this is a no-op there. (The notebook also picks a memory-adaptive util.)
# ---------------------------------------------------------------------------
RUN F=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py && \
    sed -i 's/assert self.init_snapshot.free_memory >= free_gpu_memory, (/assert self.init_snapshot.free_memory >= free_gpu_memory - (80<<30), (  # shared-GPU slack/' "$F" && \
    grep -q 'shared-GPU slack' "$F"

WORKDIR /opt/Model-Optimizer/examples/megatron_bridge/notebooks
