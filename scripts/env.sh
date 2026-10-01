# Common environment for every job and smoke test. Source it: `source scripts/env.sh`.
# Everything (venv, Hugging Face cache, data, outputs) lives under the project directory.
export PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PY="$PROJECT/.venv/bin/python"
export HF_HOME="$PROJECT/cache/hf_cache"
export HF_HUB_OFFLINE=1            # the checkpoint is pre-downloaded; compute nodes stay offline
export TRANSFORMERS_OFFLINE=1
export USE_TF=0                    # laya model card: avoids a TensorFlow-probe deadlock at load
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "$PROJECT"

print_env() {
  echo "host: $(hostname)   date: $(date)   job: ${SLURM_JOB_ID:-none}"
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
  "$PY" - <<'EOF'
import sys, torch, transformers, laya, rdflib, numpy, pandas, rapidfuzz, huggingface_hub
print("python", sys.version.split()[0], "| torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available(), "| transformers", transformers.__version__,
      "| laya", laya.__version__, "| rdflib", rdflib.__version__, "| numpy", numpy.__version__,
      "| pandas", pandas.__version__, "| rapidfuzz", rapidfuzz.__version__,
      "| huggingface_hub", huggingface_hub.__version__)
EOF
  echo "git: $(git -C "$PROJECT" rev-parse --short HEAD 2>/dev/null || echo 'no commit')"
}
