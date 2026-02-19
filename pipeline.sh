#!/usr/bin/env bash
# Zen4 Full Pipeline: Push abliterated base -> Identity train -> Convert
# Usage: ./pipeline.sh [model_key]  (e.g., mini, base, pro, max, maxpro, flash, coder)
# Or:    ./pipeline.sh all

set -euo pipefail
ZEN4_DIR="$(cd "$(dirname "$0")" && pwd)"
ZEN_DIR="$(dirname "$ZEN4_DIR")"

# Model definitions: key|dir|zen4_name|display|params
declare -a MODELS=(
  "mini|mini|zen4-mini|Zen4 Mini|4B"
  "base|base|zen4|Zen4|8B"
  "pro|pro|zen4-pro|Zen4 Pro|14B"
  "max|max|zen4-max|Zen4 Max|30B MoE"
  "maxpro|pro-max|zen4-pro-max|Zen4 Pro Max|80B MoE"
  "flash|coder-flash|zen4-coder-flash|Zen4 Coder Flash|31B MoE"
  "coder|coder|zen4-coder|Zen4 Coder|80B MoE"
)

check_lfs_complete() {
  local repo_dir="$1"
  # Check if any safetensors are still LFS pointers (< 1KB)
  local pointer_count=$(find "$repo_dir" -maxdepth 1 -name "*.safetensors" -size -1k 2>/dev/null | wc -l | tr -d ' ')
  local total_count=$(ls "$repo_dir"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
  if [ "$pointer_count" -eq 0 ] && [ "$total_count" -gt 0 ]; then
    return 0  # All complete
  fi
  return 1  # Still downloading
}

push_to_hf() {
  local dir="$1" zen4name="$2" display="$3"
  local repo_dir="${ZEN4_DIR}/${dir}/repo"

  echo "  Pushing ${display} to zenlm/${zen4name} via hf upload..."

  # Create repo if needed
  hf repo create "zenlm/${zen4name}" --type model 2>/dev/null || true

  # Upload all files (handles LFS objects correctly)
  if hf upload "zenlm/${zen4name}" "$repo_dir" --repo-type model 2>&1; then
    echo "  PUSHED: https://huggingface.co/zenlm/${zen4name}"
    return 0
  else
    echo "  PUSH FAILED - retrying..."
    hf upload "zenlm/${zen4name}" "$repo_dir" --repo-type model 2>&1 || return 1
  fi
}

train_identity() {
  local key="$1" display="$2"
  echo "  Training identity for ${display}..."
  cd "$ZEN_DIR"
  python train_identity.py --model "$key" 2>&1
}

convert_formats() {
  local dir="$1" zen4name="$2" display="$3"
  local repo_dir="${ZEN4_DIR}/${dir}/repo"
  local adapter_dir="${ZEN4_DIR}/${dir}/training/output/adapters"

  echo "  Converting ${display} to GGUF + MLX..."

  # Check if adapter exists
  if [ ! -d "$adapter_dir" ] || [ ! -f "$adapter_dir/adapters.safetensors" ]; then
    echo "  SKIP conversion: no adapters found (training not complete?)"
    return 1
  fi

  # Fuse adapter into base model
  local fused_dir="${ZEN4_DIR}/${dir}/fused"
  echo "  Fusing LoRA adapter into base model..."
  python -m mlx_lm.fuse \
    --model "$repo_dir" \
    --adapter-path "$adapter_dir" \
    --save-path "$fused_dir" 2>&1

  # Convert to GGUF (requires llama.cpp)
  local gguf_dir="${ZEN4_DIR}/${dir}/gguf"
  mkdir -p "$gguf_dir"

  local LLAMA_DIR="${ZEN_DIR}/llama.cpp"
  if command -v python3 &>/dev/null && [ -f "${LLAMA_DIR}/convert_hf_to_gguf.py" ]; then
    echo "  Converting to GGUF..."
    python3 "${LLAMA_DIR}/convert_hf_to_gguf.py" \
      "$fused_dir" \
      --outfile "${gguf_dir}/${zen4name}.gguf" \
      --outtype f16 2>&1

    # Quantize to Q4_K_M
    if [ -x "${LLAMA_DIR}/build/bin/llama-quantize" ]; then
      echo "  Quantizing to Q4_K_M..."
      "${LLAMA_DIR}/build/bin/llama-quantize" \
        "${gguf_dir}/${zen4name}.gguf" \
        "${gguf_dir}/${zen4name}-Q4_K_M.gguf" \
        Q4_K_M 2>&1
    else
      echo "  SKIP quantize: build llama.cpp first (cmake -B build && cmake --build build)"
    fi
  else
    echo "  SKIP GGUF: llama.cpp not found at ${LLAMA_DIR}"
  fi

  # Convert to MLX format
  local mlx_dir="${ZEN4_DIR}/${dir}/mlx"
  echo "  Converting to MLX 4-bit..."
  python -m mlx_lm.convert \
    --hf-path "$fused_dir" \
    --mlx-path "$mlx_dir" \
    --quantize \
    --q-bits 4 2>&1

  echo "  Conversion complete for ${display}"
}

upload_final() {
  local dir="$1" zen4name="$2" display="$3"
  local fused_dir="${ZEN4_DIR}/${dir}/fused"
  local gguf_dir="${ZEN4_DIR}/${dir}/gguf"
  local mlx_dir="${ZEN4_DIR}/${dir}/mlx"

  echo "  Uploading final ${display} to HuggingFace..."

  # Upload fused model to main repo
  if [ -d "$fused_dir" ]; then
    hf upload "zenlm/${zen4name}" "$fused_dir" --repo-type model 2>&1 || echo "  (fused upload issue)"
  fi

  # Upload GGUF to separate repo
  if [ -d "$gguf_dir" ] && ls "$gguf_dir"/*.gguf &>/dev/null; then
    hf repo create "zenlm/${zen4name}-GGUF" --repo-type model 2>/dev/null || true
    hf upload "zenlm/${zen4name}-GGUF" "$gguf_dir" --repo-type model 2>&1 || echo "  (gguf upload issue)"
    echo "  GGUF: https://huggingface.co/zenlm/${zen4name}-GGUF"
  fi

  # Upload MLX to separate repo
  if [ -d "$mlx_dir" ]; then
    hf repo create "zenlm/${zen4name}-MLX" --repo-type model 2>/dev/null || true
    hf upload "zenlm/${zen4name}-MLX" "$mlx_dir" --repo-type model 2>&1 || echo "  (mlx upload issue)"
    echo "  MLX: https://huggingface.co/zenlm/${zen4name}-MLX"
  fi
}

process_model() {
  local entry="$1"
  IFS='|' read -r key dir zen4name display params <<< "$entry"
  local repo_dir="${ZEN4_DIR}/${dir}/repo"

  echo ""
  echo "============================================================"
  echo "Pipeline: ${display} (${params})"
  echo "============================================================"

  # Step 1: Check if LFS download is complete
  if ! check_lfs_complete "$repo_dir"; then
    echo "  WAITING: LFS download not complete yet"
    local pointer_count=$(find "$repo_dir" -maxdepth 1 -name "*.safetensors" -size -1k 2>/dev/null | wc -l | tr -d ' ')
    local total_count=$(ls "$repo_dir"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
    local ready=$((total_count - pointer_count))
    echo "  Progress: ${ready}/${total_count} safetensors downloaded"
    return 1
  fi

  echo "  [1/5] LFS download complete"

  # Step 2: Push abliterated base to HF
  echo "  [2/5] Pushing abliterated base to HuggingFace..."
  push_to_hf "$dir" "$zen4name" "$display"

  # Step 3: Identity training
  echo "  [3/5] Identity training..."
  train_identity "$key" "$display"

  # Step 4: Convert to GGUF + MLX
  echo "  [4/5] Converting formats..."
  convert_formats "$dir" "$zen4name" "$display"

  # Step 5: Upload final models
  echo "  [5/5] Uploading final models..."
  upload_final "$dir" "$zen4name" "$display"

  echo ""
  echo "  COMPLETE: ${display}"
  echo "  HF: https://huggingface.co/zenlm/${zen4name}"
}

# Parse args
TARGET="${1:-all}"

echo "Zen4 Pipeline - $(date)"
echo "Target: ${TARGET}"

for entry in "${MODELS[@]}"; do
  key=$(echo "$entry" | cut -d'|' -f1)
  if [ "$TARGET" = "all" ] || [ "$TARGET" = "$key" ]; then
    process_model "$entry" || true
  fi
done

echo ""
echo "============================================================"
echo "Pipeline complete. Check: https://huggingface.co/zenlm"
echo "============================================================"
