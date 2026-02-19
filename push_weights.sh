#!/usr/bin/env bash
# Push downloaded model weights to zenlm/ HuggingFace repos
# Run after LFS downloads complete
# Usage: ./push_weights.sh [model_dir]  or  ./push_weights.sh all

set -euo pipefail
ZEN4_DIR="$(cd "$(dirname "$0")" && pwd)"

declare -A TARGETS=(
  ["mini"]="zen4-mini"
  ["base"]="zen4"
  ["pro"]="zen4-pro"
  ["max"]="zen4-max"
  ["pro-max"]="zen4-pro-max"
  ["coder-flash"]="zen4-coder-flash"
  ["coder"]="zen4-coder"
)

push_model() {
  local dir="$1"
  local target="${TARGETS[$dir]}"
  local repo="${ZEN4_DIR}/${dir}/repo"

  # Check if any real safetensors exist (not just pointer files)
  local real_count=$(find "$repo" -maxdepth 1 -name "*.safetensors" -size +1k 2>/dev/null | wc -l | tr -d ' ')
  local total_count=$(ls "$repo"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')

  if [ "$real_count" -eq 0 ]; then
    echo "SKIP ${dir}: LFS not complete yet (${real_count}/${total_count} safetensors)"
    return 1
  fi

  echo ""
  echo "============================================================"
  echo "Pushing ${dir} -> zenlm/${target} (${real_count}/${total_count} safetensors)"
  echo "============================================================"

  cd "$repo"

  # Set remote to our repo
  git remote set-url origin "https://huggingface.co/zenlm/${target}"

  # Stage all files
  git add -A

  # Commit
  git commit -m "Add model weights for ${target}" 2>/dev/null || echo "(no changes)"

  # Push (force to overwrite our README-only initial commit)
  if git push --force origin HEAD:main 2>&1; then
    echo "  PUSHED: https://huggingface.co/zenlm/${target}"
    return 0
  elif git push --force origin HEAD:master 2>&1; then
    echo "  PUSHED: https://huggingface.co/zenlm/${target}"
    return 0
  else
    echo "  PUSH FAILED for ${target}"
    return 1
  fi
}

TARGET="${1:-all}"

if [ "$TARGET" = "all" ]; then
  for dir in "${!TARGETS[@]}"; do
    push_model "$dir" || true
  done
else
  push_model "$TARGET"
fi

echo ""
echo "Done. Check: https://huggingface.co/zenlm"
