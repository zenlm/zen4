#!/usr/bin/env bash
# Watch for LFS downloads to complete and run pipeline for each model
# Usage: ./watch_and_run.sh

set -uo pipefail
ZEN4_DIR="$(cd "$(dirname "$0")" && pwd)"

declare -A MODELS=(
  ["mini"]="zen4-mini|Zen4 Mini"
  ["base"]="zen4|Zen4"
  ["pro"]="zen4-pro|Zen4 Pro"
  ["max"]="zen4-max|Zen4 Max"
  ["pro-max"]="zen4-pro-max|Zen4 Pro Max"
  ["coder-flash"]="zen4-coder-flash|Zen4 Coder Flash"
  ["coder"]="zen4-coder|Zen4 Coder"
)

declare -A PROCESSED=()

while true; do
  all_done=true

  for dir in mini base pro max pro-max coder-flash coder; do
    # Skip already processed
    if [ "${PROCESSED[$dir]:-}" = "1" ]; then
      continue
    fi

    all_done=false
    repo="${ZEN4_DIR}/${dir}/repo"

    # Check if safetensors are real files (not LFS pointers)
    pointer_count=$(find "$repo" -maxdepth 1 -name "*.safetensors" -size -1k 2>/dev/null | wc -l | tr -d ' ')
    total_count=$(ls "$repo"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
    ready=$((total_count - pointer_count))

    if [ "$pointer_count" -eq 0 ] && [ "$total_count" -gt 0 ]; then
      IFS='|' read -r zen4name display <<< "${MODELS[$dir]}"
      echo ""
      echo "$(date): ${display} LFS COMPLETE! (${ready}/${total_count} safetensors)"
      echo "Starting pipeline for ${display}..."

      "${ZEN4_DIR}/pipeline.sh" "$(echo "$dir" | sed 's/pro-max/maxpro/' | sed 's/coder-flash/flash/')" 2>&1 | tee "${ZEN4_DIR}/${dir}/pipeline.log"

      PROCESSED[$dir]=1
      echo "$(date): ${display} pipeline DONE"
    else
      size=$(du -sh "$repo" 2>/dev/null | cut -f1)
      echo "$(date): ${dir}: ${size} (${ready}/${total_count} safetensors ready)"
    fi
  done

  if [ "$all_done" = true ]; then
    echo ""
    echo "All models processed!"
    break
  fi

  # Check every 60 seconds
  sleep 60
done
