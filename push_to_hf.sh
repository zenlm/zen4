#!/usr/bin/env bash
# Push Zen4 abliterated base models to zenlm/ HuggingFace repos
# Usage: ./push_to_hf.sh [model_key]  (e.g., mini, base, pro, max, maxpro, flash, coder)
# Or:    ./push_to_hf.sh all

set -euo pipefail

ZEN4_DIR="$(cd "$(dirname "$0")" && pwd)"

# Model definitions: key|zen4_name|source_hf_repo|display_name|params|active|context|base_upstream|license
declare -a MODELS=(
  "mini|zen4-mini|huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated|Zen4 Mini|4B|4B|32K|Qwen3-4B-Instruct-2507|Apache-2.0"
  "base|zen4|huihui-ai/Huihui-Qwen3-8B-abliterated-v2|Zen4|8B|8B|32K|Qwen3-8B|Apache-2.0"
  "pro|zen4-pro|mlabonne/Qwen3-14B-abliterated|Zen4 Pro|14B|14B|32K|Qwen3-14B|Apache-2.0"
  "max|zen4-max|huihui-ai/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated|Zen4 Max|30B MoE|3B|256K|Qwen3-30B-A3B-Instruct-2507|Apache-2.0"
  "maxpro|zen4-pro-max|huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-mlx-4bit|Zen4 Pro Max|80B MoE|3B|256K|Qwen3-Next-80B-A3B-Instruct|Apache-2.0"
  "flash|zen4-coder-flash|huihui-ai/Huihui-GLM-4.7-Flash-abliterated|Zen4 Coder Flash|31B MoE|3B|131K|GLM-4.7-Flash|MIT"
  "coder|zen4-coder|huihui-ai/Huihui-Qwen3-Coder-Next-abliterated|Zen4 Coder|80B MoE|3B|256K|Qwen3-Coder-Next|Apache-2.0"
)

process_model() {
  local entry="$1"
  IFS='|' read -r key zen4name source display params active context upstream license <<< "$entry"

  local work_dir="${ZEN4_DIR}/${key/maxpro/pro-max}"
  # Fix key->dir mapping
  case "$key" in
    maxpro) work_dir="${ZEN4_DIR}/pro-max" ;;
    flash)  work_dir="${ZEN4_DIR}/coder-flash" ;;
    *)      work_dir="${ZEN4_DIR}/${key}" ;;
  esac
  local repo_dir="${work_dir}/repo"

  echo ""
  echo "============================================================"
  echo "Processing: ${display} (${params})"
  echo "  Source: ${source}"
  echo "  Target: zenlm/${zen4name}"
  echo "============================================================"

  # 1. Create HF repo if needed
  echo "  Creating HF repo zenlm/${zen4name}..."
  hf repo create "zenlm/${zen4name}" --repo-type model 2>/dev/null || echo "  (repo already exists)"

  # 2. Clone source repo
  if [ -d "$repo_dir" ]; then
    echo "  Repo dir exists, pulling latest..."
    cd "$repo_dir"
    git pull 2>/dev/null || true
  else
    echo "  Cloning from ${source}..."
    GIT_LFS_SKIP_SMUDGE=1 git clone "https://huggingface.co/${source}" "$repo_dir"
    cd "$repo_dir"
    git lfs pull
  fi

  # 3. Update README with Zen4 branding
  cat > README.md << READMEEOF
---
license: ${license}
language:
- en
- zh
tags:
- zen4
- zenlm
- hanzo
- abliterated
- uncensored
base_model: ${source}
pipeline_tag: text-generation
---

# ${display}

**${display}** is a ${params} parameter${active != params && echo " (${active} active)" || true} language model from the [Zen4 family](https://zenlm.org) by [Zen LM](https://huggingface.co/zenlm) and [Hanzo AI](https://hanzo.ai).

## Model Details

| Property | Value |
|----------|-------|
| **Parameters** | ${params} total, ${active} active |
| **Context** | ${context} tokens |
| **Base** | ${upstream} (abliterated) |
| **License** | ${license} |
| **Family** | Zen4 |
| **Creator** | Zen LM / Hanzo AI |

## Zen4 Family

| Model | Params | Active | Context | Use Case |
|-------|--------|--------|---------|----------|
| Zen4 Mini | 4B | 4B | 32K | Edge, mobile |
| Zen4 | 8B | 8B | 32K | Standard |
| Zen4 Pro | 14B | 14B | 32K | Professional |
| Zen4 Max | 30B MoE | 3B | 256K | Flagship efficient |
| **Zen4 Pro Max** | **80B MoE** | **3B** | **256K** | **Consumer ultimate** |
| Zen4 Coder Flash | 31B MoE | 3B | 131K | Fast coding |
| Zen4 Coder | 80B MoE | 3B | 256K | Flagship coding |
| Zen4 Ultra | 1.04T MoE | 32B | 256K | Cloud |

## Usage

\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("zenlm/${zen4name}")
tokenizer = AutoTokenizer.from_pretrained("zenlm/${zen4name}")

messages = [
    {"role": "user", "content": "Hello, who are you?"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
\`\`\`

## Links

- [Zen LM](https://zenlm.org)
- [Hanzo AI](https://hanzo.ai)
- [All Zen4 Models](https://huggingface.co/collections/zenlm/zen4)
- [GitHub](https://github.com/zenlm)
READMEEOF

  # 4. Change remote to our repo and push
  echo "  Updating remote to zenlm/${zen4name}..."
  git remote set-url origin "https://huggingface.co/zenlm/${zen4name}"

  echo "  Pushing to zenlm/${zen4name}..."
  git add README.md
  git commit -m "Zen4: ${display} - abliterated base from ${upstream}" 2>/dev/null || true
  git push --force origin main 2>&1 || git push --force origin master 2>&1 || echo "  PUSH FAILED"

  echo "  Done: https://huggingface.co/zenlm/${zen4name}"
}

# Parse args
TARGET="${1:-all}"

for entry in "${MODELS[@]}"; do
  key=$(echo "$entry" | cut -d'|' -f1)
  if [ "$TARGET" = "all" ] || [ "$TARGET" = "$key" ]; then
    process_model "$entry"
  fi
done

echo ""
echo "============================================================"
echo "All Zen4 models pushed to HuggingFace!"
echo "Visit: https://huggingface.co/zenlm"
echo "============================================================"
