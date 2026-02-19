#!/usr/bin/env python3
"""Rebrand cloned abliterated models and push to zenlm/ HuggingFace repos."""

import os
import subprocess
import sys
from pathlib import Path

ZEN4_DIR = Path(__file__).parent

MODELS = [
    {
        "key": "mini",
        "dir": "mini",
        "zen4_name": "zen4-mini",
        "display": "Zen4 Mini",
        "params": "4B",
        "active": "4B",
        "context": "32K",
        "upstream": "Qwen3-4B-Instruct-2507",
        "source": "huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated",
        "license": "Apache-2.0",
    },
    {
        "key": "base",
        "dir": "base",
        "zen4_name": "zen4",
        "display": "Zen4",
        "params": "8B",
        "active": "8B",
        "context": "32K",
        "upstream": "Qwen3-8B",
        "source": "huihui-ai/Huihui-Qwen3-8B-abliterated-v2",
        "license": "Apache-2.0",
    },
    {
        "key": "pro",
        "dir": "pro",
        "zen4_name": "zen4-pro",
        "display": "Zen4 Pro",
        "params": "14B",
        "active": "14B",
        "context": "32K",
        "upstream": "Qwen3-14B",
        "source": "mlabonne/Qwen3-14B-abliterated",
        "license": "Apache-2.0",
    },
    {
        "key": "max",
        "dir": "max",
        "zen4_name": "zen4-max",
        "display": "Zen4 Max",
        "params": "30B MoE",
        "active": "3B",
        "context": "256K",
        "upstream": "Qwen3-30B-A3B-Instruct-2507",
        "source": "huihui-ai/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated",
        "license": "Apache-2.0",
    },
    {
        "key": "maxpro",
        "dir": "pro-max",
        "zen4_name": "zen4-pro-max",
        "display": "Zen4 Pro Max",
        "params": "80B MoE",
        "active": "3B",
        "context": "256K",
        "upstream": "Qwen3-Next-80B-A3B-Instruct",
        "source": "huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-mlx-4bit",
        "license": "Apache-2.0",
    },
    {
        "key": "flash",
        "dir": "coder-flash",
        "zen4_name": "zen4-coder-flash",
        "display": "Zen4 Coder Flash",
        "params": "31B MoE",
        "active": "3B",
        "context": "131K",
        "upstream": "GLM-4.7-Flash",
        "source": "huihui-ai/Huihui-GLM-4.7-Flash-abliterated",
        "license": "MIT",
    },
    {
        "key": "coder",
        "dir": "coder",
        "zen4_name": "zen4-coder",
        "display": "Zen4 Coder",
        "params": "80B MoE",
        "active": "3B",
        "context": "256K",
        "upstream": "Qwen3-Coder-Next",
        "source": "huihui-ai/Huihui-Qwen3-Coder-Next-abliterated",
        "license": "Apache-2.0",
    },
]


def generate_readme(m):
    active_note = f" ({m['active']} active)" if m['active'] != m['params'] else ""
    return f"""---
license: {m['license']}
language:
- en
- zh
tags:
- zen4
- zenlm
- hanzo
- abliterated
- uncensored
base_model: {m['source']}
pipeline_tag: text-generation
---

# {m['display']}

**{m['display']}** is a {m['params']}{active_note} parameter language model from the [Zen4 family](https://zenlm.org) by [Zen LM](https://huggingface.co/zenlm) and [Hanzo AI](https://hanzo.ai).

Built on abliterated (uncensored) weights from {m['upstream']} for unrestricted, open-ended AI assistance.

## Model Details

| Property | Value |
|----------|-------|
| **Parameters** | {m['params']} total, {m['active']} active |
| **Context** | {m['context']} tokens |
| **Base** | {m['upstream']} (abliterated) |
| **License** | {m['license']} |
| **Family** | Zen4 |
| **Creator** | Zen LM / Hanzo AI |

## Zen4 Family

| Model | Params | Active | Context | HuggingFace |
|-------|--------|--------|---------|-------------|
| Zen4 Mini | 4B | 4B | 32K | [zenlm/zen4-mini](https://huggingface.co/zenlm/zen4-mini) |
| Zen4 | 8B | 8B | 32K | [zenlm/zen4](https://huggingface.co/zenlm/zen4) |
| Zen4 Pro | 14B | 14B | 32K | [zenlm/zen4-pro](https://huggingface.co/zenlm/zen4-pro) |
| Zen4 Max | 30B MoE | 3B | 256K | [zenlm/zen4-max](https://huggingface.co/zenlm/zen4-max) |
| **Zen4 Pro Max** | **80B MoE** | **3B** | **256K** | [zenlm/zen4-pro-max](https://huggingface.co/zenlm/zen4-pro-max) |
| Zen4 Coder Flash | 31B MoE | 3B | 131K | [zenlm/zen4-coder-flash](https://huggingface.co/zenlm/zen4-coder-flash) |
| **Zen4 Coder** | **80B MoE** | **3B** | **256K** | [zenlm/zen4-coder](https://huggingface.co/zenlm/zen4-coder) |
| Zen4 Ultra | 1.04T MoE | 32B | 256K | [zenlm/zen4-ultra](https://huggingface.co/zenlm/zen4-ultra) |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("zenlm/{m['zen4_name']}")
tokenizer = AutoTokenizer.from_pretrained("zenlm/{m['zen4_name']}")

messages = [{{"role": "user", "content": "Hello, who are you?"}}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Links

- [Zen LM](https://zenlm.org) | [Hanzo AI](https://hanzo.ai)
- [GitHub](https://github.com/zenlm/{m['zen4_name']})
- [All Zen4 Models](https://huggingface.co/zenlm)
"""


def process_model(m):
    repo_dir = ZEN4_DIR / m["dir"] / "repo"

    if not repo_dir.exists():
        print(f"  SKIP {m['display']}: repo not cloned yet at {repo_dir}")
        return False

    # Check if model files exist (at least config.json)
    if not (repo_dir / "config.json").exists():
        print(f"  SKIP {m['display']}: no config.json (LFS pull incomplete?)")
        return False

    print(f"\n{'='*60}")
    print(f"Rebranding: {m['display']} ({m['params']})")
    print(f"  Source: {m['source']}")
    print(f"  Target: zenlm/{m['zen4_name']}")
    print(f"{'='*60}")

    # Write README
    readme_path = repo_dir / "README.md"
    readme_path.write_text(generate_readme(m))
    print(f"  Updated README.md")

    # Change remote to our repo
    os.chdir(repo_dir)
    subprocess.run(
        ["git", "remote", "set-url", "origin", f"https://huggingface.co/zenlm/{m['zen4_name']}"],
        check=True,
    )

    # Commit and push
    subprocess.run(["git", "add", "README.md"], check=True)
    result = subprocess.run(
        ["git", "commit", "-m", f"Zen4: {m['display']} - abliterated base from {m['upstream']}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  (no changes to commit)")

    print(f"  Pushing to zenlm/{m['zen4_name']}...")
    for branch in ["main", "master"]:
        result = subprocess.run(
            ["git", "push", "--force", "origin", f"HEAD:{branch}"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"  Pushed to {branch}")
            print(f"  https://huggingface.co/zenlm/{m['zen4_name']}")
            return True
        else:
            if "error" in result.stderr.lower():
                print(f"  Push to {branch} failed: {result.stderr[:200]}")

    print(f"  PUSH FAILED for {m['display']}")
    return False


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    results = {}
    for m in MODELS:
        if target == "all" or target == m["key"]:
            results[m["key"]] = process_model(m)

    print(f"\n{'='*60}")
    print("Rebrand & Push Summary")
    print(f"{'='*60}")
    for key, ok in results.items():
        name = next(m["display"] for m in MODELS if m["key"] == key)
        print(f"  {name}: {'OK' if ok else 'FAILED/SKIP'}")


if __name__ == "__main__":
    main()
