# Zen4 Model Family - Local Training Workspace

Organized directory for the complete Zen4 lineup with abliterated base models
ready for identity fine-tuning via MLX LoRA on Apple Silicon.

## Directory Structure

```
zen4/
├── mini/              Zen4 Mini (4B dense)
│   ├── training/      Training data + adapters
│   └── models/        Converted output models
├── base/              Zen4 (8B dense)
├── pro/               Zen4 Pro (14B dense)
├── max/               Zen4 Max (30B MoE, 3B active)
├── pro-max/           Zen4 Pro Max (80B MoE, 3B active)
├── coder-flash/       Zen4 Coder Flash (31B MoE, 3B active)
├── coder/             Zen4 Coder (80B MoE, 3B active)
└── README.md
```

## Model Lineup

| Dir | Model | Base (Abliterated) | Total | Active | Context |
|-----|-------|--------------------|-------|--------|---------|
| mini | Zen4 Mini | Qwen3-4B-Instruct-2507 | 4B | 4B | 32K |
| base | Zen4 | Qwen3-8B | 8B | 8B | 32K |
| pro | Zen4 Pro | Qwen3-14B | 14B | 14B | 32K |
| max | Zen4 Max | Qwen3-30B-A3B-Instruct-2507 | 30B MoE | 3B | 256K |
| pro-max | Zen4 Pro Max | Qwen3-Next-80B-A3B-Instruct | 80B MoE | 3B | 256K |
| coder-flash | Zen4 Coder Flash | GLM-4.7-Flash | 31B MoE | 3B | 131K |
| coder | Zen4 Coder | Qwen3-Coder-Next | 80B MoE | 3B | 256K |

## Abliterated Base Models (HuggingFace)

- `huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated`
- `huihui-ai/Huihui-Qwen3-8B-abliterated-v2`
- `mlabonne/Qwen3-14B-abliterated`
- `huihui-ai/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated`
- `huihui-ai/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-mlx-4bit`
- `huihui-ai/Huihui-GLM-4.7-Flash-abliterated`
- `huihui-ai/Huihui-Qwen3-Coder-Next-abliterated`

## Training

```bash
# From ~/work/zen/
python train_identity.py --lineup          # Show lineup
python train_identity.py --generate        # Generate training data
python train_identity.py --model mini      # Train single model
python train_identity.py                   # Train all models
python train_identity.py --test            # Test after training
```

## Hardware Requirements (Apple Silicon)

- M1 Max 64GB can train all models (MoE models only activate 3B params)
- Dense models: 4-bit quantization via MLX
- Training: ~200 iterations, 8 LoRA layers, 1e-5 learning rate
- Time: ~5-15 min per model depending on size

## Output

After training, adapters are saved to `{model}/training/output/adapters/`.
Models are then fused, quantized, and uploaded to `zenlm/` on HuggingFace.
