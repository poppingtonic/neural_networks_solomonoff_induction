# Pareto Models - Quick Reference Card

## 🚀 TL;DR - Start Here

### One-Line Recommendation
**Use `Skywork/Skywork-Reward-V2-Qwen3-4B` for best balance of performance and efficiency**

### Quick Start Commands

```bash
# RECOMMENDED: 4B Model (Best Balance)
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-4B \
    --utm_steps 10000 --ctw_steps 5000 \
    --batch_size 32 --lr 1e-5 \
    --output_dir ./checkpoints/pareto_4b

# ELITE: 8B Model (Maximum Performance)
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path Skywork/Skywork-Reward-V2-Llama-3.1-8B \
    --utm_steps 10000 --ctw_steps 5000 \
    --batch_size 32 --lr 1e-5 \
    --output_dir ./checkpoints/pareto_8b

# BUDGET: 0.6B Model (Current Baseline)
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B \
    --utm_steps 10000 --ctw_steps 5000 \
    --batch_size 64 --lr 1e-5 \
    --output_dir ./checkpoints/pareto_0.6b
```

---

## 📊 Pareto-Optimal Models Summary

| Rank | Model | Size | Score | Math | Best For |
|------|-------|------|-------|------|----------|
| 🥇 | **Skywork-Reward-V2-Llama-3.1-8B** | 8B | 84.13 | 77.60 | Overall Best |
| 🥈 | **Skywork-Reward-V2-Qwen3-4B** | 4B | 75.51 | 73.22 | Best Efficiency |
| 🥉 | **Skywork-Reward-V2-Qwen3-8B** | 8B | 78.37 | 77.05 | Math Focus |

---

## 💰 Choose by Budget

| VRAM | Model | Size | Score | Command Flag |
|------|-------|------|-------|--------------|
| < 8GB | Skywork-Reward-V2-Qwen3-0.6B | 0.6B | 61.25 | `--model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-0.6B` |
| 8-16GB | **Skywork-Reward-V2-Qwen3-4B** ⭐ | 4B | 75.51 | `--model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-4B` |
| 16-24GB | **Skywork-Reward-V2-Llama-3.1-8B** ⭐ | 8B | 84.13 | `--model_name_or_path Skywork/Skywork-Reward-V2-Llama-3.1-8B` |
| > 80GB | LMUnit-qwen2.5-72b | 72B | 82.08 | `--model_name_or_path ContextualAI/LMUnit-qwen2.5-72b` |

---

## 🎯 Choose by Priority

| Priority | Model | Why |
|----------|-------|-----|
| **Best Overall** | Skywork-Reward-V2-Llama-3.1-8B | Highest score (84.13), excellent all-around |
| **Best Efficiency** | Skywork-Reward-V2-Qwen3-4B | 75.51 score at only 4B params |
| **Best Math** | Skywork-Reward-V2-Qwen3-8B | 77.05 math score (critical for UTM) |
| **Ultra-Safe** | INF-ORM-Llama3.1-70B | 96.44% safety rating |
| **Current Baseline** | Skywork-Reward-V2-Qwen3-0.6B | Already in SEQUENTIAL_FINETUNING_RECIPE.md |

---

## 📈 Progressive Scaling Strategy

```
Start Small → Scale Up

Stage 0: Skywork-Reward-V2-Qwen3-0.6B  (Warmup, 2-4h)
    ↓
Stage 1: Skywork-Reward-V2-Qwen3-1.7B  (Transfer, 4-6h)
    ↓
Stage 2: Skywork-Reward-V2-Qwen3-4B    (Final, 8-12h)

Total Training Time: ~20h
Final Performance: ~75 score (vs 61 baseline)
```

---

## 🔬 Analysis Tools

```bash
# Run Pareto analysis
python scripts/analyze_pareto_simple.py

# Output shows:
# - Top Pareto-optimal models
# - Recommendations by size class
# - Best models for your budget
```

---

## 📚 Full Documentation

- **Comprehensive Guide**: [`PARETO_REWARD_MODELS_MATRIX.md`](PARETO_REWARD_MODELS_MATRIX.md)
- **Training Recipe**: [`SEQUENTIAL_FINETUNING_RECIPE.md`](SEQUENTIAL_FINETUNING_RECIPE.md)
- **Data Source**: [`current-rbv2-data.csv`](current-rbv2-data.csv) (70 models)

---

## ⚡ Key Insights

1. **Skywork models dominate** across all size classes
2. **4B is the sweet spot** for efficiency (75.51 score, 4B params)
3. **8B Llama** is best overall (84.13 score) if you have 24GB+ VRAM
4. **Math score matters** for UTM stage (look for 70+ math score)
5. **Qwen architecture** performs well on math tasks

---

## 🎓 Decision Tree

```
Do you have 24GB+ VRAM?
├─ YES → Use Skywork-Reward-V2-Llama-3.1-8B (84.13 score)
└─ NO
   └─ Do you have 16GB VRAM?
      ├─ YES → Use Skywork-Reward-V2-Qwen3-4B (75.51 score) ⭐ RECOMMENDED
      └─ NO  → Use Skywork-Reward-V2-Qwen3-0.6B (61.25 score)
```

---

**Last Updated**: 2025-10-25  
**Status**: ✅ Production Ready  
**Default Recommendation**: `Skywork/Skywork-Reward-V2-Qwen3-4B`
