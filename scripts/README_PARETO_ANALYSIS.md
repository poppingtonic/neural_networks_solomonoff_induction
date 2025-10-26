# Pareto Analysis Scripts - README

## Overview

Two scripts for analyzing Pareto-optimal reward models from the RewardBench v2 dataset:

1. **`analyze_pareto_models.py`** - Full-featured pandas-based analysis with visualizations
2. **`analyze_pareto_simple.py`** - Lightweight version (numpy only, no pandas)

## Installation

### For Full-Featured Analysis (Recommended)

```bash
# Install pandas and visualization libraries
pip install pandas>=1.5.0 matplotlib>=3.5.0 seaborn>=0.12.0

# Or install all sequential finetuning requirements
pip install -r requirements_sequential_finetune.txt
```

### For Lightweight Analysis

```bash
# Only numpy required (already in base requirements.txt)
pip install numpy
```

## Usage

### Full-Featured Analysis (`analyze_pareto_models.py`)

```bash
# Run basic analysis (console output only)
python scripts/analyze_pareto_models.py

# Export CSV files to analysis_results/
python scripts/analyze_pareto_models.py --export

# Generate visualization plots
python scripts/analyze_pareto_models.py --visualize

# Both export and visualize
python scripts/analyze_pareto_models.py --all
```

### Lightweight Analysis (`analyze_pareto_simple.py`)

```bash
# Run analysis (console output only)
python scripts/analyze_pareto_simple.py
```

## Output Files

### CSV Exports (with `--export`)

Located in `analysis_results/`:

- **`pareto_frontier_models.csv`** - All Pareto-optimal models
- **`tier_elite_models.csv`** - Elite tier models (Score > 78, Math > 70)
- **`tier_specialized_models.csv`** - Specialized models
- **`tier_efficient_models.csv`** - Efficient small models
- **`tier_specialists_models.csv`** - Task specialists
- **`efficiency_rankings.csv`** - Top 50 models by efficiency (Score/Size)
- **`summary_statistics.csv`** - Overall statistics

### Visualizations (with `--visualize`)

Located in `analysis_results/plots/`:

1. **`pareto_score_vs_math.png`** - Pareto frontier scatter plot
   - Shows all models with Pareto-optimal models highlighted
   - X-axis: Math score, Y-axis: Overall score
   - Top 5 models annotated

2. **`size_vs_score.png`** - Model size vs performance
   - Color-coded by efficiency (Score/Size)
   - Log scale for size
   - Sequence classifiers only

3. **`architecture_comparison.png`** - Box plot by architecture family
   - Compares Llama, Qwen, Gemma, etc.
   - Shows distribution of scores

4. **`top5_radar.png`** - Multi-dimensional radar chart
   - Top 5 Pareto models
   - Dimensions: Score, Math, Safety, Focus, Factuality

5. **`size_class_heatmap.png`** - Performance by size class
   - Rows: Size classes (0-1B, 1-3B, 3-5B, 5-10B, 10B+)
   - Columns: Metrics (Score, Math, Safety, Focus)
   - Color intensity shows average performance

## Features

### Full-Featured Script (`analyze_pareto_models.py`)

✅ **Pareto Frontier Analysis**
- Multi-objective optimization (Score, Math, Safety, Focus)
- Identifies non-dominated models

✅ **Tier Classification**
- Elite: Score > 78, Math > 70
- Specialized: Score > 75, strong in specific dimension
- Efficient: Size < 5B, Score > 60
- Specialists: Exceptional in one metric

✅ **Architecture Family Analysis**
- Groups by Llama, Qwen, Gemma, Mistral, etc.
- Best model per family
- Average scores per family

✅ **Budget-Based Recommendations**
- Low (<8GB VRAM)
- Medium (16GB VRAM)
- High (24-40GB VRAM)
- Very High (>80GB VRAM)

✅ **Efficiency Analysis**
- Score per billion parameters
- Identifies best bang-for-buck models

✅ **Sequential Finetuning Strategy**
- Progressive capacity scaling
- Size class recommendations

✅ **Rich Visualizations** (5 plots)
- Pareto frontier scatter
- Size vs performance
- Architecture comparison
- Multi-dimensional radar
- Size class heatmap

### Lightweight Script (`analyze_pareto_simple.py`)

✅ **Core Pareto Analysis**
- Identifies Pareto-optimal models
- Size class recommendations
- Top recommendations by budget

✅ **No External Dependencies**
- Only numpy (already in base requirements)
- Faster execution
- Perfect for CI/CD pipelines

## Example Output

```
================================================================================
PARETO FRONTIER ANALYSIS - REWARD MODELS
================================================================================

Loading data from: /path/to/current-rbv2-data.csv
Loaded 70 models
Model types: {'Seq. Classifier': 50, 'Generative': 18, 'Custom Classifier': 2}

================================================================================
PARETO FRONTIER - PRIMARY OBJECTIVES
================================================================================

Pareto-optimal models (from 70 total): 5

Top 10 Pareto-Optimal Models:
--------------------------------------------------------------------------------

Skywork/Skywork-Reward-V2-Llama-3.1-8B
  Size: 8.0B | Score: 84.13 | Math: 77.60 | Safety: 96.67 | Focus: 98.38
  Type: Seq. Classifier

google/gemini-2.5-pro
  Size: 7.0B | Score: 79.48 | Math: 89.80 | Safety: 88.10 | Focus: 80.50
  Type: Generative

...

================================================================================
TOP RECOMMENDATIONS
================================================================================

⭐ RECOMMENDED (4B - Best Balance):
   Skywork/Skywork-Reward-V2-Qwen3-4B
   Score: 75.51 | Size: 4.0B

⭐ ELITE (8B - Maximum Performance):
   Skywork/Skywork-Reward-V2-Llama-3.1-8B
   Score: 84.13 | Size: 8.0B

⭐ BUDGET (0.6-1B - Baseline):
   Skywork/Skywork-Reward-V2-Qwen3-0.6B
   Score: 61.25 | Size: 0.6B
```

## Integration with Sequential Finetuning

The analysis results directly inform model selection for the sequential finetuning pipeline:

```bash
# Use recommended 4B model
python torch_training/train_sequential_finetune.py \
    --stage all \
    --model_name_or_path Skywork/Skywork-Reward-V2-Qwen3-4B \
    --utm_steps 10000 --ctw_steps 5000 \
    --batch_size 32 --lr 1e-5 \
    --output_dir ./checkpoints/pareto_4b
```

## Performance Metrics Explained

- **Score**: Overall performance across all tasks (0-100)
- **Math**: Mathematical reasoning capability (0-100)
- **Safety**: Safety alignment score (0-100)
- **Focus**: Task focus/attention score (0-100)
- **Factuality**: Factual accuracy (0-100)
- **Size_B**: Model size in billions of parameters

## Pareto Optimality

A model is **Pareto-optimal** if no other model is better in ALL dimensions simultaneously.

**Example:**
- Model A: Score=80, Math=75, Safety=90
- Model B: Score=82, Math=70, Safety=85

Model A is Pareto-optimal because Model B has better overall score but worse math and safety. Both are on the Pareto frontier.

## Troubleshooting

### ModuleNotFoundError: No module named 'pandas'

```bash
pip install pandas matplotlib seaborn
```

Or use the lightweight script:
```bash
python scripts/analyze_pareto_simple.py
```

### Visualizations Not Generated

Ensure matplotlib and seaborn are installed:
```bash
pip install matplotlib seaborn
```

### CSV File Not Found

Make sure `current-rbv2-data.csv` exists in the project root:
```bash
ls -l current-rbv2-data.csv
```

## Development

### Adding New Visualizations

Edit `create_visualizations()` function in `analyze_pareto_models.py`:

```python
def create_visualizations(df: pd.DataFrame, pareto_df: pd.DataFrame, output_dir: Path):
    # Add your custom visualization here
    plt.figure()
    # ... your plot code ...
    plt.savefig(plots_dir / 'my_new_plot.png')
    plt.close()
```

### Custom Tier Definitions

Edit `tier_models()` function:

```python
def tier_models(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    tiers = {}
    
    # Add custom tier
    tiers['my_tier'] = df[
        (df['Score'] > 70) & 
        (df['Custom_Metric'] > 80)
    ].copy()
    
    return tiers
```

## References

- **Data Source**: `current-rbv2-data.csv` (RewardBench v2, 70 models)
- **Documentation**: `PARETO_REWARD_MODELS_MATRIX.md`
- **Quick Reference**: `PARETO_QUICK_REFERENCE.md`
- **Summary**: `PARETO_SUMMARY.txt`

## License

Same as parent project.

---

**Last Updated**: 2025-10-25  
**Maintained by**: Pareto analysis workflow  
**Status**: ✅ Production Ready
