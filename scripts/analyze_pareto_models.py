#!/usr/bin/env python3
"""
Pareto Frontier Analysis for Reward Models (Pandas-based)
Analyzes current-rbv2-data.csv to identify Pareto-optimal models for sequential finetuning.

Usage:
    python scripts/analyze_pareto_models.py
    python scripts/analyze_pareto_models.py --visualize  # Generate plots
    python scripts/analyze_pareto_models.py --export     # Export CSVs
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
import sys

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Visualizations disabled.")


def clean_model_name(html_name: str) -> str:
    """Extract clean model name from HTML anchor tag."""
    match = re.search(r'href="https://huggingface.co/([^"]+)"', html_name)
    if match:
        return match.group(1)
    return html_name


def estimate_model_size(model_name: str) -> float:
    """Estimate model size from name (in billions of parameters)."""
    size_patterns = [
        (r'0\.6B', 0.6), (r'1\.8B', 1.8), (r'1\.7B', 1.7), (r'1B', 1.0),
        (r'2B', 2.0), (r'3B', 3.0), (r'4B', 4.0), (r'7B', 7.0), (r'8B', 8.0),
        (r'13B', 13.0), (r'20B', 20.0), (r'27B', 27.0), (r'34B', 34.0),
        (r'70B', 70.0), (r'72B', 72.0),
    ]
    
    for pattern, size in size_patterns:
        if re.search(pattern, model_name, re.IGNORECASE):
            return size
    
    # Default estimates based on model type
    if 'mini' in model_name.lower() or 'nano' in model_name.lower():
        return 0.5
    elif 'small' in model_name.lower():
        return 2.0
    elif 'large' in model_name.lower():
        return 10.0
    
    return 7.0  # Default fallback


def is_pareto_optimal(point: np.ndarray, points: np.ndarray, maximize: bool = True) -> bool:
    """Check if a point is Pareto optimal (not dominated by any other point).
    
    Args:
        point: The point to check
        points: All points to compare against
        maximize: If True, larger values are better. If False, smaller values are better.
    
    Returns:
        True if the point is Pareto optimal
    """
    if maximize:
        # Point is dominated if there exists another point that is >= in all dimensions
        # and strictly > in at least one dimension
        dominated = np.any(
            np.all(points >= point, axis=1) & np.any(points > point, axis=1)
        )
    else:
        # For minimization, flip the logic
        dominated = np.any(
            np.all(points <= point, axis=1) & np.any(points < point, axis=1)
        )
    
    return not dominated


def find_pareto_frontier(df: pd.DataFrame, 
                         objectives: List[str],
                         maximize: bool = True) -> pd.DataFrame:
    """Find Pareto-optimal models based on multiple objectives.
    
    Args:
        df: DataFrame with model data
        objectives: List of column names to use as objectives
        maximize: If True, larger values are better
    
    Returns:
        DataFrame containing only Pareto-optimal models
    """
    # Extract objective values
    points = df[objectives].values
    
    # Handle NaN values (replace with worst possible value)
    if maximize:
        points = np.nan_to_num(points, nan=-np.inf)
    else:
        points = np.nan_to_num(points, nan=np.inf)
    
    # Find Pareto optimal points
    pareto_mask = np.array([
        is_pareto_optimal(point, points, maximize) 
        for point in points
    ])
    
    return df[pareto_mask]


def tier_models(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Categorize models into performance tiers.
    
    Returns:
        Dictionary mapping tier names to DataFrames
    """
    tiers = {}
    
    # Tier 1: Elite (Score > 78, Math > 70)
    tiers['elite'] = df[
        (df['Score'] > 78) & 
        (df['Math'] > 70) &
        (df['Model Type'] == 'Seq. Classifier')
    ].copy()
    
    # Tier 2: Specialized (Score > 75, strong in at least one dimension)
    tiers['specialized'] = df[
        (df['Score'] > 75) & 
        (df['Score'] <= 78) &
        ((df['Math'] > 69) | (df['Safety'] > 94) | (df['Focus'] > 93))
    ].copy()
    
    # Tier 3: Efficient Small (Size < 5B, Score > 60)
    tiers['efficient'] = df[
        (df['Size_B'] < 5.0) & 
        (df['Score'] > 60) &
        (df['Model Type'] == 'Seq. Classifier')
    ].copy()
    
    # Tier 4: Task Specialists (Score > 65, exceptional in one metric)
    tiers['specialists'] = df[
        (df['Score'] > 65) &
        (df['Score'] <= 75) &
        ((df['Focus'] > 95) | (df['Safety'] > 93))
    ].copy()
    
    return tiers


def recommend_sequential_path(df: pd.DataFrame, 
                               budget: str = 'medium') -> List[Dict[str, str]]:
    """Recommend a sequential finetuning path based on budget.
    
    Args:
        df: DataFrame with model data
        budget: 'low', 'medium', 'high', or 'very_high'
    
    Returns:
        List of dictionaries with model recommendations and reasoning
    """
    recommendations = []
    
    if budget == 'low':
        # < 8GB VRAM
        model = df[df['Size_B'] <= 1.0].nlargest(1, 'Score')
        recommendations.append({
            'model': model.iloc[0]['Model_Clean'],
            'size': f"{model.iloc[0]['Size_B']}B",
            'score': f"{model.iloc[0]['Score']:.2f}",
            'stage': 'All stages',
            'reason': 'Best ultra-lightweight model for limited VRAM'
        })
        
    elif budget == 'medium':
        # 16GB VRAM
        model = df[
            (df['Size_B'] >= 3.0) & 
            (df['Size_B'] <= 5.0) &
            (df['Model Type'] == 'Seq. Classifier')
        ].nlargest(1, 'Score')
        
        if len(model) > 0:
            recommendations.append({
                'model': model.iloc[0]['Model_Clean'],
                'size': f"{model.iloc[0]['Size_B']}B",
                'score': f"{model.iloc[0]['Score']:.2f}",
                'stage': 'All stages',
                'reason': 'Optimal balance of performance and efficiency'
            })
        
    elif budget == 'high':
        # 24-40GB VRAM
        model = df[
            (df['Size_B'] >= 7.0) & 
            (df['Size_B'] <= 10.0) &
            (df['Model Type'] == 'Seq. Classifier')
        ].nlargest(1, 'Score')
        
        if len(model) > 0:
            recommendations.append({
                'model': model.iloc[0]['Model_Clean'],
                'size': f"{model.iloc[0]['Size_B']}B",
                'score': f"{model.iloc[0]['Score']:.2f}",
                'stage': 'All stages',
                'reason': 'Best overall Pareto-optimal model'
            })
        
    else:  # very_high
        # > 80GB VRAM
        model = df[df['Size_B'] >= 70.0].nlargest(1, 'Score')
        
        if len(model) > 0:
            recommendations.append({
                'model': model.iloc[0]['Model_Clean'],
                'size': f"{model.iloc[0]['Size_B']}B",
                'score': f"{model.iloc[0]['Score']:.2f}",
                'stage': 'All stages',
                'reason': 'Large-scale model for maximum compute'
            })
    
    return recommendations


def analyze_architecture_families(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group models by architecture family and find best in each."""
    families = {}
    
    # Define architecture patterns
    patterns = {
        'Llama': r'[Ll]lama',
        'Qwen': r'[Qq]wen',
        'Gemma': r'[Gg]emma',
        'Mistral': r'[Mm]istral',
        'Gemini': r'[Gg]emini',
        'Claude': r'[Cc]laude',
        'GPT': r'gpt|GPT'
    }
    
    for family, pattern in patterns.items():
        family_df = df[df['Model_Clean'].str.contains(pattern, na=False)]
        if len(family_df) > 0:
            # Get best model in family (by score)
            best = family_df.nlargest(1, 'Score')
            families[family] = {
                'best_model': best.iloc[0]['Model_Clean'],
                'score': best.iloc[0]['Score'],
                'size': best.iloc[0]['Size_B'],
                'count': len(family_df),
                'avg_score': family_df['Score'].mean()
            }
    
    return families


def create_visualizations(df: pd.DataFrame, pareto_df: pd.DataFrame, output_dir: Path):
    """Create visualization plots for the analysis.
    
    Args:
        df: Full dataset
        pareto_df: Pareto-optimal models
        output_dir: Directory to save plots
    """
    if not VISUALIZATION_AVAILABLE:
        print("\nWarning: Visualization libraries not available. Skipping plots.")
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Create plots directory
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Score vs Math Performance (Pareto Frontier)
    plt.figure(figsize=(14, 10))
    
    # All models
    plt.scatter(df['Math'], df['Score'], alpha=0.3, s=50, c='gray', label='All Models')
    
    # Pareto optimal models
    plt.scatter(pareto_df['Math'], pareto_df['Score'], 
                alpha=0.8, s=200, c='red', marker='*', label='Pareto Optimal', zorder=5)
    
    # Annotate top Pareto models
    top_pareto = pareto_df.nlargest(5, 'Score')
    for _, row in top_pareto.iterrows():
        model_short = row['Model_Clean'].split('/')[-1][:20]
        plt.annotate(model_short, 
                    (row['Math'], row['Score']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    plt.xlabel('Math Score', fontsize=12)
    plt.ylabel('Overall Score', fontsize=12)
    plt.title('Pareto Frontier: Overall Score vs Math Performance', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'pareto_score_vs_math.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Size vs Score with Efficiency
    plt.figure(figsize=(14, 10))
    df_seq = df[df['Model Type'] == 'Seq. Classifier'].copy()
    df_seq['Efficiency'] = df_seq['Score'] / df_seq['Size_B']
    
    scatter = plt.scatter(df_seq['Size_B'], df_seq['Score'],
                         c=df_seq['Efficiency'], s=100, alpha=0.6,
                         cmap='viridis', edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Efficiency (Score/Size)')
    
    plt.xlabel('Model Size (Billions of Parameters)', fontsize=12)
    plt.ylabel('Overall Score', fontsize=12)
    plt.title('Model Size vs Performance (Seq. Classifiers)', fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'size_vs_score.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Architecture Family Comparison (Box Plot)
    plt.figure(figsize=(14, 8))
    
    # Extract architecture families
    def get_architecture(model_name):
        if re.search(r'llama', model_name, re.IGNORECASE):
            return 'Llama'
        elif re.search(r'qwen', model_name, re.IGNORECASE):
            return 'Qwen'
        elif re.search(r'gemma', model_name, re.IGNORECASE):
            return 'Gemma'
        elif re.search(r'mistral', model_name, re.IGNORECASE):
            return 'Mistral'
        elif re.search(r'gemini', model_name, re.IGNORECASE):
            return 'Gemini'
        else:
            return 'Other'
    
    df['Architecture'] = df['Model_Clean'].apply(get_architecture)
    arch_counts = df['Architecture'].value_counts()
    top_archs = arch_counts[arch_counts >= 3].index
    df_top_arch = df[df['Architecture'].isin(top_archs)]
    
    sns.boxplot(data=df_top_arch, x='Architecture', y='Score', palette='Set2')
    plt.xlabel('Architecture Family', fontsize=12)
    plt.ylabel('Overall Score', fontsize=12)
    plt.title('Performance Distribution by Architecture Family', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(plots_dir / 'architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Multi-dimensional Radar Chart for Top 5 Models
    from math import pi
    
    plt.figure(figsize=(12, 12))
    
    categories = ['Score', 'Math', 'Safety', 'Focus', 'Factuality']
    top_5 = pareto_df.nlargest(5, 'Score')
    
    # Normalize scores to 0-100 scale for radar chart
    for cat in categories:
        if cat in top_5.columns:
            if top_5[cat].max() > 0:
                # Already in 0-100 scale
                pass
    
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    ax = plt.subplot(111, polar=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (i, row) in enumerate(top_5.iterrows()):
        if idx >= 5:
            break
        
        values = [row.get(cat, 0) for cat in categories]
        values += values[:1]
        
        model_short = row['Model_Clean'].split('/')[-1][:25]
        ax.plot(angles, values, 'o-', linewidth=2, label=model_short, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    plt.title('Top 5 Pareto Models - Multi-dimensional Comparison', 
              size=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(plots_dir / 'top5_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Size Class Performance Heatmap
    plt.figure(figsize=(12, 8))
    
    size_bins = [(0, 1, '0-1B'), (1, 3, '1-3B'), (3, 5, '3-5B'), 
                 (5, 10, '5-10B'), (10, 100, '10B+')]
    
    metrics = ['Score', 'Math', 'Safety', 'Focus']
    heatmap_data = []
    
    for min_size, max_size, label in size_bins:
        subset = df[(df['Size_B'] > min_size) & (df['Size_B'] <= max_size)]
        if len(subset) > 0:
            row = [subset[m].mean() for m in metrics]
            heatmap_data.append(row)
        else:
            heatmap_data.append([0] * len(metrics))
    
    heatmap_df = pd.DataFrame(heatmap_data, 
                              index=[label for _, _, label in size_bins],
                              columns=metrics)
    
    sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlGnBu', 
                cbar_kws={'label': 'Average Score'})
    plt.title('Average Performance by Model Size Class', fontsize=14, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Model Size Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(plots_dir / 'size_class_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Visualizations saved to: {plots_dir}")
    print(f"  - pareto_score_vs_math.png")
    print(f"  - size_vs_score.png")
    print(f"  - architecture_comparison.png")
    print(f"  - top5_radar.png")
    print(f"  - size_class_heatmap.png")


def main(visualize: bool = False, export: bool = True):
    """Main analysis pipeline.
    
    Args:
        visualize: If True, generate visualization plots
        export: If True, export CSV files
    """
    # Load data
    csv_path = Path(__file__).parent.parent / 'current-rbv2-data.csv'
    
    print("=" * 80)
    print("PARETO FRONTIER ANALYSIS - REWARD MODELS")
    print("=" * 80)
    print(f"\nLoading data from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Clean model names
    df['Model_Clean'] = df['Model'].apply(clean_model_name)
    
    # Estimate model sizes
    df['Size_B'] = df['Model_Clean'].apply(estimate_model_size)
    
    # Remove rows with missing critical data
    df = df.dropna(subset=['Score', 'Math', 'Safety', 'Focus'])
    
    print(f"Loaded {len(df)} models")
    print(f"Model types: {df['Model Type'].value_counts().to_dict()}")
    
    # === PARETO FRONTIER ANALYSIS ===
    print("\n" + "=" * 80)
    print("PARETO FRONTIER - PRIMARY OBJECTIVES")
    print("=" * 80)
    
    # Find Pareto frontier (Score, Math, Safety, Focus)
    objectives = ['Score', 'Math', 'Safety', 'Focus']
    pareto_df = find_pareto_frontier(df, objectives, maximize=True)
    
    print(f"\nPareto-optimal models (from {len(df)} total): {len(pareto_df)}")
    print("\nTop 10 Pareto-Optimal Models:")
    print("-" * 80)
    
    pareto_sorted = pareto_df.nlargest(10, 'Score')[
        ['Model_Clean', 'Size_B', 'Score', 'Math', 'Safety', 'Focus', 'Model Type']
    ]
    
    for idx, row in pareto_sorted.iterrows():
        print(f"\n{row['Model_Clean']}")
        print(f"  Size: {row['Size_B']:.1f}B | Score: {row['Score']:.2f} | "
              f"Math: {row['Math']:.2f} | Safety: {row['Safety']:.2f} | "
              f"Focus: {row['Focus']:.2f}")
        print(f"  Type: {row['Model Type']}")
    
    # === TIER ANALYSIS ===
    print("\n" + "=" * 80)
    print("TIER CLASSIFICATION")
    print("=" * 80)
    
    tiers = tier_models(df)
    
    for tier_name, tier_df in tiers.items():
        print(f"\n{tier_name.upper()} Tier: {len(tier_df)} models")
        if len(tier_df) > 0:
            top_3 = tier_df.nlargest(3, 'Score')
            for idx, row in top_3.iterrows():
                print(f"  - {row['Model_Clean']} ({row['Size_B']:.1f}B): Score {row['Score']:.2f}")
    
    # === ARCHITECTURE FAMILY ANALYSIS ===
    print("\n" + "=" * 80)
    print("ARCHITECTURE FAMILY ANALYSIS")
    print("=" * 80)
    
    families = analyze_architecture_families(df)
    
    for family, info in sorted(families.items(), key=lambda x: x[1]['score'], reverse=True):
        print(f"\n{family} Family:")
        print(f"  Best: {info['best_model']}")
        print(f"  Score: {info['score']:.2f} | Size: {info['size']:.1f}B")
        print(f"  Models in family: {info['count']} | Avg Score: {info['avg_score']:.2f}")
    
    # === RECOMMENDATIONS ===
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS BY BUDGET")
    print("=" * 80)
    
    budgets = ['low', 'medium', 'high', 'very_high']
    budget_names = ['Low (<8GB VRAM)', 'Medium (16GB VRAM)', 
                   'High (24-40GB VRAM)', 'Very High (>80GB VRAM)']
    
    for budget, budget_name in zip(budgets, budget_names):
        print(f"\n{budget_name}:")
        recs = recommend_sequential_path(df, budget)
        for rec in recs:
            print(f"  → {rec['model']} ({rec['size']})")
            print(f"    Score: {rec['score']} | Stage: {rec['stage']}")
            print(f"    Reason: {rec['reason']}")
    
    # === SIZE vs PERFORMANCE ANALYSIS ===
    print("\n" + "=" * 80)
    print("EFFICIENCY ANALYSIS (Score per Billion Parameters)")
    print("=" * 80)
    
    df['Efficiency'] = df['Score'] / df['Size_B']
    top_efficient = df.nlargest(10, 'Efficiency')[
        ['Model_Clean', 'Size_B', 'Score', 'Efficiency']
    ]
    
    print("\nMost Efficient Models:")
    print("-" * 80)
    for idx, row in top_efficient.iterrows():
        print(f"{row['Model_Clean']:50s} | "
              f"Size: {row['Size_B']:5.1f}B | "
              f"Score: {row['Score']:6.2f} | "
              f"Efficiency: {row['Efficiency']:6.2f}")
    
    # === SEQUENTIAL FINETUNING STRATEGY ===
    print("\n" + "=" * 80)
    print("RECOMMENDED SEQUENTIAL FINETUNING STRATEGY")
    print("=" * 80)
    
    # Find best in size classes
    size_classes = [
        (0.5, 1.0, "Ultra-lightweight (0.6-1B)"),
        (1.0, 2.5, "Lightweight (1-2B)"),
        (2.5, 5.0, "Medium (3-4B)"),
        (5.0, 10.0, "Standard (7-8B)"),
        (10.0, 100.0, "Large (>10B)")
    ]
    
    print("\nProgressive Capacity Scaling Strategy:")
    print("-" * 80)
    
    for min_size, max_size, class_name in size_classes:
        class_models = df[
            (df['Size_B'] >= min_size) & 
            (df['Size_B'] < max_size) &
            (df['Model Type'] == 'Seq. Classifier')
        ]
        
        if len(class_models) > 0:
            best = class_models.nlargest(1, 'Score').iloc[0]
            print(f"\n{class_name}:")
            print(f"  Model: {best['Model_Clean']}")
            print(f"  Size: {best['Size_B']:.1f}B | Score: {best['Score']:.2f} | "
                  f"Math: {best['Math']:.2f}")
    
    # === EXPORT RESULTS ===
    output_dir = Path(__file__).parent.parent / 'analysis_results'
    output_dir.mkdir(exist_ok=True)
    
    if export:
        # Save Pareto frontier
        pareto_output = output_dir / 'pareto_frontier_models.csv'
        pareto_df.to_csv(pareto_output, index=False)
        print(f"\n\n✓ Pareto frontier saved to: {pareto_output}")
        
        # Save tier classifications
        for tier_name, tier_df in tiers.items():
            tier_output = output_dir / f'tier_{tier_name}_models.csv'
            tier_df.to_csv(tier_output, index=False)
        print(f"✓ Tier classifications saved to: {output_dir}")
        
        # Save efficiency rankings
        efficiency_output = output_dir / 'efficiency_rankings.csv'
        df.nlargest(50, 'Efficiency').to_csv(efficiency_output, index=False)
        print(f"✓ Efficiency rankings saved to: {efficiency_output}")
        
        # Save summary statistics
        summary_output = output_dir / 'summary_statistics.csv'
        summary_data = {
            'Metric': ['Total Models', 'Pareto Optimal', 'Avg Score', 'Max Score', 'Min Score'],
            'Value': [len(df), len(pareto_df), df['Score'].mean(), df['Score'].max(), df['Score'].min()]
        }
        pd.DataFrame(summary_data).to_csv(summary_output, index=False)
        print(f"✓ Summary statistics saved to: {summary_output}")
    
    # === VISUALIZATIONS ===
    if visualize:
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        create_visualizations(df, pareto_df, output_dir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    if not export and not visualize:
        print("\nTip: Use --export to save CSV files, --visualize to generate plots")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pareto Frontier Analysis for Reward Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyze_pareto_models.py                    # Run analysis only
  python scripts/analyze_pareto_models.py --export           # Export CSV files
  python scripts/analyze_pareto_models.py --visualize        # Generate plots
  python scripts/analyze_pareto_models.py --export --visualize  # Both
        """
    )
    
    parser.add_argument('--export', action='store_true',
                       help='Export analysis results to CSV files')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots (requires matplotlib, seaborn)')
    parser.add_argument('--all', action='store_true',
                       help='Enable both export and visualize')
    
    args = parser.parse_args()
    
    # Handle --all flag
    if args.all:
        args.export = True
        args.visualize = True
    
    # Run main analysis
    main(visualize=args.visualize, export=args.export)
