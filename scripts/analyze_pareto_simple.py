#!/usr/bin/env python3
"""
Simplified Pareto Analysis for Reward Models (no pandas dependency)
"""

import csv
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


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
    return 7.0


def is_pareto_optimal(scores: List[float], all_scores: List[List[float]]) -> bool:
    """Check if a model is Pareto optimal."""
    for other in all_scores:
        if other == scores:
            continue
        # Check if 'other' dominates 'scores'
        if all(o >= s for o, s in zip(other, scores)) and any(o > s for o, s in zip(other, scores)):
            return False
    return True


def main():
    csv_path = Path(__file__).parent.parent / 'current-rbv2-data.csv'
    
    print("=" * 80)
    print("PARETO ANALYSIS - REWARD MODELS (SIMPLIFIED)")
    print("=" * 80)
    
    # Load CSV
    models = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                model_name = clean_model_name(row['Model'])
                score = float(row['Score'])
                math = float(row['Math']) if row['Math'] else 0
                safety = float(row['Safety']) if row['Safety'] else 0
                focus = float(row['Focus']) if row['Focus'] else 0
                size = estimate_model_size(model_name)
                
                models.append({
                    'name': model_name,
                    'type': row['Model Type'],
                    'score': score,
                    'math': math,
                    'safety': safety,
                    'focus': focus,
                    'size': size,
                    'pareto_scores': [score, math, safety, focus]
                })
            except (ValueError, KeyError):
                continue
    
    print(f"\nLoaded {len(models)} models\n")
    
    # Find Pareto frontier
    all_scores = [m['pareto_scores'] for m in models]
    for model in models:
        model['is_pareto'] = is_pareto_optimal(model['pareto_scores'], all_scores)
    
    pareto_models = [m for m in models if m['is_pareto']]
    pareto_models.sort(key=lambda x: x['score'], reverse=True)
    
    print("=" * 80)
    print(f"PARETO FRONTIER: {len(pareto_models)} models")
    print("=" * 80)
    
    print("\nTop 15 Pareto-Optimal Models:")
    print("-" * 80)
    for i, m in enumerate(pareto_models[:15], 1):
        print(f"{i:2d}. {m['name']}")
        print(f"    Size: {m['size']:.1f}B | Score: {m['score']:.2f} | Math: {m['math']:.2f} | "
              f"Safety: {m['safety']:.2f} | Focus: {m['focus']:.2f}")
    
    # Size class recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDED MODELS BY SIZE CLASS")
    print("=" * 80)
    
    size_classes = [
        (0.5, 1.0, "Ultra-lightweight (0.6-1B)"),
        (1.0, 2.5, "Lightweight (1-2B)"),
        (2.5, 5.0, "Medium (3-4B)"),
        (5.0, 10.0, "Standard (7-8B)"),
        (10.0, 100.0, "Large (>10B)")
    ]
    
    seq_classifiers = [m for m in models if 'Seq. Classifier' in m['type']]
    
    for min_size, max_size, class_name in size_classes:
        class_models = [m for m in seq_classifiers 
                       if min_size <= m['size'] < max_size]
        if class_models:
            best = max(class_models, key=lambda x: x['score'])
            print(f"\n{class_name}:")
            print(f"  ⭐ {best['name']}")
            print(f"     Score: {best['score']:.2f} | Math: {best['math']:.2f} | "
                  f"Safety: {best['safety']:.2f}")
    
    # Top recommendations
    print("\n" + "=" * 80)
    print("TOP RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n⭐ RECOMMENDED (4B - Best Balance):")
    medium = [m for m in seq_classifiers if 3 <= m['size'] <= 5]
    if medium:
        best = max(medium, key=lambda x: x['score'])
        print(f"   {best['name']}")
        print(f"   Score: {best['score']:.2f} | Size: {best['size']:.1f}B")
    
    print("\n⭐ ELITE (8B - Maximum Performance):")
    standard = [m for m in seq_classifiers if 7 <= m['size'] <= 9]
    if standard:
        best = max(standard, key=lambda x: x['score'])
        print(f"   {best['name']}")
        print(f"   Score: {best['score']:.2f} | Size: {best['size']:.1f}B")
    
    print("\n⭐ BUDGET (0.6-1B - Baseline):")
    small = [m for m in seq_classifiers if m['size'] <= 1.0]
    if small:
        best = max(small, key=lambda x: x['score'])
        print(f"   {best['name']}")
        print(f"   Score: {best['score']:.2f} | Size: {best['size']:.1f}B")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
