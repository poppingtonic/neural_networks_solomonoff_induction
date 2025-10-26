#!/usr/bin/env python3
"""Analyze CTW logs from AIXI implementation.

Reads and analyzes logs from: /home/muhia/src/aixi/aixictwxcode/mc-aixi-ctw/log/*.log
"""

import argparse
import glob
import re
from pathlib import Path
from typing import List, Dict, Any
import json


def parse_ctw_log(log_path: Path) -> Dict[str, Any]:
    """Parse a single CTW log file.
    
    Args:
        log_path: Path to log file
        
    Returns:
        Dictionary with parsed log data
    """
    data = {
        "path": str(log_path),
        "lines": [],
        "tree_info": [],
        "predictions": [],
        "errors": [],
    }
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                data["lines"].append(line.strip())
                
                # Extract tree information (example patterns - adjust based on actual log format)
                if "tree" in line.lower() or "depth" in line.lower():
                    data["tree_info"].append(line.strip())
                
                # Extract prediction information
                if "predict" in line.lower() or "prob" in line.lower():
                    data["predictions"].append(line.strip())
                
                # Extract errors
                if "error" in line.lower() or "exception" in line.lower():
                    data["errors"].append(line.strip())
    
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        data["errors"].append(f"Failed to read file: {e}")
    
    return data


def analyze_logs(log_dir: str) -> List[Dict[str, Any]]:
    """Analyze all CTW logs in directory.
    
    Args:
        log_dir: Directory containing log files
        
    Returns:
        List of parsed log data
    """
    log_pattern = str(Path(log_dir) / "*.log")
    log_files = glob.glob(log_pattern)
    
    print(f"Found {len(log_files)} log files in {log_dir}")
    
    all_data = []
    for log_path in sorted(log_files):
        print(f"Parsing: {log_path}")
        data = parse_ctw_log(Path(log_path))
        all_data.append(data)
    
    return all_data


def summarize_logs(all_data: List[Dict[str, Any]]):
    """Print summary of log analysis.
    
    Args:
        all_data: List of parsed log data
    """
    print("\n" + "="*60)
    print("CTW Log Analysis Summary")
    print("="*60)
    
    total_lines = sum(len(d["lines"]) for d in all_data)
    total_tree_info = sum(len(d["tree_info"]) for d in all_data)
    total_predictions = sum(len(d["predictions"]) for d in all_data)
    total_errors = sum(len(d["errors"]) for d in all_data)
    
    print(f"Total log files: {len(all_data)}")
    print(f"Total lines: {total_lines}")
    print(f"Tree info entries: {total_tree_info}")
    print(f"Prediction entries: {total_predictions}")
    print(f"Error entries: {total_errors}")
    
    # Show sample tree info
    if total_tree_info > 0:
        print("\nSample tree information:")
        for d in all_data:
            if d["tree_info"]:
                for info in d["tree_info"][:3]:
                    print(f"  {info}")
                break
    
    # Show sample predictions
    if total_predictions > 0:
        print("\nSample predictions:")
        for d in all_data:
            if d["predictions"]:
                for pred in d["predictions"][:3]:
                    print(f"  {pred}")
                break
    
    # Show errors
    if total_errors > 0:
        print("\nErrors found:")
        for d in all_data:
            if d["errors"]:
                print(f"  File: {d['path']}")
                for error in d["errors"][:5]:
                    print(f"    {error}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze CTW logs from AIXI implementation"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/muhia/src/aixi/aixictwxcode/mc-aixi-ctw/log",
        help="Directory containing CTW log files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for parsed data",
    )
    
    args = parser.parse_args(argv)
    
    # Analyze logs
    all_data = analyze_logs(args.log_dir)
    
    # Print summary
    summarize_logs(all_data)
    
    # Save to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_data, f, indent=2)
        print(f"\n✓ Saved parsed data to {args.output}")


if __name__ == "__main__":
    main()
