# Copyright 2025
# LSTM-CA Environment Generator for ARC-AGI-3 style grids
# Based on Zhou et al. 2024 "Learning Cellular Automata with LSTM"
# Integrates with M-ary CTW for controllable Markov complexity

"""
LSTM-CA Environment Generator

This module generates grid-based environments compatible with ARC-AGI-3:
- 64x64 max grid size
- 16 colors (0-15)
- Temporal sequences (frames)
- Action-state transitions

The LSTM-CA approach (Zhou et al. 2024) uses trained LSTMs as controllable
Markov sources. Temperature controls the effective Markov order:
- Low temp → deterministic, low-k Markov
- High temp → stochastic, high-k Markov
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Optional, Sequence, Literal
import copy

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# ARC-AGI-3 Color Palette (16 colors)
# =============================================================================

# Official ARC-AGI-3 color palette (approximate)
ARC_AGI_3_PALETTE = [
    "#000000",  # 0: Black
    "#0074D9",  # 1: Blue
    "#FF4136",  # 2: Red
    "#2ECC40",  # 3: Green
    "#FFDC00",  # 4: Yellow
    "#AAAAAA",  # 5: Gray
    "#F012BE",  # 6: Magenta/Pink
    "#FF851B",  # 7: Orange
    "#7FDBFF",  # 8: Light Blue/Cyan
    "#870C25",  # 9: Dark Red/Maroon
    "#00FFFF",  # 10: Cyan
    "#FFB6C1",  # 11: Light Pink
    "#800080",  # 12: Purple
    "#008080",  # 13: Teal
    "#C0C0C0",  # 14: Silver
    "#FFFFFF",  # 15: White
]


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclasses.dataclass
class GridConfig:
    """Configuration for ARC-AGI-3 style grids."""
    max_height: int = 64
    max_width: int = 64
    num_colors: int = 16  # M=16 for ARC-AGI-3
    default_height: int = 30
    default_width: int = 30
    background_color: int = 0  # Black


@dataclasses.dataclass
class LSTMCAConfig:
    """Configuration for LSTM-CA environment generation."""
    grid_config: GridConfig = dataclasses.field(default_factory=GridConfig)

    # LSTM checkpoint path (for trained model)
    checkpoint_path: Optional[str] = None

    # Markov complexity control
    temperature: float = 1.0  # Higher = more stochastic
    min_temperature: float = 0.1
    max_temperature: float = 2.0

    # Sequence generation
    context_length: int = 64  # Context window for LSTM
    max_steps: int = 100  # Max generation steps

    # Grid encoding
    use_separator_tokens: bool = True  # Row separators in sequence
    separator_token: int = 16  # Special token (beyond 0-15)

    # Random fallback (when no checkpoint)
    use_ctw_fallback: bool = True
    ctw_max_depth: int = 12
    ctw_spawn_prob: float = 0.5

    # Spatial pattern generation (alternative to CTW for 2D structure)
    # When True, generates grids with proper 2D spatial dependencies
    # using cellular automata rules and pattern-based generation
    use_spatial_patterns: bool = False
    spatial_pattern_types: tuple = (
        "cellular_automata",  # Game of Life style CA evolution
        "blocks",             # Regular block patterns
        "flood_fill",         # Connected regions
        "gradients",          # Smooth color gradients
        "noise",              # Coherent noise (Perlin-like)
    )

    # Hybrid mode: combine temporal (LSTM/CTW) with spatial structure
    # This gives grids that have BOTH types of dependencies
    use_hybrid_generation: bool = False
    hybrid_blend_mode: str = "modulate"  # "modulate", "add", "mask", "condition"
    hybrid_spatial_weight: float = 0.5  # Weight for spatial pattern in blend


@dataclasses.dataclass
class ARC3Environment:
    """
    ARC-AGI-3 compatible environment state.

    Matches the FrameResponse schema from arc3v1.yaml.
    """
    # Core grid data
    frame: np.ndarray  # Shape: (height, width), dtype: uint8, values 0-15

    # Metadata
    game_id: str = "lstm-ca-generated"
    guid: str = ""
    state: Literal["NOT_FINISHED", "WIN", "GAME_OVER"] = "NOT_FINISHED"
    score: int = 0
    win_score: int = 254
    available_actions: list[int] = dataclasses.field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6]
    )

    # Generation metadata (not in ARC-AGI-3 spec but useful)
    markov_order_estimate: Optional[int] = None
    temperature_used: Optional[float] = None
    ctw_tree_depth: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary (ARC-AGI-3 format)."""
        # Convert numpy types to native Python types for JSON serialization
        def to_python(x):
            if isinstance(x, (np.integer, np.int64, np.int32)):
                return int(x)
            if isinstance(x, (np.floating, np.float64, np.float32)):
                return float(x)
            if isinstance(x, np.ndarray):
                return x.tolist()
            return x

        return {
            "game_id": self.game_id,
            "guid": self.guid,
            "frame": [self.frame.tolist()],  # Wrapped in list per spec
            "state": self.state,
            "score": to_python(self.score),
            "win_score": to_python(self.win_score),
            "available_actions": [to_python(a) for a in self.available_actions],
            "action_input": {"id": 0, "data": {}},
            # Extended metadata
            "_metadata": {
                "markov_order_estimate": to_python(self.markov_order_estimate),
                "temperature_used": to_python(self.temperature_used),
                "ctw_tree_depth": to_python(self.ctw_tree_depth),
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ARC3Environment":
        """Load from JSON dictionary."""
        frame_data = data["frame"]
        # Handle wrapped frame format
        if isinstance(frame_data[0][0], list):
            frame_data = frame_data[0]

        metadata = data.get("_metadata", {})

        return cls(
            frame=np.array(frame_data, dtype=np.uint8),
            game_id=data.get("game_id", "unknown"),
            guid=data.get("guid", ""),
            state=data.get("state", "NOT_FINISHED"),
            score=data.get("score", 0),
            win_score=data.get("win_score", 254),
            available_actions=data.get("available_actions", [1, 2, 3, 4, 5, 6]),
            markov_order_estimate=metadata.get("markov_order_estimate"),
            temperature_used=metadata.get("temperature_used"),
            ctw_tree_depth=metadata.get("ctw_tree_depth"),
        )


# =============================================================================
# Grid Encoding/Decoding
# =============================================================================

def grid_to_sequence(
    grid: np.ndarray,
    use_separators: bool = True,
    separator_token: int = 16,
) -> np.ndarray:
    """
    Convert 2D grid to 1D sequence (row-major order).

    Args:
        grid: 2D array of shape (H, W) with values 0-15
        use_separators: If True, insert separator tokens between rows
        separator_token: Token value for row separators

    Returns:
        1D array representing the grid
    """
    height, width = grid.shape

    if use_separators:
        # Flatten with separators: row0 | sep | row1 | sep | ... | rowN
        parts = []
        for i, row in enumerate(grid):
            parts.append(row)
            if i < height - 1:
                parts.append(np.array([separator_token], dtype=grid.dtype))
        sequence = np.concatenate(parts)
    else:
        sequence = grid.flatten()

    return sequence


def sequence_to_grid(
    sequence: np.ndarray,
    height: int,
    width: int,
    use_separators: bool = True,
    separator_token: int = 16,
) -> np.ndarray:
    """
    Convert 1D sequence back to 2D grid.

    Args:
        sequence: 1D array from grid_to_sequence
        height: Target grid height
        width: Target grid width
        use_separators: If True, expect separator tokens between rows
        separator_token: Token value for row separators

    Returns:
        2D array of shape (height, width)
    """
    if use_separators:
        # Remove separators and reshape
        mask = sequence != separator_token
        flat = sequence[mask]
    else:
        flat = sequence

    # Truncate or pad to exact size
    expected_size = height * width
    if len(flat) < expected_size:
        flat = np.pad(flat, (0, expected_size - len(flat)), constant_values=0)
    else:
        flat = flat[:expected_size]

    return flat.reshape(height, width)


def estimate_sequence_markov_order(
    sequence: np.ndarray,
    max_order: int = 16,
    alphabet_size: int = 16,
) -> int:
    """
    Estimate the Markov order of a sequence using KT estimator log-loss.

    Uses k-gram context prediction with Krichevsky-Trofimov estimator
    (same as CTW leaf nodes). Lower log-loss at order k indicates
    the sequence is well-modeled by a k-th order Markov source.

    Args:
        sequence: 1D array of symbols
        max_order: Maximum order to test
        alphabet_size: Number of distinct symbols (M)

    Returns:
        Estimated Markov order (elbow in log-loss curve)
    """
    if len(sequence) < max_order + 10:
        return 0

    seq = sequence.astype(np.int32)
    log_losses = []

    for k in range(max_order + 1):
        # Build k-gram context counts and compute KT estimator log-loss
        # KT estimator: P(x|context) = (count(x, context) + 0.5) / (total(context) + M/2)
        context_counts = {}  # context -> array of symbol counts
        total_log_loss = 0.0
        num_predictions = 0

        for i in range(k, len(seq)):
            # Get k-gram context (empty tuple for k=0)
            context = tuple(seq[i-k:i]) if k > 0 else ()
            target = seq[i]

            # Initialize context if not seen
            if context not in context_counts:
                context_counts[context] = np.zeros(alphabet_size, dtype=np.float64)

            counts = context_counts[context]
            total = counts.sum()

            # KT estimator probability
            prob = (counts[target] + 0.5) / (total + alphabet_size / 2.0)
            total_log_loss -= np.log(prob + 1e-300)
            num_predictions += 1

            # Update counts
            counts[target] += 1

        # Average log-loss per symbol
        avg_log_loss = total_log_loss / max(num_predictions, 1)
        log_losses.append(avg_log_loss)

    # Find elbow: where improvement in log-loss becomes negligible
    log_losses = np.array(log_losses)
    improvements = -np.diff(log_losses)  # Negative because lower is better

    if len(improvements) == 0 or improvements.max() < 1e-6:
        return 0

    # Markov order is where improvement drops below 10% of max improvement
    threshold = improvements.max() * 0.1
    for k, imp in enumerate(improvements):
        if imp < threshold:
            return k

    return max_order


def estimate_grid_markov_order_2d(
    grid: np.ndarray,
    max_order: int = 3,
    alphabet_size: int = 16,
    neighborhood: str = "moore",
) -> dict:
    """
    Estimate Markov order of a 2D grid using spatial context.

    Unlike 1D estimation, this uses neighboring cells as context, which is
    more appropriate for grids with 2D spatial structure (like cellular automata).

    For order k:
    - k=0: No context, just prior distribution
    - k=1: Direct neighbors (4 von Neumann or 8 Moore)
    - k=2: Neighbors + neighbors-of-neighbors (2-hop radius)
    - k=3: 3-hop radius

    Args:
        grid: 2D array of shape (height, width) with values in [0, alphabet_size)
        max_order: Maximum spatial order to test (radius in cells)
        alphabet_size: Number of distinct symbols (M=16 for ARC-AGI-3)
        neighborhood: "moore" (8-connected) or "vonneumann" (4-connected)

    Returns:
        dict with:
        - estimated_order: Best Markov order
        - log_losses: List of log-loss per order
        - improvements: Log-loss improvement from k-1 to k
        - bits_per_symbol: Entropy in bits per symbol at each order
    """
    grid = np.asarray(grid, dtype=np.int32)
    if grid.ndim != 2:
        raise ValueError(f"Expected 2D grid, got shape {grid.shape}")

    height, width = grid.shape
    log_losses = []

    # Define neighbor offsets based on neighborhood type
    if neighborhood == "vonneumann":
        base_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connected
    else:  # moore
        base_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]  # 8-connected

    def get_neighborhood_offsets(order: int) -> list:
        """Get all cell offsets within 'order' hops."""
        if order == 0:
            return []

        # Use Chebyshev distance (max of abs differences) for Moore
        # Use Manhattan distance for von Neumann
        offsets = []
        for dy in range(-order, order + 1):
            for dx in range(-order, order + 1):
                if dy == 0 and dx == 0:
                    continue
                if neighborhood == "vonneumann":
                    dist = abs(dy) + abs(dx)
                else:  # moore (Chebyshev)
                    dist = max(abs(dy), abs(dx))
                if dist <= order:
                    offsets.append((dy, dx))
        # Sort for consistent context hashing
        return sorted(offsets)

    for k in range(max_order + 1):
        offsets = get_neighborhood_offsets(k)
        context_counts = {}
        total_log_loss = 0.0
        num_predictions = 0

        for y in range(height):
            for x in range(width):
                target = grid[y, x]

                # Build context from neighbors
                if k == 0:
                    context = ()
                else:
                    context_values = []
                    for dy, dx in offsets:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            context_values.append(grid[ny, nx])
                        else:
                            context_values.append(-1)  # Boundary marker
                    context = tuple(context_values)

                # Initialize context if not seen
                if context not in context_counts:
                    context_counts[context] = np.zeros(alphabet_size, dtype=np.float64)

                counts = context_counts[context]
                total = counts.sum()

                # KT estimator probability
                prob = (counts[target] + 0.5) / (total + alphabet_size / 2.0)
                total_log_loss -= np.log(prob + 1e-300)
                num_predictions += 1

                # Update counts
                counts[target] += 1

        avg_log_loss = total_log_loss / max(num_predictions, 1)
        log_losses.append(avg_log_loss)

    # Compute improvements and find elbow
    log_losses = np.array(log_losses)
    improvements = -np.diff(log_losses)  # Positive if log-loss decreased

    # Convert to bits per symbol
    bits_per_symbol = log_losses / np.log(2)

    # Find estimated order using elbow detection
    if len(improvements) == 0 or improvements.max() <= 0:
        estimated_order = 0
    else:
        threshold = improvements.max() * 0.1
        estimated_order = 0
        for k, imp in enumerate(improvements):
            if imp >= threshold:
                estimated_order = k + 1  # Order k+1 gave improvement from k
            else:
                break

    return {
        "estimated_order": estimated_order,
        "log_losses": log_losses.tolist(),
        "improvements": improvements.tolist() if len(improvements) > 0 else [],
        "bits_per_symbol": bits_per_symbol.tolist(),
        "neighborhood": neighborhood,
        "num_contexts_per_order": [
            len(get_neighborhood_offsets(k)) for k in range(max_order + 1)
        ],
    }


def estimate_grid_spatial_correlation(
    grid: np.ndarray,
    alphabet_size: int = 16,
) -> dict:
    """
    Estimate spatial correlation strength using simpler aggregated features.

    This avoids the data sparsity problem of full k-gram contexts by using
    aggregated features like "same as majority neighbor" or "average neighbor".

    Args:
        grid: 2D array of shape (height, width)
        alphabet_size: Number of distinct symbols

    Returns:
        dict with various correlation metrics
    """
    grid = np.asarray(grid, dtype=np.int32)
    height, width = grid.shape

    # Metrics to compute
    same_as_left = 0
    same_as_up = 0
    same_as_majority = 0
    total_interior = 0

    # Moore neighborhood
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(height):
        for x in range(width):
            current = grid[y, x]

            # Same as left neighbor
            if x > 0:
                same_as_left += int(grid[y, x - 1] == current)

            # Same as up neighbor
            if y > 0:
                same_as_up += int(grid[y - 1, x] == current)

            # Same as majority of 8 neighbors
            if y > 0 and y < height - 1 and x > 0 and x < width - 1:
                neighbors = [grid[y + dy, x + dx] for dy, dx in offsets]
                counts = np.bincount(neighbors, minlength=alphabet_size)
                majority = counts.argmax()
                same_as_majority += int(majority == current)
                total_interior += 1

    # Compute probabilities
    total_h = height * (width - 1)  # Horizontal pairs
    total_v = (height - 1) * width  # Vertical pairs

    # Expected random match probability
    random_match_prob = 1.0 / alphabet_size

    result = {
        "horizontal_correlation": same_as_left / max(1, total_h),
        "vertical_correlation": same_as_up / max(1, total_v),
        "majority_correlation": same_as_majority / max(1, total_interior),
        "random_baseline": random_match_prob,
        "has_spatial_structure": False,
    }

    # Check if correlations exceed random baseline significantly
    threshold = random_match_prob * 1.5  # 50% above random
    if (result["horizontal_correlation"] > threshold or
        result["vertical_correlation"] > threshold or
        result["majority_correlation"] > threshold):
        result["has_spatial_structure"] = True

    return result


# =============================================================================
# Spatial Pattern Generators (2D-aware)
# =============================================================================

def generate_cellular_automata_grid(
    height: int,
    width: int,
    num_colors: int,
    rng: np.random.Generator,
    num_steps: int = 5,
    rule_type: str = "majority",
) -> np.ndarray:
    """
    Generate a grid using cellular automata evolution.

    Creates grids with proper 2D spatial structure via CA rules:
    - majority: Cell becomes majority color of neighbors
    - totalistic: Cell color depends on sum of neighbor colors mod num_colors
    - game_of_life: Binary with birth/survive rules (generalized to M colors)

    Args:
        height, width: Grid dimensions
        num_colors: Number of colors (M)
        rng: Random generator
        num_steps: Number of CA evolution steps
        rule_type: CA rule to use

    Returns:
        2D grid with spatial structure
    """
    # Initialize with sparse random pattern
    grid = np.zeros((height, width), dtype=np.int32)

    # Seed with random cells (10-30% fill)
    num_seed = rng.integers(height * width // 10, height * width // 3)
    seed_y = rng.integers(0, height, size=num_seed)
    seed_x = rng.integers(0, width, size=num_seed)
    seed_colors = rng.integers(1, num_colors, size=num_seed)  # Non-background
    grid[seed_y, seed_x] = seed_colors

    # Moore neighborhood offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for _ in range(num_steps):
        new_grid = grid.copy()

        for y in range(height):
            for x in range(width):
                # Count neighbor colors
                neighbor_colors = []
                for dy, dx in offsets:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        neighbor_colors.append(grid[ny, nx])

                if not neighbor_colors:
                    continue

                if rule_type == "majority":
                    # Cell becomes most common neighbor color
                    counts = np.bincount(neighbor_colors, minlength=num_colors)
                    # Add current cell with weight
                    counts[grid[y, x]] += 2
                    new_grid[y, x] = counts.argmax()

                elif rule_type == "totalistic":
                    # Sum-based rule
                    total = sum(neighbor_colors) + grid[y, x]
                    new_grid[y, x] = total % num_colors

                elif rule_type == "game_of_life":
                    # Generalized GoL: birth if 3 live neighbors, survive if 2-3
                    live = sum(1 for c in neighbor_colors if c > 0)
                    if grid[y, x] == 0:  # Dead cell
                        if live == 3:
                            new_grid[y, x] = rng.integers(1, num_colors)
                    else:  # Live cell
                        if live < 2 or live > 3:
                            new_grid[y, x] = 0

        grid = new_grid

    return grid


def generate_block_pattern_grid(
    height: int,
    width: int,
    num_colors: int,
    rng: np.random.Generator,
    block_size_range: tuple = (2, 6),
) -> np.ndarray:
    """
    Generate grid with rectangular block patterns.

    Creates distinct rectangular regions with uniform colors,
    providing clear spatial structure.
    """
    grid = np.zeros((height, width), dtype=np.int32)

    # Generate random blocks
    num_blocks = rng.integers(5, 20)

    for _ in range(num_blocks):
        # Random block size and position
        bh = rng.integers(block_size_range[0], block_size_range[1] + 1)
        bw = rng.integers(block_size_range[0], block_size_range[1] + 1)
        by = rng.integers(0, height - bh + 1)
        bx = rng.integers(0, width - bw + 1)
        color = rng.integers(1, num_colors)  # Non-background

        grid[by:by + bh, bx:bx + bw] = color

    return grid


def generate_flood_fill_grid(
    height: int,
    width: int,
    num_colors: int,
    rng: np.random.Generator,
    num_regions: int = 8,
) -> np.ndarray:
    """
    Generate grid with connected flood-fill regions.

    Uses random walks from seed points to create organic
    connected regions with spatial coherence.
    """
    grid = np.zeros((height, width), dtype=np.int32)

    for region_id in range(num_regions):
        color = (region_id % (num_colors - 1)) + 1  # Cycle through colors

        # Random seed point
        y, x = rng.integers(0, height), rng.integers(0, width)

        # Random walk to fill region
        region_size = rng.integers(height * width // 20, height * width // 5)

        for _ in range(region_size):
            if 0 <= y < height and 0 <= x < width:
                grid[y, x] = color

            # Random walk step (biased towards staying in bounds)
            dy, dx = rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
            y = max(0, min(height - 1, y + dy))
            x = max(0, min(width - 1, x + dx))

    return grid


def generate_gradient_grid(
    height: int,
    width: int,
    num_colors: int,
    rng: np.random.Generator,
    gradient_type: str = "radial",
) -> np.ndarray:
    """
    Generate grid with smooth color gradients.

    Creates grids where nearby cells have similar colors,
    providing strong local spatial correlations.
    """
    grid = np.zeros((height, width), dtype=np.int32)

    if gradient_type == "radial":
        # Radial gradient from random center
        cy, cx = rng.integers(0, height), rng.integers(0, width)
        max_dist = np.sqrt(height**2 + width**2) / 2

        for y in range(height):
            for x in range(width):
                dist = np.sqrt((y - cy)**2 + (x - cx)**2)
                # Map distance to color (with noise)
                t = dist / max_dist
                t = min(1.0, t + rng.uniform(-0.1, 0.1))
                grid[y, x] = int(t * (num_colors - 1))

    elif gradient_type == "linear":
        # Linear gradient along random direction
        angle = rng.uniform(0, 2 * np.pi)
        dy, dx = np.sin(angle), np.cos(angle)

        for y in range(height):
            for x in range(width):
                t = (y * dy + x * dx) / (height * abs(dy) + width * abs(dx) + 1)
                t = (t + 1) / 2  # Normalize to [0, 1]
                t = min(1.0, max(0.0, t + rng.uniform(-0.1, 0.1)))
                grid[y, x] = int(t * (num_colors - 1))

    elif gradient_type == "diagonal":
        # Simple diagonal gradient
        for y in range(height):
            for x in range(width):
                t = (y + x) / (height + width)
                t = min(1.0, max(0.0, t + rng.uniform(-0.1, 0.1)))
                grid[y, x] = int(t * (num_colors - 1))

    return grid


def generate_coherent_noise_grid(
    height: int,
    width: int,
    num_colors: int,
    rng: np.random.Generator,
    scale: float = 8.0,
) -> np.ndarray:
    """
    Generate grid with coherent (Perlin-like) noise.

    Simple coherent noise via interpolated random grid,
    creating smooth spatial variations.
    """
    # Create coarse random grid
    coarse_h = max(2, int(height / scale))
    coarse_w = max(2, int(width / scale))
    coarse = rng.uniform(0, 1, size=(coarse_h + 1, coarse_w + 1))

    # Interpolate to full resolution
    grid = np.zeros((height, width), dtype=np.float64)

    for y in range(height):
        for x in range(width):
            # Map to coarse grid coordinates
            cy = y / height * coarse_h
            cx = x / width * coarse_w

            # Bilinear interpolation
            y0, x0 = int(cy), int(cx)
            y1, x1 = min(y0 + 1, coarse_h), min(x0 + 1, coarse_w)
            fy, fx = cy - y0, cx - x0

            v00 = coarse[y0, x0]
            v01 = coarse[y0, x1]
            v10 = coarse[y1, x0]
            v11 = coarse[y1, x1]

            v0 = v00 * (1 - fx) + v01 * fx
            v1 = v10 * (1 - fx) + v11 * fx
            grid[y, x] = v0 * (1 - fy) + v1 * fy

    # Quantize to colors
    return (grid * (num_colors - 1)).astype(np.int32)


def generate_spatial_pattern(
    height: int,
    width: int,
    num_colors: int,
    rng: np.random.Generator,
    pattern_type: Optional[str] = None,
    temperature: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """
    Generate a grid with specified spatial pattern type.

    Args:
        height, width: Grid dimensions
        num_colors: Number of colors
        rng: Random generator
        pattern_type: Pattern type or None for random selection
        temperature: Controls pattern complexity/randomness

    Returns:
        Tuple of (grid, metadata dict)
    """
    pattern_types = [
        "cellular_automata",
        "blocks",
        "flood_fill",
        "gradients",
        "noise",
    ]

    if pattern_type is None:
        pattern_type = rng.choice(pattern_types)

    metadata = {"pattern_type": pattern_type, "temperature": temperature}

    if pattern_type == "cellular_automata":
        rule = rng.choice(["majority", "totalistic", "game_of_life"])
        steps = max(1, int(5 * temperature))
        grid = generate_cellular_automata_grid(
            height, width, num_colors, rng, num_steps=steps, rule_type=rule
        )
        metadata["ca_rule"] = rule
        metadata["ca_steps"] = steps

    elif pattern_type == "blocks":
        # Larger blocks at low temp, smaller at high temp
        min_size = max(2, int(6 - 3 * temperature))
        max_size = max(min_size + 1, int(8 - 2 * temperature))
        grid = generate_block_pattern_grid(
            height, width, num_colors, rng, block_size_range=(min_size, max_size)
        )
        metadata["block_size_range"] = (min_size, max_size)

    elif pattern_type == "flood_fill":
        # More regions at higher temp
        num_regions = max(3, int(5 + 8 * temperature))
        grid = generate_flood_fill_grid(
            height, width, num_colors, rng, num_regions=num_regions
        )
        metadata["num_regions"] = num_regions

    elif pattern_type == "gradients":
        grad_type = rng.choice(["radial", "linear", "diagonal"])
        grid = generate_gradient_grid(
            height, width, num_colors, rng, gradient_type=grad_type
        )
        metadata["gradient_type"] = grad_type

    elif pattern_type == "noise":
        # Smaller scale (more detail) at high temp
        scale = max(2.0, 12.0 - 6.0 * temperature)
        grid = generate_coherent_noise_grid(
            height, width, num_colors, rng, scale=scale
        )
        metadata["noise_scale"] = scale

    else:
        # Fallback: random grid
        grid = rng.integers(0, num_colors, size=(height, width), dtype=np.int32)
        metadata["pattern_type"] = "random"

    return grid, metadata


def blend_temporal_spatial(
    temporal_grid: np.ndarray,
    spatial_grid: np.ndarray,
    num_colors: int,
    blend_mode: str = "modulate",
    spatial_weight: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, dict]:
    """
    Blend temporal (LSTM/CTW) and spatial pattern grids.

    This creates grids with BOTH temporal sequence structure AND
    2D spatial structure by combining outputs from both generators.

    Blend modes:
    - "modulate": spatial_grid modulates temporal_grid colors
    - "add": weighted sum of both grids (mod num_colors)
    - "mask": spatial_grid provides regions, temporal fills them
    - "condition": spatial_grid biases temporal predictions
    - "interleave": alternating cells from each source

    Args:
        temporal_grid: Grid from LSTM/CTW (has temporal structure)
        spatial_grid: Grid from spatial pattern (has 2D structure)
        num_colors: Number of colors (M)
        blend_mode: How to combine the grids
        spatial_weight: Weight for spatial pattern (0-1)
        rng: Random generator for stochastic blending

    Returns:
        Tuple of (blended_grid, metadata)
    """
    if rng is None:
        rng = np.random.default_rng()

    height, width = temporal_grid.shape
    metadata = {
        "blend_mode": blend_mode,
        "spatial_weight": spatial_weight,
    }

    if blend_mode == "modulate":
        # Spatial grid modulates temporal: output = (temp + spatial_offset) % M
        # spatial_grid values act as offsets that preserve temporal transitions
        # but add spatial structure
        offset = (spatial_grid * spatial_weight).astype(np.int32)
        blended = (temporal_grid + offset) % num_colors

    elif blend_mode == "add":
        # Weighted sum: output = (w1 * temp + w2 * spatial) % M
        temp_weight = 1.0 - spatial_weight
        weighted_sum = (
            temporal_grid * temp_weight + spatial_grid * spatial_weight
        )
        blended = (weighted_sum.astype(np.int32)) % num_colors

    elif blend_mode == "mask":
        # Spatial grid provides mask, temporal fills non-background regions
        # Background cells (0) use temporal, non-background use spatial structure
        mask = spatial_grid > 0
        blended = np.where(mask, spatial_grid, temporal_grid)
        metadata["mask_coverage"] = float(mask.sum()) / mask.size

    elif blend_mode == "condition":
        # Spatial grid biases the color choice from temporal
        # output = temporal if random > spatial_influence, else spatial
        influence = spatial_grid / max(1, num_colors - 1)  # Normalize to [0,1]
        random_vals = rng.random(size=(height, width))
        use_spatial = random_vals < (influence * spatial_weight)
        blended = np.where(use_spatial, spatial_grid, temporal_grid)
        metadata["spatial_cells"] = float(use_spatial.sum()) / use_spatial.size

    elif blend_mode == "interleave":
        # Checkerboard interleaving of spatial and temporal
        checker = np.zeros((height, width), dtype=bool)
        for y in range(height):
            for x in range(width):
                checker[y, x] = (y + x) % 2 == 0
        # Spatial weight controls probability of using spatial
        if spatial_weight != 0.5:
            random_flip = rng.random(size=(height, width)) < abs(spatial_weight - 0.5)
            if spatial_weight > 0.5:
                checker = checker | random_flip
            else:
                checker = checker & ~random_flip
        blended = np.where(checker, spatial_grid, temporal_grid)

    elif blend_mode == "layered":
        # Spatial provides base layer, temporal provides overlay detail
        # Non-zero temporal values override spatial
        blended = np.where(temporal_grid > 0, temporal_grid, spatial_grid)

    elif blend_mode == "smooth":
        # Smooth transition: blend colors in local windows
        blended = np.zeros_like(temporal_grid)
        for y in range(height):
            for x in range(width):
                # Local blend factor based on spatial structure
                local_weight = spatial_weight
                if y > 0 and spatial_grid[y, x] == spatial_grid[y - 1, x]:
                    local_weight += 0.2  # Increase spatial weight in uniform regions
                if x > 0 and spatial_grid[y, x] == spatial_grid[y, x - 1]:
                    local_weight += 0.2
                local_weight = min(1.0, local_weight)

                if rng.random() < local_weight:
                    blended[y, x] = spatial_grid[y, x]
                else:
                    blended[y, x] = temporal_grid[y, x]

    else:
        # Unknown mode: fall back to simple average
        blended = ((temporal_grid + spatial_grid) // 2) % num_colors
        metadata["blend_mode"] = "average_fallback"

    return blended.astype(np.int32), metadata


# =============================================================================
# LSTM-CA Generator
# =============================================================================

class LSTMCAEnvironmentGenerator:
    """
    Generate ARC-AGI-3 style grid environments using LSTM-CA.

    Uses trained LSTM models as controllable Markov sources.
    Falls back to M-ary CTW when no checkpoint is available.
    """

    def __init__(
        self,
        config: Optional[LSTMCAConfig] = None,
        rng: Optional[int | np.random.Generator] = None,
    ):
        """
        Initialize LSTM-CA generator.

        Args:
            config: Generator configuration
            rng: Random seed or numpy Generator
        """
        self.config = config or LSTMCAConfig()

        if isinstance(rng, int):
            self._rng = np.random.default_rng(rng)
        elif rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

        # Load LSTM model if checkpoint provided
        self._model = None
        self._device = None
        if self.config.checkpoint_path and HAS_TORCH:
            self._load_lstm_checkpoint()

        # Initialize CTW fallback
        self._ctw_generator = None
        if self.config.use_ctw_fallback:
            self._init_ctw_fallback()

    def _load_lstm_checkpoint(self) -> None:
        """Load trained LSTM model from checkpoint."""
        if not HAS_TORCH:
            print("Warning: PyTorch not available, using CTW fallback")
            return

        try:
            checkpoint = torch.load(
                self.config.checkpoint_path,
                map_location="cpu"
            )

            # Import LSTM model
            from torch_models.lstm import LSTMDecoderLM, LSTMConfig

            model_config = LSTMConfig(**checkpoint.get("config", {}))
            self._model = LSTMDecoderLM(model_config)
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.eval()

            # Use GPU if available
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._model.to(self._device)

            print(f"Loaded LSTM from {self.config.checkpoint_path}")

        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            self._model = None

    def _init_ctw_fallback(self) -> None:
        """Initialize M-ary CTW generator as fallback."""
        try:
            from data.mary_ctw_data_generator import (
                MaryCTWGenerator,
                MaryCTWConfig,
            )

            ctw_config = MaryCTWConfig(
                alphabet_size=self.config.grid_config.num_colors,
                max_depth=self.config.ctw_max_depth,
                spawn_prob=self.config.ctw_spawn_prob,
                dirichlet_alpha=0.5,
                sparse_symbols=True,
            )

            grid_cfg = self.config.grid_config
            seq_len = grid_cfg.default_height * grid_cfg.default_width
            if self.config.use_separator_tokens:
                seq_len += grid_cfg.default_height - 1  # Row separators

            self._ctw_generator = MaryCTWGenerator(
                batch_size=1,
                seq_length=seq_len,
                rng=self._rng,
                config=ctw_config,
            )

        except ImportError as e:
            print(f"Warning: Could not initialize CTW fallback: {e}")

    def _sample_with_lstm(
        self,
        height: int,
        width: int,
        temperature: float,
    ) -> np.ndarray:
        """Generate sequence using LSTM model."""
        if not HAS_TORCH or self._model is None:
            raise RuntimeError("LSTM model not available")

        seq_len = height * width
        if self.config.use_separator_tokens:
            seq_len += height - 1

        # Start with random seed
        seed_len = min(16, seq_len // 4)
        seed = self._rng.integers(
            0, self.config.grid_config.num_colors, size=seed_len
        )

        sequence = list(seed)

        with torch.no_grad():
            for _ in range(seq_len - seed_len):
                # Prepare input
                context = sequence[-self.config.context_length:]
                x = torch.tensor([context], dtype=torch.long, device=self._device)

                # Get logits
                logits = self._model(x)
                logits = logits[0, -1, :self.config.grid_config.num_colors]

                # Apply temperature
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

                # Sample
                next_token = self._rng.choice(
                    self.config.grid_config.num_colors,
                    p=probs
                )
                sequence.append(next_token)

        return np.array(sequence, dtype=np.uint8)

    def _sample_with_ctw(
        self,
        height: int,
        width: int,
        temperature: float,
    ) -> tuple[np.ndarray, dict]:
        """Generate sequence using M-ary CTW."""
        if self._ctw_generator is None:
            raise RuntimeError("CTW generator not available")

        # Reseed for this generation
        seq_len = height * width
        if self.config.use_separator_tokens:
            seq_len += height - 1

        # Adjust CTW parameters based on temperature
        # Lower temp = deeper trees (more deterministic)
        adjusted_depth = int(self.config.ctw_max_depth * (2.0 - temperature))
        adjusted_depth = max(4, min(16, adjusted_depth))

        # Generate using CTW
        from data.mary_ctw_data_generator import MaryCTWConfig, MaryCTWGenerator

        temp_config = MaryCTWConfig(
            alphabet_size=self.config.grid_config.num_colors,
            max_depth=adjusted_depth,
            spawn_prob=self.config.ctw_spawn_prob * temperature,
            dirichlet_alpha=0.5 * temperature,
            sparse_symbols=True,
        )

        temp_gen = MaryCTWGenerator(
            batch_size=1,
            seq_length=seq_len,
            rng=self._rng,
            config=temp_config,
        )

        sequences, metadata = temp_gen.sample()
        return sequences[0], metadata

    def generate(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        temperature: Optional[float] = None,
        game_id: Optional[str] = None,
    ) -> ARC3Environment:
        """
        Generate a single ARC-AGI-3 style environment.

        Args:
            height: Grid height (default from config)
            width: Grid width (default from config)
            temperature: Markov complexity control (default from config)
            game_id: Optional game identifier

        Returns:
            ARC3Environment with generated grid
        """
        height = height or self.config.grid_config.default_height
        width = width or self.config.grid_config.default_width
        temperature = temperature or self.config.temperature

        # Clamp dimensions
        height = min(height, self.config.grid_config.max_height)
        width = min(width, self.config.grid_config.max_width)

        # Clamp temperature
        temperature = max(
            self.config.min_temperature,
            min(self.config.max_temperature, temperature)
        )

        # Generate grid (different methods based on config)
        metadata = {}
        sequence = None

        if self.config.use_hybrid_generation:
            # HYBRID MODE: Combine temporal (LSTM/CTW) with spatial structure
            # Step 1: Generate temporal grid
            if self._model is not None:
                temporal_seq = self._sample_with_lstm(height, width, temperature)
                temporal_meta = {"source": "lstm"}
            else:
                temporal_seq, temporal_meta = self._sample_with_ctw(height, width, temperature)
                temporal_meta["source"] = "ctw"

            temporal_grid = sequence_to_grid(
                temporal_seq,
                height,
                width,
                use_separators=self.config.use_separator_tokens,
                separator_token=self.config.separator_token,
            )

            # Step 2: Generate spatial pattern grid
            pattern_type = None
            if self.config.spatial_pattern_types:
                pattern_type = self._rng.choice(self.config.spatial_pattern_types)
            spatial_grid, spatial_meta = generate_spatial_pattern(
                height,
                width,
                self.config.grid_config.num_colors,
                self._rng,
                pattern_type=pattern_type,
                temperature=temperature,
            )

            # Step 3: Blend temporal and spatial
            grid, blend_meta = blend_temporal_spatial(
                temporal_grid,
                spatial_grid,
                self.config.grid_config.num_colors,
                blend_mode=self.config.hybrid_blend_mode,
                spatial_weight=self.config.hybrid_spatial_weight,
                rng=self._rng,
            )

            # Combine metadata
            metadata = {
                "generation_mode": "hybrid",
                "temporal": temporal_meta,
                "spatial": spatial_meta,
                "blend": blend_meta,
            }

            # Create sequence from blended grid
            sequence = grid_to_sequence(
                grid,
                use_separators=self.config.use_separator_tokens,
                separator_token=self.config.separator_token,
            )

        elif self._model is not None:
            # LSTM-CA model available - use it
            sequence = self._sample_with_lstm(height, width, temperature)
            grid = sequence_to_grid(
                sequence,
                height,
                width,
                use_separators=self.config.use_separator_tokens,
                separator_token=self.config.separator_token,
            )
            metadata["generation_mode"] = "lstm"

        elif self.config.use_spatial_patterns:
            # Generate grid with proper 2D spatial structure only
            pattern_type = None  # Random selection
            if self.config.spatial_pattern_types:
                pattern_type = self._rng.choice(self.config.spatial_pattern_types)
            grid, metadata = generate_spatial_pattern(
                height,
                width,
                self.config.grid_config.num_colors,
                self._rng,
                pattern_type=pattern_type,
                temperature=temperature,
            )
            metadata["generation_mode"] = "spatial_only"
            # Create sequence from grid for compatibility
            sequence = grid_to_sequence(
                grid,
                use_separators=self.config.use_separator_tokens,
                separator_token=self.config.separator_token,
            )

        else:
            # CTW fallback (1D temporal structure, no 2D spatial structure)
            sequence, metadata = self._sample_with_ctw(height, width, temperature)
            metadata["generation_mode"] = "ctw_only"
            grid = sequence_to_grid(
                sequence,
                height,
                width,
                use_separators=self.config.use_separator_tokens,
                separator_token=self.config.separator_token,
            )

        # Ensure values are in range
        grid = np.clip(grid, 0, self.config.grid_config.num_colors - 1)

        # Estimate Markov order - use both 1D and 2D for hybrid mode
        if self.config.use_hybrid_generation or self.config.use_spatial_patterns:
            # Remove separator tokens before 1D estimation
            seq_no_sep = sequence[sequence < self.config.grid_config.num_colors]
            # Compute both 1D (temporal) and 2D (spatial) Markov orders
            result_1d = estimate_sequence_markov_order(
                seq_no_sep,
                max_order=min(16, self.config.ctw_max_depth),
                alphabet_size=self.config.grid_config.num_colors,
            )
            result_2d = estimate_grid_markov_order_2d(
                grid,
                max_order=3,
                alphabet_size=self.config.grid_config.num_colors,
            )
            spatial_corr = estimate_grid_spatial_correlation(
                grid,
                alphabet_size=self.config.grid_config.num_colors,
            )

            # For hybrid, report both; use max as primary estimate
            markov_order = max(result_1d, result_2d["estimated_order"])
            metadata["markov_order_1d"] = result_1d
            metadata["markov_order_2d"] = result_2d
            metadata["spatial_correlation"] = spatial_corr
        else:
            markov_order = estimate_sequence_markov_order(
                sequence,
                max_order=min(16, self.config.ctw_max_depth),
                alphabet_size=self.config.grid_config.num_colors,
            )

        # Create environment
        import uuid
        env = ARC3Environment(
            frame=grid.astype(np.uint8),
            game_id=game_id or f"lstm-ca-{self._rng.integers(10000):04d}",
            guid=str(uuid.uuid4()),
            state="NOT_FINISHED",
            score=0,
            win_score=254,
            available_actions=[1, 2, 3, 4, 5, 6],
            markov_order_estimate=markov_order,
            temperature_used=temperature,
            ctw_tree_depth=metadata.get("tree_depths", [None])[0] if metadata else None,
        )

        return env

    def generate_batch(
        self,
        batch_size: int,
        **kwargs,
    ) -> list[ARC3Environment]:
        """Generate multiple environments."""
        return [self.generate(**kwargs) for _ in range(batch_size)]

    def generate_sequence(
        self,
        num_frames: int,
        height: Optional[int] = None,
        width: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> list[ARC3Environment]:
        """
        Generate a sequence of related frames (temporal evolution).

        Each frame is generated with slight variations to simulate
        environment evolution over time.
        """
        frames = []
        base_temp = temperature or self.config.temperature

        for i in range(num_frames):
            # Vary temperature slightly for temporal variation
            temp_variation = self._rng.normal(0, 0.1)
            frame_temp = base_temp + temp_variation

            env = self.generate(
                height=height,
                width=width,
                temperature=frame_temp,
                game_id=f"lstm-ca-seq-{i:03d}",
            )
            frames.append(env)

        return frames


# =============================================================================
# Visualization
# =============================================================================

def create_colormap(palette: list[str] = ARC_AGI_3_PALETTE) -> Any:
    """Create matplotlib colormap from palette."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for visualization")

    return mcolors.ListedColormap(palette)


def visualize_grid(
    env: ARC3Environment,
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    show_grid_lines: bool = True,
    cell_size: float = 0.5,
) -> Any:
    """
    Visualize a single ARC-AGI-3 style grid.

    Args:
        env: Environment to visualize
        ax: Matplotlib axes (created if None)
        title: Optional title
        show_grid_lines: Whether to show grid lines
        cell_size: Size of each cell in inches

    Returns:
        Matplotlib axes
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for visualization")

    grid = env.frame
    height, width = grid.shape

    if ax is None:
        fig_width = width * cell_size
        fig_height = height * cell_size
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = create_colormap()

    # Plot grid
    ax.imshow(
        grid,
        cmap=cmap,
        vmin=0,
        vmax=15,
        interpolation='nearest',
        aspect='equal',
    )

    if show_grid_lines:
        # Draw grid lines
        for i in range(height + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        for j in range(width + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.5)

    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title)
    elif env.game_id:
        ax.set_title(f"{env.game_id} (k≈{env.markov_order_estimate})")

    return ax


def visualize_comparison(
    env1: ARC3Environment,
    env2: ARC3Environment,
    title1: str = "Before",
    title2: str = "After",
    figsize: tuple[float, float] = (12, 5),
) -> Any:
    """
    Visualize two grids side by side (input/output style).
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for visualization")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    visualize_grid(env1, ax=ax1, title=title1)
    visualize_grid(env2, ax=ax2, title=title2)

    plt.tight_layout()
    return fig


def visualize_sequence(
    envs: list[ARC3Environment],
    cols: int = 4,
    cell_size: float = 0.4,
) -> Any:
    """
    Visualize a sequence of frames in a grid layout.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for visualization")

    n = len(envs)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * envs[0].frame.shape[1] * cell_size,
                 rows * envs[0].frame.shape[0] * cell_size),
    )

    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]

    for i, env in enumerate(envs):
        row = i // cols
        col = i % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        visualize_grid(env, ax=ax, title=f"Frame {i}")

    # Hide unused axes
    for i in range(n, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        ax.axis('off')

    plt.tight_layout()
    return fig


# =============================================================================
# JSON Export/Import
# =============================================================================

def save_environment(env: ARC3Environment, path: str | Path) -> None:
    """Save environment to JSON file."""
    path = Path(path)
    with open(path, 'w') as f:
        json.dump(env.to_dict(), f, indent=2)


def load_environment(path: str | Path) -> ARC3Environment:
    """Load environment from JSON file."""
    path = Path(path)
    with open(path, 'r') as f:
        data = json.load(f)
    return ARC3Environment.from_dict(data)


def save_batch(envs: list[ARC3Environment], directory: str | Path) -> list[Path]:
    """Save multiple environments to a directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    paths = []
    for i, env in enumerate(envs):
        path = directory / f"env_{i:04d}.json"
        save_environment(env, path)
        paths.append(path)

    # Save manifest
    manifest = {
        "num_environments": len(envs),
        "files": [str(p.name) for p in paths],
        "grid_sizes": [(e.frame.shape[0], e.frame.shape[1]) for e in envs],
        "markov_orders": [e.markov_order_estimate for e in envs],
    }
    with open(directory / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    return paths


def load_batch(directory: str | Path) -> list[ARC3Environment]:
    """Load all environments from a directory."""
    directory = Path(directory)

    # Try to load from manifest
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        return [
            load_environment(directory / fname)
            for fname in manifest["files"]
        ]

    # Fallback: glob for JSON files
    return [
        load_environment(p)
        for p in sorted(directory.glob("env_*.json"))
    ]


# =============================================================================
# CLI Demo
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ARC-AGI-3 style grid environments"
    )
    parser.add_argument(
        "--num", type=int, default=4,
        help="Number of environments to generate"
    )
    parser.add_argument(
        "--height", type=int, default=10,
        help="Grid height"
    )
    parser.add_argument(
        "--width", type=int, default=10,
        help="Grid width"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Markov complexity (0.1-2.0)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./generated_environments",
        help="Output directory for saved environments"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Show visualization"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    print("LSTM-CA Environment Generator Demo")
    print("=" * 50)

    # Create generator
    config = LSTMCAConfig(
        grid_config=GridConfig(
            default_height=args.height,
            default_width=args.width,
        ),
        temperature=args.temperature,
    )

    generator = LSTMCAEnvironmentGenerator(config=config, rng=args.seed)

    # Generate environments
    print(f"\nGenerating {args.num} environments...")
    envs = generator.generate_batch(
        batch_size=args.num,
        height=args.height,
        width=args.width,
        temperature=args.temperature,
    )

    # Print info
    for i, env in enumerate(envs):
        print(f"  {i}: {env.game_id}, shape={env.frame.shape}, "
              f"k≈{env.markov_order_estimate}, temp={env.temperature_used:.2f}")

    # Save
    output_dir = Path(args.output_dir)
    print(f"\nSaving to {output_dir}...")
    paths = save_batch(envs, output_dir)
    print(f"  Saved {len(paths)} environments")

    # Visualize
    if args.visualize and HAS_MATPLOTLIB:
        print("\nDisplaying visualization...")
        fig = visualize_sequence(envs, cols=min(4, args.num))
        plt.savefig(output_dir / "preview.png", dpi=150, bbox_inches='tight')
        plt.show()
    elif args.visualize:
        print("Warning: matplotlib not available for visualization")

    print("\nDone!")
