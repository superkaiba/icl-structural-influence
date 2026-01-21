#!/usr/bin/env python3
"""
Unit test for hierarchical color encoding algorithm.

Tests the _hierarchy_path_to_color function to ensure it correctly maps
hierarchy paths to color values using recursive range subdivision.
"""

from src.visualization.animation import _hierarchy_path_to_color


def test_binary_tree():
    """Test with simple 2-level binary tree."""
    print("Test 1: Binary tree [2, 2]")
    branching = [2, 2]

    # Expected ranges and midpoints:
    # (0, 0): [0.0, 0.25] → midpoint 0.125
    # (0, 1): [0.25, 0.5] → midpoint 0.375
    # (1, 0): [0.5, 0.75] → midpoint 0.625
    # (1, 1): [0.75, 1.0] → midpoint 0.875

    paths = [(0, 0), (0, 1), (1, 0), (1, 1)]
    expected = [0.125, 0.375, 0.625, 0.875]

    for path, exp in zip(paths, expected):
        result = _hierarchy_path_to_color(path, branching)
        assert abs(result - exp) < 1e-6, f"Path {path}: expected {exp}, got {result}"
        print(f"  ✓ Path {path} → {result:.4f} (expected {exp:.4f})")

    print("  All assertions passed!\n")


def test_asymmetric_tree():
    """Test with asymmetric branching [2, 2, 4]."""
    print("Test 2: Asymmetric tree [2, 2, 4]")
    branching = [2, 2, 4]

    # Test a few key paths
    # Branching [2, 2, 4]:
    # Level 1: [0,1] / 2 → child 0: [0, 0.5], child 1: [0.5, 1]
    # Level 2 child 0: [0, 0.5] / 2 → child 0: [0, 0.25], child 1: [0.25, 0.5]
    # Level 3 (0,0): [0, 0.25] / 4 → child 0: [0, 0.0625], child 1: [0.0625, 0.125], ...
    test_cases = [
        ((0, 0, 0), 0.03125),   # [0.0, 0.0625] → midpoint 0.03125
        ((0, 0, 3), 0.21875),   # [0.1875, 0.25] → midpoint 0.21875
        ((1, 1, 3), 0.96875),   # [0.9375, 1.0] → midpoint 0.96875
        ((0, 1, 2), 0.40625),   # [0.375, 0.4375] → midpoint 0.40625
    ]

    for path, exp in test_cases:
        result = _hierarchy_path_to_color(path, branching)
        assert abs(result - exp) < 1e-6, f"Path {path}: expected {exp}, got {result}"
        print(f"  ✓ Path {path} → {result:.4f} (expected {exp:.4f})")

    print("  All assertions passed!\n")


def test_hierarchical_similarity():
    """Test that similar paths have similar color values."""
    print("Test 3: Hierarchical similarity")
    branching = [2, 3, 4]

    # Paths sharing level 1 parent should be closer than different parents
    path_a1 = (0, 0, 0)
    path_a2 = (0, 1, 2)  # Same super-cluster (0)
    path_b = (1, 0, 0)   # Different super-cluster (1)

    color_a1 = _hierarchy_path_to_color(path_a1, branching)
    color_a2 = _hierarchy_path_to_color(path_a2, branching)
    color_b = _hierarchy_path_to_color(path_b, branching)

    # Distance within same super-cluster
    dist_within = abs(color_a1 - color_a2)
    # Distance across super-clusters
    dist_across = abs(color_a1 - color_b)

    print(f"  Path {path_a1} → {color_a1:.4f}")
    print(f"  Path {path_a2} → {color_a2:.4f}")
    print(f"  Path {path_b} → {color_b:.4f}")
    print(f"  Distance within super-cluster: {dist_within:.4f}")
    print(f"  Distance across super-clusters: {dist_across:.4f}")

    # Nodes in same super-cluster should be closer than across super-clusters
    assert dist_within < dist_across, \
        f"Same super-cluster nodes should be closer! within={dist_within}, across={dist_across}"

    print("  ✓ Hierarchical similarity preserved!\n")


def test_five_level_hierarchy():
    """Test with 5-level hierarchy [2, 2, 2, 3, 3]."""
    print("Test 4: Five-level hierarchy [2, 2, 2, 3, 3]")
    branching = [2, 2, 2, 3, 3]

    # Test first and last nodes
    first_node = (0, 0, 0, 0, 0)
    last_node = (1, 1, 1, 2, 2)

    color_first = _hierarchy_path_to_color(first_node, branching)
    color_last = _hierarchy_path_to_color(last_node, branching)

    print(f"  First node {first_node} → {color_first:.6f}")
    print(f"  Last node {last_node} → {color_last:.6f}")

    # First node should be close to 0, last close to 1
    assert color_first < 0.1, f"First node should be near 0: {color_first}"
    assert color_last > 0.9, f"Last node should be near 1: {color_last}"

    # Test that color values are in valid range
    assert 0 <= color_first <= 1, f"Color out of range: {color_first}"
    assert 0 <= color_last <= 1, f"Color out of range: {color_last}"

    print("  ✓ Five-level hierarchy works correctly!\n")


def test_range_bounds():
    """Test that all colors stay within [0, 1] range."""
    print("Test 5: Range bounds")
    branching = [3, 5, 2]

    # Test all possible paths
    min_color = 1.0
    max_color = 0.0

    for i in range(3):
        for j in range(5):
            for k in range(2):
                path = (i, j, k)
                color = _hierarchy_path_to_color(path, branching)
                min_color = min(min_color, color)
                max_color = max(max_color, color)

                assert 0 <= color <= 1, \
                    f"Color out of bounds for path {path}: {color}"

    print(f"  Tested {3*5*2} paths")
    print(f"  Min color: {min_color:.4f}")
    print(f"  Max color: {max_color:.4f}")
    print("  ✓ All colors within [0, 1] range!\n")


def main():
    """Run all tests."""
    print("="*60)
    print("HIERARCHICAL COLOR ENCODING UNIT TESTS")
    print("="*60 + "\n")

    test_binary_tree()
    test_asymmetric_tree()
    test_hierarchical_similarity()
    test_five_level_hierarchy()
    test_range_bounds()

    print("="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    main()
