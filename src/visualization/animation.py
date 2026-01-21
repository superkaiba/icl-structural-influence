"""
Animation utilities for visualizing representation evolution.

Provides GIF/video generation functions to show how token representations
evolve across context lengths, revealing the dynamics of structural learning.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, Literal
import warnings


def compute_hierarchical_color_values(
    graph,
    nodes: list[int]
) -> np.ndarray:
    """
    Compute hierarchical color values for nodes using full hierarchy path.

    Maps each node's complete hierarchy path to a color value in [0, 1] using
    recursive range subdivision. Nodes closer in the hierarchy tree get more
    similar color values.

    Args:
        graph: DeepHierarchicalGraph instance with hierarchy_paths attribute
        nodes: List of node indices

    Returns:
        Array of color values in [0, 1] for each node

    Example:
        >>> colors = compute_hierarchical_color_values(graph, [0, 1, 2])
        >>> cmap = plt.cm.twilight
        >>> rgb_colors = [cmap(c) for c in colors]
    """
    branching_factors = graph.config.branching_factors
    color_values = np.zeros(len(nodes))

    for i, node in enumerate(nodes):
        path = graph.hierarchy_paths[node]
        color_values[i] = _hierarchy_path_to_color(path, branching_factors)

    return color_values


def _hierarchy_path_to_color(path: tuple, branching_factors: list[int]) -> float:
    """
    Convert hierarchy path to color value in [0,1] via recursive subdivision.

    Each level of the hierarchy subdivides the color range. Nodes sharing
    common ancestors have overlapping color ranges, ensuring hierarchical
    similarity is preserved visually.

    Args:
        path: Hierarchy path tuple (e.g., (0, 1, 2) for 3-level hierarchy)
        branching_factors: Branching factors at each level (e.g., [2, 2, 4])

    Returns:
        Color value in [0, 1] representing midpoint of node's color range

    Example:
        >>> # Path (0, 1, 2) with branching [2, 2, 4]
        >>> # Level 1: cluster 0 → [0.0, 0.5]
        >>> # Level 2: cluster 1 → [0.25, 0.5]
        >>> # Level 3: cluster 2 → [0.3125, 0.375]
        >>> _hierarchy_path_to_color((0, 1, 2), [2, 2, 4])
        0.34375  # Midpoint of [0.3125, 0.375]
    """
    color_value = 0.0
    range_size = 1.0

    for cluster_id, branching_factor in zip(path, branching_factors):
        range_size /= branching_factor
        color_value += cluster_id * range_size

    # Return midpoint of final range for better color distribution
    return color_value + range_size / 2


def create_representation_evolution_gif(
    token_reps_by_context: dict[int, dict[str, np.ndarray]],
    graph,  # DeepHierarchicalGraph instance
    context_lengths: list[int],
    output_path: Union[str, Path],
    method: Literal["mds", "pca", "umap", "tsne"] = "mds",
    fps: int = 2,
    tight_bounds: bool = True,
    color_by_level: Optional[int] = None,
    hierarchical_colors: bool = False,
    colormap: str = 'twilight',
    figsize: tuple = (10, 8),
):
    """
    Create GIF showing how token representations evolve over context length.

    The animation reveals:
    - How clusters form with increasing context
    - Whether hierarchical structure emerges gradually
    - Which tokens move together (same hierarchy level)

    Key design features:
    - **Consistent embedding**: All representations projected once into 2D space
    - **Tight bounding box**: Graph axes automatically sized to fit all points
    - **Color by hierarchy**: Nodes colored by cluster at specified level

    Args:
        token_reps_by_context: Dict mapping context_length -> {token: representation}
            where representation is a numpy array of shape (hidden_dim,)
        graph: DeepHierarchicalGraph instance for hierarchy information
        context_lengths: List of context lengths to include in animation (ordered)
        output_path: Path to save GIF file (e.g., "evolution.gif")
        method: Dimensionality reduction method:
            - "mds": Multidimensional Scaling (preserves pairwise distances)
            - "pca": Principal Component Analysis (fast, linear)
            - "umap": UMAP (preserves local + global structure, requires umap-learn)
            - "tsne": t-SNE (preserves local structure, slower)
        fps: Frames per second in output GIF
        tight_bounds: If True, compute minimal bounding box for all points
        color_by_level: Which hierarchy level to use for coloring (1, 2, 3, ...)
            Used only when hierarchical_colors=False. If None with hierarchical_colors=False,
            raises ValueError.
        hierarchical_colors: If True, use full hierarchy path for coloring with continuous
            colormap. If False, use categorical colors based on single level (legacy mode).
        colormap: Colormap name for hierarchical coloring ('twilight', 'hsv', 'viridis', etc.)
            Only used when hierarchical_colors=True. 'twilight' is recommended for cyclic,
            colorblind-friendly visualization.
        figsize: Figure size for each frame

    Returns:
        None (saves GIF to output_path)

    Example:
        >>> # Collect representations at different context lengths
        >>> reps_by_ctx = {
        ...     10: {'token1': rep1_ctx10, 'token2': rep2_ctx10, ...},
        ...     20: {'token1': rep1_ctx20, 'token2': rep2_ctx20, ...},
        ...     50: {'token1': rep1_ctx50, 'token2': rep2_ctx50, ...},
        ... }
        >>> create_representation_evolution_gif(
        ...     reps_by_ctx,
        ...     graph,
        ...     context_lengths=[10, 20, 50],
        ...     output_path="results/evolution.gif",
        ...     method="mds",
        ...     tight_bounds=True
        ... )
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("imageio required for GIF generation. Install with: pip install imageio")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect all representations into single matrix
    print(f"Collecting representations from {len(context_lengths)} context lengths...")
    all_reps = []
    all_labels = []  # (token, context_length) pairs

    for ctx_len in context_lengths:
        if ctx_len not in token_reps_by_context:
            warnings.warn(f"Context length {ctx_len} not found in data, skipping")
            continue

        for token, rep in token_reps_by_context[ctx_len].items():
            all_reps.append(rep)
            all_labels.append((token, ctx_len))

    if not all_reps:
        raise ValueError("No representations found in token_reps_by_context")

    data_matrix = np.array(all_reps)
    print(f"Data matrix shape: {data_matrix.shape}")

    # Step 2: Apply dimensionality reduction ONCE to all data
    print(f"Applying {method.upper()} dimensionality reduction...")

    if method == "mds":
        from sklearn.manifold import MDS
        reducer = MDS(n_components=2, random_state=42, n_init=4, max_iter=300, dissimilarity='euclidean')
        embedded = reducer.fit_transform(data_matrix)
    elif method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        embedded = reducer.fit_transform(data_matrix)
    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn required for UMAP. Install with: pip install umap-learn")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedded = reducer.fit_transform(data_matrix)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data_matrix) - 1))
        embedded = reducer.fit_transform(data_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Embedded shape: {embedded.shape}")

    # Step 3: Compute axis bounds
    if tight_bounds:
        x_min, x_max = embedded[:, 0].min(), embedded[:, 0].max()
        y_min, y_max = embedded[:, 1].min(), embedded[:, 1].max()

        # Add 10% padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding_x = 0.1 * x_range if x_range > 0 else 1.0
        padding_y = 0.1 * y_range if y_range > 0 else 1.0

        xlim = (x_min - padding_x, x_max + padding_x)
        ylim = (y_min - padding_y, y_max + padding_y)
    else:
        xlim = ylim = None

    # Step 4: Generate frames
    print(f"Generating {len(context_lengths)} frames...")
    frames = []

    for ctx_len in context_lengths:
        if ctx_len not in token_reps_by_context:
            continue

        fig, ax = plt.subplots(figsize=figsize)

        # Filter to this context length
        indices = [i for i, (_, ctx) in enumerate(all_labels) if ctx == ctx_len]
        tokens_this_ctx = [all_labels[i][0] for i in indices]
        points = embedded[indices]

        # Get hierarchy colors
        if hierarchical_colors:
            # NEW MODE: Full hierarchy path determines color
            nodes = [graph.token_to_node[token] for token in tokens_this_ctx
                     if token in graph.token_to_node]
            color_values = compute_hierarchical_color_values(graph, nodes)
            cmap = plt.cm.get_cmap(colormap)

            # Plot points
            node_idx_map = {graph.token_to_node[token]: i
                           for i, token in enumerate(tokens_this_ctx)
                           if token in graph.token_to_node}

            for i, token in enumerate(tokens_this_ctx):
                if token not in graph.token_to_node:
                    color = 'gray'
                else:
                    node = graph.token_to_node[token]
                    color_idx = node_idx_map[node]
                    color = cmap(color_values[color_idx])

                ax.scatter(
                    points[i, 0], points[i, 1],
                    c=[color], s=200,
                    edgecolors='black', linewidths=1.5,
                    alpha=0.8, zorder=2
                )

                # Add token label
                ax.annotate(
                    token,
                    (points[i, 0], points[i, 1]),
                    fontsize=8,
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    zorder=3
                )
        else:
            # LEGACY MODE: Color by single level (existing code)
            if color_by_level is None:
                raise ValueError("Must specify color_by_level when hierarchical_colors=False")

            color_map = {}
            for token in tokens_this_ctx:
                if token in graph.token_to_node:
                    node = graph.token_to_node[token]
                    cluster = graph.get_cluster_at_level(node, color_by_level)
                    color_map[token] = cluster
                else:
                    color_map[token] = -1  # Unknown

            unique_clusters = sorted(set(color_map.values()))
            cmap = plt.cm.tab10

            # Plot points
            for i, token in enumerate(tokens_this_ctx):
                cluster = color_map[token]
                if cluster == -1:
                    color = 'gray'
                else:
                    color = cmap(cluster / max(len(unique_clusters), 1))

                ax.scatter(
                    points[i, 0], points[i, 1],
                    c=[color], s=200,
                    edgecolors='black', linewidths=1.5,
                    alpha=0.8, zorder=2
                )

                # Add token label
                ax.annotate(
                    token,
                    (points[i, 0], points[i, 1]),
                    fontsize=8,
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    zorder=3
                )

        # Set axis limits
        if xlim:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        ax.set_xlabel(f"{method.upper()} Dimension 1", fontsize=11)
        ax.set_ylabel(f"{method.upper()} Dimension 2", fontsize=11)

        # Set title based on coloring mode
        if hierarchical_colors:
            title_suffix = f"(Hierarchical colors: {colormap})"
        else:
            title_suffix = f"(Colored by Level {color_by_level})"

        ax.set_title(
            f"Token Representations at N={ctx_len}\n{title_suffix}",
            fontsize=14,
            fontweight='bold',
            pad=15
        )
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')

        # Convert figure to image array
        fig.canvas.draw()
        # Use buffer_rgba() for modern matplotlib compatibility
        buf = fig.canvas.buffer_rgba()
        image = np.frombuffer(buf, dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB
        image = image[:, :, :3]
        frames.append(image)

        plt.close(fig)

    # Step 5: Save GIF
    if frames:
        print(f"Saving GIF with {len(frames)} frames to {output_path}...")
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
        print(f"✓ GIF saved: {output_path} ({len(frames)} frames @ {fps} fps)")
    else:
        warnings.warn("No frames generated, GIF not saved")


def create_phi_evolution_video(
    trajectory_data: dict,
    output_path: Union[str, Path],
    fps: int = 10,
    duration_per_context: float = 0.5,
    figsize: tuple = (12, 8),
):
    """
    Create animated video showing Phi trajectories growing over context lengths.

    Like the multi-line plot, but animated to reveal emergence order dynamically.

    Args:
        trajectory_data: Dict from compute_levelwise_phi_trajectory()
        output_path: Path to save video (e.g., "phi_evolution.mp4")
        fps: Frames per second
        duration_per_context: How long to show each context length (seconds)
        figsize: Figure size

    Returns:
        None (saves video to output_path)
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("imageio required for video generation. Install with: pip install imageio")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    context_lengths = trajectory_data['context_lengths']

    # Find levels
    levels = []
    for key in trajectory_data.keys():
        if key.startswith('phi_trajectory_level_'):
            level = int(key.split('_')[-1])
            levels.append(level)
    levels = sorted(levels)

    # Generate frames
    frames = []
    frames_per_context = int(duration_per_context * fps)

    for ctx_idx in range(1, len(context_lengths) + 1):
        # Show data up to this context
        visible_contexts = context_lengths[:ctx_idx]

        for _ in range(frames_per_context):
            fig, ax = plt.subplots(figsize=figsize)

            # Plot each level up to current context
            for level in levels:
                phi = np.array(trajectory_data[f'phi_trajectory_level_{level}'])[:ctx_idx]

                if np.all(np.isnan(phi)):
                    continue

                color = plt.cm.tab10(level / max(len(levels), 1))

                ax.plot(
                    visible_contexts, phi,
                    'o-', color=color, linewidth=2.5, markersize=8,
                    label=f'Level {level}'
                )

            ax.set_xlabel("Context Length (N)", fontsize=12)
            ax.set_ylabel("Cluster Separation (Φ)", fontsize=12)
            ax.set_title(f"Hierarchical Structure Emergence (N ≤ {context_lengths[ctx_idx-1]})",
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Set fixed axis limits
            ax.set_xlim(context_lengths[0] - 5, context_lengths[-1] + 5)

            # Get y-limits from full data
            all_phi = []
            for level in levels:
                phi = np.array(trajectory_data[f'phi_trajectory_level_{level}'])
                all_phi.extend(phi[~np.isnan(phi)])
            if all_phi:
                ax.set_ylim(0, max(all_phi) * 1.1)

            plt.tight_layout()

            # Convert to image
            fig.canvas.draw()
            # Use buffer_rgba() for modern matplotlib compatibility
            buf = fig.canvas.buffer_rgba()
            image = np.frombuffer(buf, dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            # Convert RGBA to RGB
            image = image[:, :, :3]
            frames.append(image)

            plt.close(fig)

    # Save video
    if output_path.suffix == '.gif':
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
    else:
        # Try to save as MP4 (requires imageio-ffmpeg)
        try:
            imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
        except Exception as e:
            warnings.warn(f"Could not save as video: {e}. Saving as GIF instead.")
            output_path = output_path.with_suffix('.gif')
            imageio.mimsave(output_path, frames, fps=fps, loop=0)

    print(f"✓ Animation saved: {output_path} ({len(frames)} frames @ {fps} fps)")
