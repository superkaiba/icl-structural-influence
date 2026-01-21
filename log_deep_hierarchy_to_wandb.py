#!/usr/bin/env python3
"""
Log deep hierarchy experiment results to Weights & Biases.

Usage:
    python log_deep_hierarchy_to_wandb.py --results-dir results/deep_hierarchy_gpt2 --project icl-hierarchy
"""

import argparse
import json
from pathlib import Path
import numpy as np

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Install with: pip install wandb")
    exit(1)


def log_experiment_to_wandb(results_dir: Path, project: str, run_name: str = None):
    """
    Log deep hierarchy experiment results to wandb.

    Args:
        results_dir: Directory containing experiment outputs
        project: wandb project name
        run_name: Optional run name (defaults to model name from results)
    """

    # Load trajectory data
    trajectory_path = results_dir / "phi_trajectories.json"
    if not trajectory_path.exists():
        print(f"ERROR: {trajectory_path} not found")
        return

    with open(trajectory_path) as f:
        trajectory_data = json.load(f)

    # Load CSS data if available
    css_files = list(results_dir.glob("css_decomposition_*.json"))
    css_data = None
    if css_files:
        with open(css_files[0]) as f:
            css_data = json.load(f)

    # Extract metadata from directory name or use default
    if run_name is None:
        run_name = results_dir.name

    # Initialize wandb
    print(f"Initializing wandb run: {run_name}")
    run = wandb.init(
        project=project,
        name=run_name,
        config={
            "experiment_type": "deep_hierarchy",
            "context_lengths": trajectory_data["context_lengths"],
            "num_levels": len([k for k in trajectory_data.keys() if k.startswith("phi_trajectory_")]),
        }
    )

    # Log configuration
    if css_data:
        wandb.config.update({
            "n_contexts_css": css_data.get("n_contexts", "unknown"),
        })

    # Create wandb Table for trajectory data
    print("Creating trajectory table...")
    trajectory_table = wandb.Table(
        columns=["context_length", "level", "phi_mean", "phi_std"]
    )

    context_lengths = trajectory_data["context_lengths"]
    num_levels = len([k for k in trajectory_data.keys() if k.startswith("phi_trajectory_")])

    for level in range(1, num_levels + 1):
        phi_traj = trajectory_data[f"phi_trajectory_level_{level}"]
        phi_std = trajectory_data.get(f"phi_std_level_{level}", [0] * len(phi_traj))

        for ctx_len, phi, std in zip(context_lengths, phi_traj, phi_std):
            trajectory_table.add_data(ctx_len, level, phi, std)

    wandb.log({"trajectory_data": trajectory_table})

    # Log individual trajectory metrics
    print("Logging per-level trajectories...")
    for ctx_idx, ctx_len in enumerate(context_lengths):
        metrics = {"context_length": ctx_len}

        for level in range(1, num_levels + 1):
            phi_traj = trajectory_data[f"phi_trajectory_level_{level}"]
            phi_std = trajectory_data.get(f"phi_std_level_{level}", [0] * len(phi_traj))

            metrics[f"phi_level_{level}"] = phi_traj[ctx_idx]
            metrics[f"phi_std_level_{level}"] = phi_std[ctx_idx]

        wandb.log(metrics, step=ctx_idx)

    # Log CSS decomposition if available
    if css_data:
        print("Logging CSS decomposition...")

        # Log per-level Phi means
        for level in range(1, num_levels + 1):
            phi_key = f"phi_mean_level_{level}"
            if phi_key in css_data:
                wandb.summary[f"css_{phi_key}"] = css_data[phi_key]
                wandb.summary[f"css_phi_std_level_{level}"] = css_data.get(f"phi_std_level_{level}", 0)

        # Log CSS statistics
        for level in range(1, num_levels + 1):
            css_key = f"css_level_{level}"
            if css_key in css_data:
                css_values = np.array(css_data[css_key])
                wandb.summary[f"css_max_level_{level}"] = float(np.max(np.abs(css_values)))
                wandb.summary[f"css_mean_level_{level}"] = float(np.mean(css_values))
                wandb.summary[f"css_std_level_{level}"] = float(np.std(css_values))

    # Log images
    print("Logging plots...")

    # Main trajectory plot
    phi_plot_path = results_dir / "phi_evolution.png"
    if phi_plot_path.exists():
        wandb.log({
            "phi_evolution_plot": wandb.Image(
                str(phi_plot_path),
                caption="Multi-level Phi trajectory showing hierarchical structure emergence"
            )
        })

    # Graph structure
    graph_path = results_dir / "graph_structure.png"
    if graph_path.exists():
        wandb.log({
            "graph_structure": wandb.Image(
                str(graph_path),
                caption="Hierarchical graph structure"
            )
        })

    # GIF animation if present
    gif_path = results_dir / "representation_evolution.gif"
    if gif_path.exists():
        print("Logging GIF animation...")
        wandb.log({"representation_evolution_gif": wandb.Image(str(gif_path), caption="MDS evolution showing token clustering across context lengths")})

    # Create custom plot in wandb
    print("Creating wandb custom plot...")

    # Line plot for trajectories
    data = []
    for level in range(1, num_levels + 1):
        phi_traj = trajectory_data[f"phi_trajectory_level_{level}"]
        for ctx_len, phi in zip(context_lengths, phi_traj):
            data.append([ctx_len, phi, f"Level {level}"])

    table = wandb.Table(data=data, columns=["context_length", "phi", "level"])
    wandb.log({
        "phi_trajectory_plot": wandb.plot.line(
            table,
            "context_length",
            "phi",
            title="Hierarchical Structure Emergence",
            stroke="level"
        )
    })

    # Summary statistics
    print("Computing summary statistics...")

    for level in range(1, num_levels + 1):
        phi_traj = trajectory_data[f"phi_trajectory_level_{level}"]
        phi_traj = [p for p in phi_traj if not np.isnan(p)]

        if phi_traj:
            wandb.summary[f"phi_initial_level_{level}"] = phi_traj[0]
            wandb.summary[f"phi_final_level_{level}"] = phi_traj[-1]
            wandb.summary[f"phi_max_level_{level}"] = max(phi_traj)
            wandb.summary[f"phi_mean_level_{level}"] = np.mean(phi_traj)

            # Compute growth ratio
            if phi_traj[0] > 0:
                wandb.summary[f"phi_growth_level_{level}"] = phi_traj[-1] / phi_traj[0]

    print(f"\n✓ Results logged to wandb: {run.url}")

    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Log deep hierarchy results to wandb")

    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="icl-hierarchy",
        help="Wandb project name"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name (defaults to directory name)"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return

    log_experiment_to_wandb(results_dir, args.project, args.run_name)


if __name__ == "__main__":
    main()
