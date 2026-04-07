"""T-032: Analyze structured vs random walk difference.

Compares safety degradation and collapse metrics between structured_walk_N
and random_tokens_N at matched vocabulary sizes (15, 50, 200, 1000).

Key question: Why do structured walks and random tokens produce nearly
identical safety degradation? What does the representation geometry look like?
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


BASE_DIR = Path("results/safety_collapse_sweep_v2/context_type")
PLOTS_DIR = Path("results/safety_collapse_sweep_v2/plots")
VOCAB_SIZES = [15, 50, 200, 1000]


def load_judged(path: Path) -> list[dict]:
    """Load judged results from experiment directory."""
    judged_path = path / "judge" / "all_results_judged.json"
    if judged_path.exists():
        with open(judged_path) as f:
            return json.load(f)
    # Fallback to unjudged
    raw_path = path / "all_results.json"
    if raw_path.exists():
        with open(raw_path) as f:
            return json.load(f)
    return []


def get_compliance_by_length(data: list[dict], baseline_only: bool = True) -> dict:
    """Compute compliance rate by context length.

    Returns: {length: {"compliance": float, "refusal": float, "incoherent": float, "n": int}}
    """
    by_length = defaultdict(list)
    for r in data:
        if baseline_only and not r.get("baseline_refused", False):
            continue
        length = r.get("actual_context_length", r.get("context_length", 0))
        judge = r.get("judge_classification", "")
        by_length[length].append(judge)

    result = {}
    for length, classifications in sorted(by_length.items()):
        n = len(classifications)
        if n == 0:
            continue
        n_comply = sum(1 for c in classifications if c == "compliance")
        n_refuse = sum(1 for c in classifications if c in ("full_refusal", "partial_refusal"))
        n_incoherent = sum(1 for c in classifications if c == "incoherent")
        result[length] = {
            "compliance": n_comply / n,
            "refusal": n_refuse / n,
            "incoherent": n_incoherent / n,
            "n": n,
            "n_comply": n_comply,
            "n_refuse": n_refuse,
        }
    return result


def get_collapse_by_length(data: list[dict]) -> dict:
    """Extract collapse metrics by context length.

    Returns: {length: {layer: {"avg_cos_sim": [values], "effective_dim": [values], ...}}}
    """
    by_length = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in data:
        length = r.get("actual_context_length", r.get("context_length", 0))
        cm = r.get("collapse_metrics", {})
        if not cm:
            continue
        for layer_key, metrics in cm.items():
            if metrics is None:
                continue
            layer = int(layer_key)
            for metric_name in ("avg_cos_sim", "effective_dim", "intrinsic_dim", "spread"):
                val = metrics.get(metric_name)
                if val is not None:
                    by_length[length][layer][metric_name].append(val)
    return dict(by_length)


def compare_pair(vocab_size: int) -> dict:
    """Compare structured_walk_N vs random_tokens_N."""
    struct_dir = BASE_DIR / f"structured_walk_{vocab_size}"
    random_dir = BASE_DIR / f"random_tokens_{vocab_size}"

    struct_data = load_judged(struct_dir)
    random_data = load_judged(random_dir)

    if not struct_data or not random_data:
        print(f"  Missing data for vocab={vocab_size}: "
              f"structured={len(struct_data)}, random={len(random_data)}")
        return {}

    # Safety comparison
    struct_safety = get_compliance_by_length(struct_data)
    random_safety = get_compliance_by_length(random_data)

    # Collapse metric comparison
    struct_collapse = get_collapse_by_length(struct_data)
    random_collapse = get_collapse_by_length(random_data)

    # Statistical tests at each length
    comparison = {
        "vocab_size": vocab_size,
        "n_struct": len(struct_data),
        "n_random": len(random_data),
        "safety_by_length": {},
        "collapse_by_length": {},
    }

    all_lengths = sorted(set(struct_safety.keys()) | set(random_safety.keys()))

    for length in all_lengths:
        s = struct_safety.get(length, {})
        r = random_safety.get(length, {})

        if not s or not r:
            continue

        # Fisher's exact test on compliance counts
        table = [
            [s.get("n_comply", 0), s.get("n", 0) - s.get("n_comply", 0)],
            [r.get("n_comply", 0), r.get("n", 0) - r.get("n_comply", 0)],
        ]
        try:
            _, fisher_p = stats.fisher_exact(table)
        except ValueError:
            fisher_p = 1.0

        # Cohen's d on compliance rates (using proportions)
        s_rate = s.get("compliance", 0)
        r_rate = r.get("compliance", 0)
        diff = s_rate - r_rate

        comparison["safety_by_length"][length] = {
            "struct_compliance": s_rate,
            "random_compliance": r_rate,
            "difference": diff,
            "fisher_p": fisher_p,
            "struct_n": s.get("n", 0),
            "random_n": r.get("n", 0),
        }

        # Collapse metrics comparison
        sc = struct_collapse.get(length, {})
        rc = random_collapse.get(length, {})

        layer_comparisons = {}
        for layer in sorted(set(sc.keys()) | set(rc.keys())):
            sm = sc.get(layer, {})
            rm = rc.get(layer, {})
            layer_comp = {}
            for metric in ("avg_cos_sim", "effective_dim", "intrinsic_dim"):
                sv = sm.get(metric, [])
                rv = rm.get(metric, [])
                if len(sv) >= 2 and len(rv) >= 2:
                    try:
                        t_stat, t_p = stats.ttest_ind(sv, rv)
                    except Exception:
                        t_stat, t_p = 0, 1.0
                    # Cohen's d
                    pooled_std = np.sqrt(
                        ((len(sv) - 1) * np.var(sv) + (len(rv) - 1) * np.var(rv))
                        / (len(sv) + len(rv) - 2)
                    )
                    cohens_d = (np.mean(sv) - np.mean(rv)) / pooled_std if pooled_std > 1e-10 else 0
                    layer_comp[metric] = {
                        "struct_mean": float(np.mean(sv)),
                        "struct_std": float(np.std(sv)),
                        "random_mean": float(np.mean(rv)),
                        "random_std": float(np.std(rv)),
                        "t_stat": float(t_stat),
                        "p_value": float(t_p),
                        "cohens_d": float(cohens_d),
                    }
            if layer_comp:
                layer_comparisons[layer] = layer_comp

        comparison["collapse_by_length"][length] = layer_comparisons

    return comparison


def print_summary(comparisons: list[dict]):
    """Print summary table to console."""
    print("\n" + "=" * 90)
    print("T-032: STRUCTURED vs RANDOM — SAFETY COMPARISON")
    print("=" * 90)

    for comp in comparisons:
        vocab = comp["vocab_size"]
        print(f"\n--- Vocab Size = {vocab} ---")
        print(f"{'Length':>8}  {'Struct Compl':>12}  {'Random Compl':>12}  {'Diff':>8}  {'Fisher p':>10}  {'Sig?':>5}")
        print("-" * 70)

        for length, s in sorted(comp["safety_by_length"].items()):
            sig = "*" if s["fisher_p"] < 0.05 else ""
            print(f"{length:>8}  {s['struct_compliance']:>12.3f}  {s['random_compliance']:>12.3f}  "
                  f"{s['difference']:>+8.3f}  {s['fisher_p']:>10.4f}  {sig:>5}")

    print("\n" + "=" * 90)
    print("T-032: COLLAPSE METRICS COMPARISON (last layer)")
    print("=" * 90)

    for comp in comparisons:
        vocab = comp["vocab_size"]
        print(f"\n--- Vocab Size = {vocab} ---")

        for length, layers in sorted(comp["collapse_by_length"].items()):
            if not layers:
                continue
            # Use last layer
            last_layer = max(layers.keys())
            lm = layers[last_layer]

            cos_sim = lm.get("avg_cos_sim", {})
            eff_dim = lm.get("effective_dim", {})

            if cos_sim:
                print(f"  Length={length}: cos_sim struct={cos_sim['struct_mean']:.3f}±{cos_sim['struct_std']:.3f} "
                      f"random={cos_sim['random_mean']:.3f}±{cos_sim['random_std']:.3f} "
                      f"d={cos_sim['cohens_d']:.2f} p={cos_sim['p_value']:.4f}")
            if eff_dim:
                print(f"  Length={length}: eff_dim struct={eff_dim['struct_mean']:.1f}±{eff_dim['struct_std']:.1f} "
                      f"random={eff_dim['random_mean']:.1f}±{eff_dim['random_std']:.1f} "
                      f"d={eff_dim['cohens_d']:.2f} p={eff_dim['p_value']:.4f}")


def plot_comparison(comparisons: list[dict], output_dir: Path):
    """Generate comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("T-032: Structured Walk vs Random Tokens\n(Safety Compliance Rate by Context Length)",
                 fontsize=14, fontweight="bold")

    for idx, comp in enumerate(comparisons):
        vocab = comp["vocab_size"]
        ax = axes[idx // 2, idx % 2]

        lengths = sorted(comp["safety_by_length"].keys())
        if not lengths:
            ax.set_title(f"Vocab={vocab} (no data)")
            continue

        struct_rates = [comp["safety_by_length"][l]["struct_compliance"] for l in lengths]
        random_rates = [comp["safety_by_length"][l]["random_compliance"] for l in lengths]

        ax.plot(lengths, struct_rates, "o-", color="#e74c3c", label="Structured walk", linewidth=2)
        ax.plot(lengths, random_rates, "s--", color="#3498db", label="Random tokens", linewidth=2)

        # Mark significant differences
        for l in lengths:
            s = comp["safety_by_length"][l]
            if s["fisher_p"] < 0.05:
                y_pos = max(s["struct_compliance"], s["random_compliance"]) + 0.02
                ax.annotate("*", (l, y_pos), ha="center", fontsize=14, color="red")

        ax.set_xscale("log")
        ax.set_xlabel("Context Length (tokens)")
        ax.set_ylabel("Compliance Rate")
        ax.set_title(f"Vocab Size = {vocab}")
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "t032_safety_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 't032_safety_comparison.png'}")

    # Plot collapse metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("T-032: Collapse Metrics — Structured Walk vs Random Tokens\n(avg_cos_sim, last layer)",
                 fontsize=14, fontweight="bold")

    for idx, comp in enumerate(comparisons):
        vocab = comp["vocab_size"]
        ax = axes[idx // 2, idx % 2]

        lengths = sorted(comp["collapse_by_length"].keys())
        if not lengths:
            ax.set_title(f"Vocab={vocab} (no data)")
            continue

        struct_cos = []
        random_cos = []
        valid_lengths = []

        for l in lengths:
            layers = comp["collapse_by_length"][l]
            if not layers:
                continue
            last_layer = max(layers.keys())
            lm = layers[last_layer]
            cs = lm.get("avg_cos_sim", {})
            if cs:
                struct_cos.append(cs["struct_mean"])
                random_cos.append(cs["random_mean"])
                valid_lengths.append(l)

        if valid_lengths:
            ax.plot(valid_lengths, struct_cos, "o-", color="#e74c3c", label="Structured walk", linewidth=2)
            ax.plot(valid_lengths, random_cos, "s--", color="#3498db", label="Random tokens", linewidth=2)

        ax.set_xscale("log")
        ax.set_xlabel("Context Length (tokens)")
        ax.set_ylabel("avg_cos_sim")
        ax.set_title(f"Vocab Size = {vocab}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "t032_collapse_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 't032_collapse_comparison.png'}")

    # Per-layer breakdown for vocab=15 (most important)
    if comparisons:
        comp = comparisons[0]  # vocab=15
        lengths = sorted(comp["collapse_by_length"].keys())
        if lengths:
            all_layers = set()
            for l in lengths:
                all_layers.update(comp["collapse_by_length"][l].keys())
            all_layers = sorted(all_layers)

            if len(all_layers) > 1:
                n_layers = min(len(all_layers), 6)
                fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
                if n_layers == 1:
                    axes = [axes]
                fig.suptitle(f"T-032: Layer-wise Collapse Comparison (Vocab=15)",
                             fontsize=13, fontweight="bold")

                for ax_idx, layer in enumerate(all_layers[:n_layers]):
                    ax = axes[ax_idx]
                    s_vals, r_vals, valid_l = [], [], []
                    for l in lengths:
                        lm = comp["collapse_by_length"][l].get(layer, {})
                        cs = lm.get("avg_cos_sim", {})
                        if cs:
                            s_vals.append(cs["struct_mean"])
                            r_vals.append(cs["random_mean"])
                            valid_l.append(l)
                    if valid_l:
                        ax.plot(valid_l, s_vals, "o-", color="#e74c3c", label="Struct", linewidth=1.5)
                        ax.plot(valid_l, r_vals, "s--", color="#3498db", label="Random", linewidth=1.5)
                    ax.set_xscale("log")
                    ax.set_title(f"Layer {layer}")
                    ax.set_xlabel("Context Length")
                    if ax_idx == 0:
                        ax.set_ylabel("avg_cos_sim")
                    ax.legend(fontsize=7)
                    ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(output_dir / "t032_layerwise_comparison.png", dpi=150, bbox_inches="tight")
                plt.close()
                print(f"Saved: {output_dir / 't032_layerwise_comparison.png'}")


def main():
    print("T-032: Analyzing structured walk vs random tokens")
    print(f"Data directory: {BASE_DIR}")
    print(f"Vocab sizes: {VOCAB_SIZES}")

    comparisons = []
    for vocab in VOCAB_SIZES:
        print(f"\nProcessing vocab={vocab}...")
        comp = compare_pair(vocab)
        if comp:
            comparisons.append(comp)

    if not comparisons:
        print("No data found!")
        return

    print_summary(comparisons)
    plot_comparison(comparisons, PLOTS_DIR)

    # Save full results
    output_path = PLOTS_DIR / "t032_analysis_results.json"
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(comparisons), f, indent=2)
    print(f"\nSaved analysis results: {output_path}")


if __name__ == "__main__":
    main()
