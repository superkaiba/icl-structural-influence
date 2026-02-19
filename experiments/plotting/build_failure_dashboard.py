#!/usr/bin/env python3
"""Build an interactive HTML dashboard showing failure modes across all experiment conditions."""

import json
import os
from pathlib import Path
from collections import defaultdict

RESULTS_BASE = Path("results")


def load_raw_trials(raw_dir):
    """Load all raw trial JSON files from a directory."""
    records = []
    if not raw_dir.exists():
        return records
    for f in sorted(raw_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
            records.extend(data)
    return records


def get_condition(rec):
    """Get condition name from record (handles both field names)."""
    return rec.get("context_type") or rec.get("condition", "unknown")


def get_collapse_summary(rec):
    """Get collapse metrics summary string."""
    metrics = rec.get("collapse_metrics", {})
    if not metrics:
        return {"cos_sim": "N/A", "eff_dim": "N/A", "layer_details": []}

    layer_details = []
    cos_sims = []
    eff_dims = []
    for layer in ["0", "7", "14", "21", "27"]:
        if layer in metrics:
            m = metrics[layer]
            cs = m.get("avg_cos_sim", 0)
            ed = m.get("effective_dim", 0)
            cos_sims.append(cs)
            eff_dims.append(ed)
            layer_details.append({"layer": int(layer), "cos_sim": cs, "eff_dim": ed})

    avg_cs = sum(cos_sims) / len(cos_sims) if cos_sims else 0
    avg_ed = sum(eff_dims) / len(eff_dims) if eff_dims else 0
    return {"cos_sim": f"{avg_cs:.3f}", "eff_dim": f"{avg_ed:.1f}", "layer_details": layer_details}


def classify_failure(rec):
    """Classify a failure into a failure mode category."""
    gen = rec.get("generated_answer", "")
    expected = rec.get("expected_answer", "")

    if not gen:
        return "empty_response"

    # Instruction echoing
    if "answer in" in gen.lower() and "few words" in gen.lower():
        return "instruction_echo"
    if "as few words as possible" in gen.lower():
        return "instruction_echo"

    # Number/token repetition
    if len(gen) > 5:
        unique_chars = set(gen.replace("-", "").replace(" ", "").replace(".", ""))
        if len(unique_chars) <= 2 and len(gen) > 8:
            return "token_repetition"

    # Code block artifacts
    if gen.startswith("```") or gen.startswith("``"):
        return "code_artifact"

    # Context over-fitting (refuses to answer from world knowledge)
    refusal_phrases = [
        "does not contain", "not mentioned", "no one mentioned",
        "text does not", "passage does not", "not relate to",
        "cannot be determined from", "not in the text",
        "does not relate"
    ]
    if any(p in gen.lower() for p in refusal_phrases):
        return "context_overfitting"

    # Language switching (Chinese characters)
    if any('\u4e00' <= c <= '\u9fff' for c in gen):
        return "language_switch"

    # Conversation artifacts
    if "Human:" in gen or "Assistant:" in gen:
        return "conversation_artifact"

    # Answer repetition loops
    if len(gen) > 50 and gen.count(expected) > 1:
        return "answer_loop"

    # Wrong but coherent answer
    return "wrong_answer"


FAILURE_MODE_LABELS = {
    "instruction_echo": "Instruction Echoing",
    "token_repetition": "Token/Number Repetition",
    "code_artifact": "Code Block Artifacts",
    "context_overfitting": "Context Over-fitting",
    "language_switch": "Language Switching",
    "conversation_artifact": "Conversation Artifacts",
    "answer_loop": "Answer Repetition Loop",
    "wrong_answer": "Wrong (Coherent) Answer",
    "empty_response": "Empty Response",
}

FAILURE_MODE_COLORS = {
    "instruction_echo": "#e74c3c",
    "token_repetition": "#9b59b6",
    "code_artifact": "#3498db",
    "context_overfitting": "#e67e22",
    "language_switch": "#1abc9c",
    "conversation_artifact": "#f39c12",
    "answer_loop": "#e91e63",
    "wrong_answer": "#95a5a6",
    "empty_response": "#7f8c8d",
}

FAILURE_MODE_DESCRIPTIONS = {
    "instruction_echo": "Model repeats the instruction prompt verbatim instead of answering the question. Dominant failure mode for structured walks at extreme context lengths.",
    "token_repetition": "Model emits repeated digits or characters (e.g., '155555555...', '6666...'). Indicates complete representational collapse where the model fixates on tokens from context.",
    "code_artifact": "Model generates code block markers (```python) before attempting to answer. Appears at moderate collapse levels.",
    "context_overfitting": "Model refuses to use world knowledge, insisting the answer must come from the provided context text. Common with long single-topic natural language.",
    "language_switch": "Model answers correctly but in the wrong language (Chinese instead of English). Occurs at extreme context lengths.",
    "conversation_artifact": "Model generates conversation markers like 'Human:' or 'Assistant:' in its response. Indicates attention bleeding from chat template.",
    "answer_loop": "Model generates the correct answer but then repeats it in a loop. Transitional state between functional and collapsed.",
    "wrong_answer": "Model generates a coherent but incorrect answer (wrong facts, arithmetic errors, synonym substitutions).",
    "empty_response": "Model generates an empty or whitespace-only response.",
}


def collect_all_data():
    """Collect all raw trial data from all experiments."""
    all_records = []

    # 1. Main probing experiment
    raw_dir = RESULTS_BASE / "probing_collapse_performance" / "raw"
    records = load_raw_trials(raw_dir)
    for r in records:
        r["experiment"] = "probing_collapse"
    all_records.extend(records)

    # 2. Misspellings
    raw_dir = RESULTS_BASE / "context_variation_v2_fixed" / "misspellings" / "raw"
    records = load_raw_trials(raw_dir)
    for r in records:
        r["experiment"] = "misspellings"
    all_records.extend(records)

    # 3. Topic changes
    raw_dir = RESULTS_BASE / "context_variation_v2_fixed" / "topic_changes" / "raw"
    records = load_raw_trials(raw_dir)
    for r in records:
        r["experiment"] = "topic_changes"
    all_records.extend(records)

    # 4. Extended length
    raw_dir = RESULTS_BASE / "context_variation_maxctx" / "extended_length" / "raw"
    records = load_raw_trials(raw_dir)
    for r in records:
        r["experiment"] = "extended_length"
    all_records.extend(records)

    return all_records


def build_html(all_records):
    """Build the complete HTML dashboard."""

    # Separate correct and incorrect
    failures = [r for r in all_records if not r.get("answer_correct", True)]
    correct = [r for r in all_records if r.get("answer_correct", True)]

    # Classify failures
    for f in failures:
        f["failure_mode"] = classify_failure(f)

    # Group failures by condition and context length
    by_condition = defaultdict(list)
    for f in failures:
        cond = get_condition(f)
        by_condition[cond].append(f)

    # Compute accuracy by condition and context length
    accuracy_data = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    for r in all_records:
        cond = get_condition(r)
        cl = r["context_length"]
        accuracy_data[cond][cl]["total"] += 1
        if r.get("answer_correct", False):
            accuracy_data[cond][cl]["correct"] += 1

    # Build failure mode counts by condition
    mode_counts_by_cond = defaultdict(lambda: defaultdict(int))
    for f in failures:
        cond = get_condition(f)
        mode_counts_by_cond[cond][f["failure_mode"]] += 1

    # Pick representative examples for each failure mode (diverse conditions)
    examples_by_mode = defaultdict(list)
    for f in failures:
        mode = f["failure_mode"]
        if len(examples_by_mode[mode]) < 12:
            examples_by_mode[mode].append(f)

    # Condition display order and labels
    CONDITION_ORDER = [
        "no_context", "natural_books", "natural_books_extended",
        "single_topic", "multi_topic_300", "multi_topic_1000",
        "misspell_10pct", "misspell_25pct", "misspell_50pct",
        "structured_walk", "repeated_token",
    ]
    CONDITION_LABELS = {
        "no_context": "No Context (Baseline)",
        "natural_books": "Natural Books",
        "natural_books_extended": "Natural Books (Extended)",
        "single_topic": "Single Topic",
        "multi_topic_300": "Multi-Topic (300 tok/switch)",
        "multi_topic_1000": "Multi-Topic (1000 tok/switch)",
        "misspell_10pct": "Misspellings 10%",
        "misspell_25pct": "Misspellings 25%",
        "misspell_50pct": "Misspellings 50%",
        "structured_walk": "Structured Walk",
        "repeated_token": "Repeated Token",
    }

    # Build accuracy table rows
    all_conditions = sorted(set(get_condition(r) for r in all_records),
                           key=lambda c: CONDITION_ORDER.index(c) if c in CONDITION_ORDER else 99)
    all_lengths = sorted(set(r["context_length"] for r in all_records))

    # Build HTML
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Failure Mode Dashboard — Representational Collapse Experiments</title>
<style>
  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --surface2: #21262d;
    --border: #30363d;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --red: #f85149;
    --orange: #d29922;
    --purple: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    padding: 24px;
  }
  h1 { font-size: 28px; margin-bottom: 8px; }
  h2 { font-size: 22px; margin: 32px 0 16px; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 8px; }
  h3 { font-size: 18px; margin: 24px 0 12px; }
  .subtitle { color: var(--text-muted); font-size: 14px; margin-bottom: 24px; }

  /* Navigation tabs */
  .tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 24px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0;
    flex-wrap: wrap;
  }
  .tab {
    padding: 8px 16px;
    cursor: pointer;
    border: 1px solid transparent;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    background: transparent;
    color: var(--text-muted);
    font-size: 14px;
    transition: all 0.15s;
  }
  .tab:hover { color: var(--text); background: var(--surface); }
  .tab.active { color: var(--text); background: var(--surface); border-color: var(--border); border-bottom: 2px solid var(--accent); }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  /* Cards */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 16px;
  }
  .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }

  /* Stats grid */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
  }
  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
  }
  .stat-value { font-size: 32px; font-weight: 700; }
  .stat-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }

  /* Failure mode badges */
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
    white-space: nowrap;
  }

  /* Tables */
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }
  th { background: var(--surface2); color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; position: sticky; top: 0; }
  tr:hover { background: var(--surface2); }

  /* Accuracy heatmap */
  .heatmap-cell {
    padding: 6px 10px;
    text-align: center;
    font-weight: 500;
    font-size: 13px;
    border-radius: 4px;
  }

  /* Example cards */
  .example {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px;
    margin-bottom: 10px;
    font-size: 13px;
  }
  .example-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; flex-wrap: wrap; gap: 6px; }
  .example-meta { color: var(--text-muted); font-size: 11px; }
  .example-q { color: var(--accent); margin-bottom: 4px; }
  .example-expected { color: var(--green); }
  .example-generated { color: var(--red); font-family: 'SFMono-Regular', Consolas, monospace; word-break: break-all; }
  .example-generated.correct { color: var(--green); }
  .collapse-bar {
    display: flex;
    gap: 8px;
    margin-top: 6px;
    font-size: 11px;
    color: var(--text-muted);
  }
  .collapse-bar span { display: inline-flex; align-items: center; gap: 3px; }

  /* Failure mode section */
  .failure-mode-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
  }
  .failure-mode-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .failure-mode-desc {
    color: var(--text-muted);
    font-size: 13px;
    margin-bottom: 16px;
    padding-left: 24px;
  }

  /* Filter controls */
  .filters {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
    flex-wrap: wrap;
    align-items: center;
  }
  .filter-group { display: flex; align-items: center; gap: 6px; }
  .filter-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; }
  select, input[type="range"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 13px;
  }

  /* Tier cards */
  .tier {
    border-left: 4px solid;
    padding-left: 16px;
    margin-bottom: 24px;
  }
  .tier-1 { border-color: var(--red); }
  .tier-2 { border-color: var(--orange); }
  .tier-3 { border-color: #d29922; }
  .tier-4 { border-color: var(--green); }
  .tier-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
    margin-bottom: 4px;
  }

  /* Scrollable table container */
  .table-scroll { overflow-x: auto; }

  /* Condition group */
  .condition-group { margin-bottom: 32px; }
  .condition-label {
    font-size: 14px;
    font-weight: 600;
    padding: 8px 12px;
    background: var(--surface2);
    border-radius: 6px;
    margin-bottom: 12px;
    display: inline-block;
  }

  /* Summary bar chart */
  .bar-chart { margin: 16px 0; }
  .bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .bar-label { width: 180px; font-size: 12px; text-align: right; flex-shrink: 0; }
  .bar-track { flex: 1; height: 24px; background: var(--surface2); border-radius: 4px; overflow: hidden; position: relative; }
  .bar-fill { height: 100%; border-radius: 4px; display: flex; align-items: center; padding-left: 8px; font-size: 11px; font-weight: 600; min-width: fit-content; transition: width 0.3s; }
  .bar-value { font-size: 12px; width: 50px; text-align: right; flex-shrink: 0; color: var(--text-muted); }
</style>
</head>
<body>
<h1>Failure Mode Dashboard</h1>
<p class="subtitle">Representational Collapse &amp; Knowledge Retrieval Experiments &mdash; Qwen2.5-7B-Instruct</p>
"""

    # Tabs
    html += """
<div class="tabs">
  <div class="tab active" onclick="showTab('overview')">Overview</div>
  <div class="tab" onclick="showTab('accuracy')">Accuracy Heatmap</div>
  <div class="tab" onclick="showTab('modes')">Failure Modes</div>
  <div class="tab" onclick="showTab('examples')">Example Browser</div>
  <div class="tab" onclick="showTab('tiers')">Failure Tiers</div>
</div>
"""

    # ==================== OVERVIEW TAB ====================
    total = len(all_records)
    total_fail = len(failures)
    total_correct = len(correct)
    n_conditions = len(all_conditions)
    n_lengths = len(all_lengths)

    # Mode distribution
    mode_counts = defaultdict(int)
    for f in failures:
        mode_counts[f["failure_mode"]] += 1

    html += f"""
<div id="tab-overview" class="tab-content active">
  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-value">{total:,}</div>
      <div class="stat-label">Total Evaluations</div>
    </div>
    <div class="stat-card">
      <div class="stat-value" style="color:var(--green)">{total_correct:,}</div>
      <div class="stat-label">Correct</div>
    </div>
    <div class="stat-card">
      <div class="stat-value" style="color:var(--red)">{total_fail:,}</div>
      <div class="stat-label">Failures</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{n_conditions}</div>
      <div class="stat-label">Conditions</div>
    </div>
  </div>

  <h2>Failure Mode Distribution</h2>
  <div class="card">
    <div class="bar-chart">
"""

    # Sort modes by count
    sorted_modes = sorted(mode_counts.items(), key=lambda x: -x[1])
    max_count = max(mode_counts.values()) if mode_counts else 1
    for mode, count in sorted_modes:
        pct = count / total_fail * 100 if total_fail > 0 else 0
        bar_pct = count / max_count * 100
        color = FAILURE_MODE_COLORS.get(mode, "#666")
        label = FAILURE_MODE_LABELS.get(mode, mode)
        html += f"""      <div class="bar-row">
        <div class="bar-label">{label}</div>
        <div class="bar-track"><div class="bar-fill" style="width:{bar_pct:.1f}%;background:{color}">{count}</div></div>
        <div class="bar-value">{pct:.1f}%</div>
      </div>
"""

    html += """    </div>
  </div>

  <h2>Failures by Condition</h2>
  <div class="card">
    <div class="bar-chart">
"""

    # Failures by condition
    cond_fail_counts = defaultdict(int)
    cond_total_counts = defaultdict(int)
    for r in all_records:
        cond = get_condition(r)
        cond_total_counts[cond] += 1
        if not r.get("answer_correct", True):
            cond_fail_counts[cond] += 1

    sorted_conds = sorted(cond_fail_counts.items(), key=lambda x: -x[1])
    max_cond_fail = max(cond_fail_counts.values()) if cond_fail_counts else 1
    for cond, count in sorted_conds:
        total_c = cond_total_counts[cond]
        pct = count / total_c * 100 if total_c > 0 else 0
        bar_pct = count / max_cond_fail * 100
        label = CONDITION_LABELS.get(cond, cond)
        color = "#f85149" if pct > 30 else "#d29922" if pct > 10 else "#3fb950"
        html += f"""      <div class="bar-row">
        <div class="bar-label">{label}</div>
        <div class="bar-track"><div class="bar-fill" style="width:{bar_pct:.1f}%;background:{color}">{count}/{total_c}</div></div>
        <div class="bar-value">{pct:.0f}%</div>
      </div>
"""

    html += """    </div>
  </div>
</div>
"""

    # ==================== ACCURACY HEATMAP TAB ====================
    html += """
<div id="tab-accuracy" class="tab-content">
  <h2>Accuracy by Condition &times; Context Length</h2>
  <div class="card">
    <div class="table-scroll">
      <table>
        <tr><th>Condition</th>"""

    for cl in all_lengths:
        if cl >= 1000:
            label = f"{cl//1000}K"
        else:
            label = str(cl)
        html += f"<th>{label}</th>"
    html += "</tr>\n"

    for cond in all_conditions:
        label = CONDITION_LABELS.get(cond, cond)
        html += f"        <tr><td><strong>{label}</strong></td>"
        for cl in all_lengths:
            data = accuracy_data[cond][cl]
            if data["total"] == 0:
                html += "<td style='color:var(--text-muted)'>—</td>"
            else:
                acc = data["correct"] / data["total"] * 100
                if acc >= 95:
                    bg = "rgba(63,185,80,0.2)"
                    fg = "var(--green)"
                elif acc >= 85:
                    bg = "rgba(210,153,34,0.15)"
                    fg = "var(--orange)"
                elif acc >= 50:
                    bg = "rgba(248,81,73,0.15)"
                    fg = "var(--red)"
                else:
                    bg = "rgba(248,81,73,0.3)"
                    fg = "var(--red)"
                html += f"<td><div class='heatmap-cell' style='background:{bg};color:{fg}'>{acc:.0f}%</div></td>"
        html += "</tr>\n"

    html += """      </table>
    </div>
  </div>
"""

    # Collapse metrics heatmap
    html += """
  <h2>Avg Cosine Similarity (Collapse Indicator)</h2>
  <div class="card">
    <div class="table-scroll">
      <table>
        <tr><th>Condition</th>"""

    for cl in all_lengths:
        label = f"{cl//1000}K" if cl >= 1000 else str(cl)
        html += f"<th>{label}</th>"
    html += "</tr>\n"

    # Compute avg cos_sim per condition x length
    cos_sim_data = defaultdict(lambda: defaultdict(list))
    for r in all_records:
        cond = get_condition(r)
        cl = r["context_length"]
        metrics = r.get("collapse_metrics", {})
        if "27" in metrics:
            cos_sim_data[cond][cl].append(metrics["27"].get("avg_cos_sim", 0))

    for cond in all_conditions:
        label = CONDITION_LABELS.get(cond, cond)
        html += f"        <tr><td><strong>{label}</strong></td>"
        for cl in all_lengths:
            vals = cos_sim_data[cond][cl]
            if not vals:
                html += "<td style='color:var(--text-muted)'>—</td>"
            else:
                avg = sum(vals) / len(vals)
                if avg >= 0.9:
                    bg = "rgba(248,81,73,0.3)"
                    fg = "var(--red)"
                elif avg >= 0.6:
                    bg = "rgba(248,81,73,0.15)"
                    fg = "var(--red)"
                elif avg >= 0.4:
                    bg = "rgba(210,153,34,0.15)"
                    fg = "var(--orange)"
                else:
                    bg = "rgba(63,185,80,0.15)"
                    fg = "var(--green)"
                html += f"<td><div class='heatmap-cell' style='background:{bg};color:{fg}'>{avg:.2f}</div></td>"
        html += "</tr>\n"

    html += """      </table>
    </div>
  </div>
</div>
"""

    # ==================== FAILURE MODES TAB ====================
    html += """
<div id="tab-modes" class="tab-content">
  <h2>Failure Mode Catalog</h2>
"""

    for mode, count in sorted_modes:
        color = FAILURE_MODE_COLORS.get(mode, "#666")
        label = FAILURE_MODE_LABELS.get(mode, mode)
        desc = FAILURE_MODE_DESCRIPTIONS.get(mode, "")
        pct = count / total_fail * 100 if total_fail > 0 else 0

        # Get representative examples (diverse conditions and lengths)
        mode_examples = [f for f in failures if f["failure_mode"] == mode]
        # Sort by severity (higher context length first for structured, etc.)
        mode_examples.sort(key=lambda x: (-x["context_length"], get_condition(x)))
        # Pick diverse examples
        seen = set()
        diverse_examples = []
        for ex in mode_examples:
            key = (get_condition(ex), ex["context_length"])
            if key not in seen and len(diverse_examples) < 6:
                seen.add(key)
                diverse_examples.append(ex)

        # Conditions where this mode appears
        mode_conditions = defaultdict(int)
        for f in mode_examples:
            mode_conditions[get_condition(f)] += 1

        html += f"""
  <div class="card">
    <div class="failure-mode-header">
      <div class="failure-mode-dot" style="background:{color}"></div>
      <h3 style="margin:0">{label}</h3>
      <span class="badge" style="background:{color}33;color:{color}">{count} failures ({pct:.1f}%)</span>
    </div>
    <p class="failure-mode-desc">{desc}</p>

    <div style="margin-bottom:12px;font-size:12px;color:var(--text-muted)">
      <strong>Appears in:</strong> {', '.join(CONDITION_LABELS.get(c, c) + f' ({n})' for c, n in sorted(mode_conditions.items(), key=lambda x: -x[1]))}
    </div>

    <h4 style="font-size:13px;margin-bottom:8px;color:var(--text-muted)">Representative Examples</h4>
"""

        for ex in diverse_examples:
            cond = get_condition(ex)
            cond_label = CONDITION_LABELS.get(cond, cond)
            cl = ex["context_length"]
            cl_label = f"{cl//1000}K" if cl >= 1000 else str(cl)
            cat = ex.get("category", "")
            collapse = get_collapse_summary(ex)
            gen = ex.get("generated_answer", "")
            # Truncate very long generated answers
            if len(gen) > 120:
                gen = gen[:120] + "..."

            html += f"""    <div class="example">
      <div class="example-header">
        <span class="example-meta">{cond_label} &middot; {cl_label} tokens &middot; {cat}</span>
        <span class="example-meta">cos_sim: {collapse['cos_sim']} &middot; eff_dim: {collapse['eff_dim']}</span>
      </div>
      <div class="example-q">Q: {ex.get('question', '')}</div>
      <div class="example-expected">Expected: {ex.get('expected_answer', '')}</div>
      <div class="example-generated">Generated: {_escape(gen)}</div>
    </div>
"""

        html += "  </div>\n"

    html += "</div>\n"

    # ==================== EXAMPLE BROWSER TAB ====================
    html += """
<div id="tab-examples" class="tab-content">
  <h2>Example Browser</h2>
  <div class="filters">
    <div class="filter-group">
      <span class="filter-label">Condition:</span>
      <select id="filter-condition" onchange="filterExamples()">
        <option value="all">All Conditions</option>
"""
    for cond in all_conditions:
        label = CONDITION_LABELS.get(cond, cond)
        html += f'        <option value="{cond}">{label}</option>\n'

    html += """      </select>
    </div>
    <div class="filter-group">
      <span class="filter-label">Show:</span>
      <select id="filter-correct" onchange="filterExamples()">
        <option value="failures">Failures Only</option>
        <option value="all">All Results</option>
        <option value="correct">Correct Only</option>
      </select>
    </div>
    <div class="filter-group">
      <span class="filter-label">Category:</span>
      <select id="filter-category" onchange="filterExamples()">
        <option value="all">All Categories</option>
        <option value="A_factual">A: Factual</option>
        <option value="B_reasoning">B: Reasoning</option>
        <option value="C_word_knowledge">C: Word Knowledge</option>
        <option value="D_multi_token">D: Multi-token</option>
      </select>
    </div>
    <div class="filter-group">
      <span class="filter-label">Failure Mode:</span>
      <select id="filter-mode" onchange="filterExamples()">
        <option value="all">All Modes</option>
"""
    for mode, _ in sorted_modes:
        label = FAILURE_MODE_LABELS.get(mode, mode)
        html += f'        <option value="{mode}">{label}</option>\n'

    html += """      </select>
    </div>
  </div>
  <div id="example-count" style="font-size:12px;color:var(--text-muted);margin-bottom:12px"></div>
  <div id="example-list"></div>
</div>
"""

    # ==================== TIERS TAB ====================
    html += """
<div id="tab-tiers" class="tab-content">
  <h2>Failure Severity Tiers</h2>
"""

    tier_data = [
        {
            "tier": 1,
            "class": "tier-1",
            "label": "Catastrophic Collapse",
            "accuracy": "0-10%",
            "color": "var(--red)",
            "conditions": "Structured Walk 20K, Repeated Token 10K+",
            "description": "Complete loss of semantic capability. The model cannot form coherent responses.",
            "modes": ["instruction_echo", "token_repetition", "empty_response"],
            "cos_sim": "> 0.93",
            "eff_dim": "< 8",
        },
        {
            "tier": 2,
            "class": "tier-2",
            "label": "Severe Degradation",
            "accuracy": "60-80%",
            "color": "var(--orange)",
            "conditions": "Structured Walk 5-10K, Repeated Token 2K, Misspelling 10% @ 128K",
            "description": "Model still attempts answers but frequently fails. Reasoning and multi-token answers degrade first.",
            "modes": ["wrong_answer", "conversation_artifact", "code_artifact"],
            "cos_sim": "0.46-0.96",
            "eff_dim": "7-11",
        },
        {
            "tier": 3,
            "class": "tier-3",
            "label": "Mild Degradation",
            "accuracy": "80-93%",
            "color": "#d29922",
            "conditions": "All natural language at 128K, Structured Walk 2-5K",
            "description": "Model is mostly functional. Failures are coherent but wrong: arithmetic errors, synonym substitutions, context over-fitting.",
            "modes": ["context_overfitting", "wrong_answer", "language_switch"],
            "cos_sim": "0.30-0.61",
            "eff_dim": "10-18",
        },
        {
            "tier": 4,
            "class": "tier-4",
            "label": "Near-Baseline",
            "accuracy": "> 93%",
            "color": "var(--green)",
            "conditions": "Natural language up to 100K, Multi-topic at all lengths",
            "description": "Performance indistinguishable from no-context baseline. Only rare, isolated errors.",
            "modes": ["wrong_answer"],
            "cos_sim": "< 0.42",
            "eff_dim": "12-22",
        },
    ]

    for t in tier_data:
        html += f"""
  <div class="tier {t['class']}">
    <div class="tier-label" style="color:{t['color']}">Tier {t['tier']}: {t['label']} — Accuracy {t['accuracy']}</div>
    <p style="margin:8px 0;font-size:14px">{t['description']}</p>
    <div style="font-size:13px;color:var(--text-muted);margin:8px 0">
      <strong>Conditions:</strong> {t['conditions']}<br>
      <strong>Collapse:</strong> cos_sim {t['cos_sim']}, eff_dim {t['eff_dim']}
    </div>
    <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap">
"""
        for mode in t["modes"]:
            color = FAILURE_MODE_COLORS.get(mode, "#666")
            label = FAILURE_MODE_LABELS.get(mode, mode)
            mc = mode_counts.get(mode, 0)
            html += f'      <span class="badge" style="background:{color}33;color:{color}">{label} ({mc})</span>\n'
        html += """    </div>
  </div>
"""

    html += "</div>\n"

    # ==================== JavaScript ====================
    # Embed data for the example browser
    browser_data = []
    for r in all_records:
        entry = {
            "cond": get_condition(r),
            "cl": r["context_length"],
            "cat": r.get("category", ""),
            "q": r.get("question", ""),
            "exp": r.get("expected_answer", ""),
            "gen": r.get("generated_answer", ""),
            "ok": r.get("answer_correct", False),
            "lp": round(r.get("answer_log_prob", 0), 3),
            "mode": r.get("failure_mode", "") if not r.get("answer_correct", True) else "",
        }
        cs = get_collapse_summary(r)
        entry["cs"] = cs["cos_sim"]
        entry["ed"] = cs["eff_dim"]
        browser_data.append(entry)

    cond_labels_json = json.dumps(CONDITION_LABELS)
    mode_labels_json = json.dumps(FAILURE_MODE_LABELS)
    mode_colors_json = json.dumps(FAILURE_MODE_COLORS)
    data_json = json.dumps(browser_data)

    html += f"""
<script>
const DATA = {data_json};
const COND_LABELS = {cond_labels_json};
const MODE_LABELS = {mode_labels_json};
const MODE_COLORS = {mode_colors_json};

function showTab(id) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  document.querySelectorAll('.tab').forEach(el => {{
    if (el.textContent.toLowerCase().replace(/\\s/g,'').includes(id.replace(/\\s/g,''))) el.classList.add('active');
  }});
  // Fix tab active state
  const tabs = document.querySelectorAll('.tab');
  const tabMap = {{'overview':0,'accuracy':1,'modes':2,'examples':3,'tiers':4}};
  tabs.forEach((t,i) => t.classList.toggle('active', i === tabMap[id]));

  if (id === 'examples') filterExamples();
}}

function esc(s) {{
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}}

function filterExamples() {{
  const cond = document.getElementById('filter-condition').value;
  const corr = document.getElementById('filter-correct').value;
  const cat = document.getElementById('filter-category').value;
  const mode = document.getElementById('filter-mode').value;

  let filtered = DATA.filter(d => {{
    if (cond !== 'all' && d.cond !== cond) return false;
    if (corr === 'failures' && d.ok) return false;
    if (corr === 'correct' && !d.ok) return false;
    if (cat !== 'all' && d.cat !== cat) return false;
    if (mode !== 'all' && d.mode !== mode) return false;
    return true;
  }});

  // Sort: failures first, then by context length desc
  filtered.sort((a,b) => {{
    if (a.ok !== b.ok) return a.ok ? 1 : -1;
    return b.cl - a.cl;
  }});

  const total = filtered.length;
  const shown = Math.min(total, 100);
  document.getElementById('example-count').textContent =
    `Showing ${{shown}} of ${{total}} results` + (total > 100 ? ' (first 100)' : '');

  const container = document.getElementById('example-list');
  let html = '';

  for (let i = 0; i < shown; i++) {{
    const d = filtered[i];
    const condLabel = COND_LABELS[d.cond] || d.cond;
    const clLabel = d.cl >= 1000 ? (d.cl/1000) + 'K' : d.cl;
    const genClass = d.ok ? 'example-generated correct' : 'example-generated';
    let gen = d.gen;
    if (gen.length > 200) gen = gen.substring(0, 200) + '...';

    let modeHtml = '';
    if (d.mode) {{
      const mLabel = MODE_LABELS[d.mode] || d.mode;
      const mColor = MODE_COLORS[d.mode] || '#666';
      modeHtml = `<span class="badge" style="background:${{mColor}}33;color:${{mColor}}">${{mLabel}}</span>`;
    }}

    const icon = d.ok ? '&#10003;' : '&#10007;';
    const iconColor = d.ok ? 'var(--green)' : 'var(--red)';

    html += `<div class="example">
      <div class="example-header">
        <span class="example-meta"><span style="color:${{iconColor}};font-weight:bold">${{icon}}</span> ${{condLabel}} &middot; ${{clLabel}} tokens &middot; ${{d.cat}}</span>
        <span style="display:flex;gap:6px;align-items:center">
          ${{modeHtml}}
          <span class="example-meta">cos_sim: ${{d.cs}} &middot; eff_dim: ${{d.ed}} &middot; log_prob: ${{d.lp}}</span>
        </span>
      </div>
      <div class="example-q">Q: ${{esc(d.q)}}</div>
      <div class="example-expected">Expected: ${{esc(d.exp)}}</div>
      <div class="${{genClass}}">Generated: ${{esc(gen)}}</div>
    </div>`;
  }}

  container.innerHTML = html;
}}

// Initialize example browser on load
document.addEventListener('DOMContentLoaded', () => {{
  filterExamples();
}});
</script>
"""

    html += """
</body>
</html>"""

    return html


def _escape(s):
    """HTML-escape a string."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def main():
    os.chdir(Path(__file__).resolve().parent.parent.parent)
    print("Loading raw trial data...")
    all_records = collect_all_data()
    print(f"  Loaded {len(all_records)} total evaluations")

    failures = [r for r in all_records if not r.get("answer_correct", True)]
    print(f"  {len(failures)} failures")

    output_dir = RESULTS_BASE / "failure_dashboard"
    output_dir.mkdir(parents=True, exist_ok=True)

    html = build_html(all_records)

    output_path = output_dir / "index.html"
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Dashboard written to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
