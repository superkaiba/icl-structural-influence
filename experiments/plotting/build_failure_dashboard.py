#!/usr/bin/env python3
"""Build an interactive HTML dashboard showing failure modes across all experiment conditions,
organized by context type."""

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
    return rec.get("context_type") or rec.get("condition", "unknown")


def get_collapse_summary(rec):
    metrics = rec.get("collapse_metrics", {})
    if not metrics:
        return {"cos_sim": "N/A", "eff_dim": "N/A", "layer_details": []}
    cos_sims, eff_dims, layer_details = [], [], []
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
    gen = rec.get("generated_answer", "")
    expected = rec.get("expected_answer", "")
    if not gen:
        return "empty_response"
    if "answer in" in gen.lower() and "few words" in gen.lower():
        return "instruction_echo"
    if "as few words as possible" in gen.lower():
        return "instruction_echo"
    if len(gen) > 5:
        unique_chars = set(gen.replace("-", "").replace(" ", "").replace(".", ""))
        if len(unique_chars) <= 2 and len(gen) > 8:
            return "token_repetition"
    if gen.startswith("```") or gen.startswith("``"):
        return "code_artifact"
    refusal_phrases = [
        "does not contain", "not mentioned", "no one mentioned",
        "text does not", "passage does not", "not relate to",
        "cannot be determined from", "not in the text", "does not relate"
    ]
    if any(p in gen.lower() for p in refusal_phrases):
        return "context_overfitting"
    if any('\u4e00' <= c <= '\u9fff' for c in gen):
        return "language_switch"
    if "Human:" in gen or "Assistant:" in gen:
        return "conversation_artifact"
    if len(gen) > 50 and gen.count(expected) > 1:
        return "answer_loop"
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
    "instruction_echo": "Model repeats the instruction prompt verbatim instead of answering.",
    "token_repetition": "Model emits repeated digits or characters (e.g., '155555...', '6666...').",
    "code_artifact": "Model generates code block markers (```python) before attempting to answer.",
    "context_overfitting": "Model refuses to use world knowledge, insisting the answer must come from the provided context.",
    "language_switch": "Model answers correctly but in the wrong language (e.g. Chinese instead of English).",
    "conversation_artifact": "Model generates conversation markers like 'Human:' or 'Assistant:' in its response.",
    "answer_loop": "Model generates the correct answer but then repeats it in a loop.",
    "wrong_answer": "Model generates a coherent but incorrect answer (wrong facts, arithmetic errors, synonym issues).",
    "empty_response": "Model generates an empty or whitespace-only response.",
}

CONDITION_ORDER = [
    "no_context", "natural_books", "natural_books_extended",
    "single_topic", "multi_topic_300", "multi_topic_1000",
    "misspell_10pct", "misspell_25pct", "misspell_50pct",
    "structured_walk", "repeated_token",
]

CONDITION_LABELS = {
    "no_context": "No Context (Baseline)",
    "natural_books": "Natural Books (0.5-20K)",
    "natural_books_extended": "Natural Books (50-128K)",
    "single_topic": "Single Topic Book",
    "multi_topic_300": "Multi-Topic (300 tok/switch)",
    "multi_topic_1000": "Multi-Topic (1000 tok/switch)",
    "misspell_10pct": "10% Misspellings",
    "misspell_25pct": "25% Misspellings",
    "misspell_50pct": "50% Misspellings",
    "structured_walk": "Structured Graph Walk",
    "repeated_token": "Repeated Token",
}

CONDITION_DESCRIPTIONS = {
    "no_context": "Baseline: questions asked with no preceding context. Establishes the model's native accuracy.",
    "natural_books": "Context filled with natural book text (Project Gutenberg). Tests whether diverse natural language causes any collapse.",
    "natural_books_extended": "Same as Natural Books but extended to very long context (50K-128K tokens).",
    "single_topic": "Context filled with text from a single book repeated. Tests whether monotonic content causes over-fitting at long lengths.",
    "multi_topic_300": "Context switches between different books every 300 tokens. Tests whether topic diversity protects against long-context degradation.",
    "multi_topic_1000": "Context switches between different books every 1000 tokens. Coarser topic switching than the 300-token variant.",
    "misspell_10pct": "Natural book text with 10% of words randomly misspelled. Tests resilience to mild noise.",
    "misspell_25pct": "Natural book text with 25% of words randomly misspelled. Tests resilience to moderate noise.",
    "misspell_50pct": "Natural book text with 50% of words randomly misspelled. Tests resilience to heavy noise.",
    "structured_walk": "Context is a random walk on a structured graph with limited vocabulary (~50 node names). Known to induce representational collapse.",
    "repeated_token": "Context is a single token repeated. Extreme collapse inducer — representations converge to a single point.",
}

# Group conditions for the "By Context" view
CONDITION_GROUPS = [
    {
        "id": "collapse",
        "label": "Collapse-Inducing Contexts",
        "color": "#f85149",
        "description": "Structured or repetitive contexts that induce representational collapse in hidden states.",
        "conditions": ["structured_walk", "repeated_token"],
    },
    {
        "id": "natural",
        "label": "Natural Language Contexts",
        "color": "#3fb950",
        "description": "Clean natural language from books. Minimal collapse expected.",
        "conditions": ["no_context", "natural_books", "natural_books_extended"],
    },
    {
        "id": "topics",
        "label": "Topic Variation",
        "color": "#58a6ff",
        "description": "Testing whether topic switches within context protect against long-context degradation.",
        "conditions": ["single_topic", "multi_topic_300", "multi_topic_1000"],
    },
    {
        "id": "misspell",
        "label": "Misspelling Noise",
        "color": "#bc8cff",
        "description": "Natural book text with varying rates of random misspellings.",
        "conditions": ["misspell_10pct", "misspell_25pct", "misspell_50pct"],
    },
]


def collect_all_data():
    all_records = []
    for label, raw_dir in [
        ("probing_collapse", RESULTS_BASE / "probing_collapse_performance" / "raw"),
        ("misspellings", RESULTS_BASE / "context_variation_v2_fixed" / "misspellings" / "raw"),
        ("topic_changes", RESULTS_BASE / "context_variation_v2_fixed" / "topic_changes" / "raw"),
        ("extended_length", RESULTS_BASE / "context_variation_maxctx" / "extended_length" / "raw"),
    ]:
        records = load_raw_trials(raw_dir)
        for r in records:
            r["experiment"] = label
        all_records.extend(records)
    return all_records


def _escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _cl_label(cl):
    if cl >= 1000:
        return f"{cl//1000}K"
    return str(cl)


def _acc_color(acc):
    if acc >= 95:
        return "rgba(63,185,80,0.2)", "var(--green)"
    elif acc >= 85:
        return "rgba(210,153,34,0.15)", "var(--orange)"
    elif acc >= 50:
        return "rgba(248,81,73,0.15)", "var(--red)"
    else:
        return "rgba(248,81,73,0.3)", "var(--red)"


def _cos_color(val):
    if val >= 0.9:
        return "rgba(248,81,73,0.3)", "var(--red)"
    elif val >= 0.6:
        return "rgba(248,81,73,0.15)", "var(--red)"
    elif val >= 0.4:
        return "rgba(210,153,34,0.15)", "var(--orange)"
    else:
        return "rgba(63,185,80,0.15)", "var(--green)"


def build_html(all_records):
    # Classify failures
    for r in all_records:
        if not r.get("answer_correct", True):
            r["failure_mode"] = classify_failure(r)
        else:
            r["failure_mode"] = ""

    failures = [r for r in all_records if not r.get("answer_correct", True)]

    # Pre-compute data structures
    all_conditions = sorted(
        set(get_condition(r) for r in all_records),
        key=lambda c: CONDITION_ORDER.index(c) if c in CONDITION_ORDER else 99
    )
    all_lengths = sorted(set(r["context_length"] for r in all_records))

    # By condition
    records_by_cond = defaultdict(list)
    for r in all_records:
        records_by_cond[get_condition(r)].append(r)

    # Accuracy data
    accuracy_data = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    for r in all_records:
        cond = get_condition(r)
        cl = r["context_length"]
        accuracy_data[cond][cl]["total"] += 1
        if r.get("answer_correct", False):
            accuracy_data[cond][cl]["correct"] += 1

    # Collapse data (layer 27)
    collapse_data = defaultdict(lambda: defaultdict(list))
    for r in all_records:
        cond = get_condition(r)
        cl = r["context_length"]
        metrics = r.get("collapse_metrics", {})
        if "27" in metrics:
            collapse_data[cond][cl].append(metrics["27"].get("avg_cos_sim", 0))

    # Mode counts
    mode_counts = defaultdict(int)
    for f in failures:
        mode_counts[f["failure_mode"]] += 1
    sorted_modes = sorted(mode_counts.items(), key=lambda x: -x[1])

    total = len(all_records)
    total_fail = len(failures)
    total_correct = total - total_fail

    # ===== BUILD HTML =====
    html = CSS_AND_HEAD

    # Tabs - one per group plus overview and browser
    html += """
<div class="tabs" id="main-tabs">
  <div class="tab active" onclick="showTab('overview')">Overview</div>
"""
    for group in CONDITION_GROUPS:
        html += f'  <div class="tab" onclick="showTab(\'{group["id"]}\')">{group["label"]}</div>\n'
    html += """  <div class="tab" onclick="showTab('examples')">Example Browser</div>
</div>
"""

    # ==================== OVERVIEW TAB ====================
    html += f"""
<div id="tab-overview" class="tab-content active">
  <div class="stats-grid">
    <div class="stat-card"><div class="stat-value">{total:,}</div><div class="stat-label">Total Evaluations</div></div>
    <div class="stat-card"><div class="stat-value" style="color:var(--green)">{total_correct:,}</div><div class="stat-label">Correct</div></div>
    <div class="stat-card"><div class="stat-value" style="color:var(--red)">{total_fail:,}</div><div class="stat-label">Failures</div></div>
    <div class="stat-card"><div class="stat-value">{len(all_conditions)}</div><div class="stat-label">Conditions</div></div>
  </div>

  <h2>Accuracy Heatmap</h2>
  <div class="card"><div class="table-scroll"><table>
    <tr><th>Condition</th>"""

    for cl in all_lengths:
        html += f"<th>{_cl_label(cl)}</th>"
    html += "</tr>\n"

    for cond in all_conditions:
        label = CONDITION_LABELS.get(cond, cond)
        html += f"    <tr><td><strong>{label}</strong></td>"
        for cl in all_lengths:
            d = accuracy_data[cond][cl]
            if d["total"] == 0:
                html += "<td style='color:var(--text-muted)'>—</td>"
            else:
                acc = d["correct"] / d["total"] * 100
                bg, fg = _acc_color(acc)
                html += f"<td><div class='heatmap-cell' style='background:{bg};color:{fg}'>{acc:.0f}%</div></td>"
        html += "</tr>\n"

    html += """  </table></div></div>

  <h2>Cosine Similarity Heatmap (Layer 27)</h2>
  <div class="card"><div class="table-scroll"><table>
    <tr><th>Condition</th>"""

    for cl in all_lengths:
        html += f"<th>{_cl_label(cl)}</th>"
    html += "</tr>\n"

    for cond in all_conditions:
        label = CONDITION_LABELS.get(cond, cond)
        html += f"    <tr><td><strong>{label}</strong></td>"
        for cl in all_lengths:
            vals = collapse_data[cond][cl]
            if not vals:
                html += "<td style='color:var(--text-muted)'>—</td>"
            else:
                avg = sum(vals) / len(vals)
                bg, fg = _cos_color(avg)
                html += f"<td><div class='heatmap-cell' style='background:{bg};color:{fg}'>{avg:.2f}</div></td>"
        html += "</tr>\n"

    html += "  </table></div></div>\n"

    # Global failure mode distribution
    html += """  <h2>Failure Mode Distribution (All Conditions)</h2>
  <div class="card"><div class="bar-chart">\n"""
    max_count = max(mode_counts.values()) if mode_counts else 1
    for mode, count in sorted_modes:
        pct = count / total_fail * 100 if total_fail > 0 else 0
        bar_pct = count / max_count * 100
        color = FAILURE_MODE_COLORS.get(mode, "#666")
        label = FAILURE_MODE_LABELS.get(mode, mode)
        html += f"""    <div class="bar-row">
      <div class="bar-label">{label}</div>
      <div class="bar-track"><div class="bar-fill" style="width:{bar_pct:.1f}%;background:{color}">{count}</div></div>
      <div class="bar-value">{pct:.1f}%</div>
    </div>\n"""
    html += "  </div></div>\n</div>\n"

    # ==================== PER-GROUP TABS ====================
    for group in CONDITION_GROUPS:
        gid = group["id"]
        html += f"""
<div id="tab-{gid}" class="tab-content">
  <h2 style="border-bottom-color:{group['color']}">{group['label']}</h2>
  <p class="group-desc">{group['description']}</p>
"""
        for cond in group["conditions"]:
            cond_records = records_by_cond.get(cond, [])
            if not cond_records:
                continue

            cond_failures = [r for r in cond_records if not r.get("answer_correct", True)]
            cond_lengths = sorted(set(r["context_length"] for r in cond_records))
            cond_total = len(cond_records)
            cond_fail_count = len(cond_failures)
            cond_label = CONDITION_LABELS.get(cond, cond)
            cond_desc = CONDITION_DESCRIPTIONS.get(cond, "")

            # Accuracy range
            accs = []
            for cl in cond_lengths:
                d = accuracy_data[cond][cl]
                if d["total"] > 0:
                    accs.append(d["correct"] / d["total"] * 100)
            acc_min = min(accs) if accs else 0
            acc_max = max(accs) if accs else 0

            html += f"""
  <div class="condition-section">
    <div class="condition-header" onclick="toggleSection(this)">
      <div class="condition-header-left">
        <span class="expand-icon">&#9660;</span>
        <h3 style="margin:0">{cond_label}</h3>
      </div>
      <div class="condition-header-right">
        <span class="badge" style="background:var(--surface2);color:var(--text-muted)">{cond_total} evals</span>
        <span class="badge" style="background:{'rgba(248,81,73,0.2)' if cond_fail_count > 0 else 'rgba(63,185,80,0.2)'};color:{'var(--red)' if cond_fail_count > 0 else 'var(--green)'}">{cond_fail_count} failures ({cond_fail_count/cond_total*100:.0f}%)</span>
        <span class="badge" style="background:var(--surface2);color:var(--text-muted)">Acc: {acc_min:.0f}-{acc_max:.0f}%</span>
      </div>
    </div>
    <div class="condition-body">
      <p class="condition-desc">{cond_desc}</p>
"""

            # Accuracy + collapse table for this condition
            html += """      <div class="mini-tables">
        <div class="card">
          <h4>Performance by Context Length</h4>
          <table>
            <tr><th>Context</th><th>Accuracy</th><th>Failures</th><th>cos_sim (L27)</th><th>eff_dim (L27)</th></tr>
"""
            for cl in cond_lengths:
                d = accuracy_data[cond][cl]
                if d["total"] == 0:
                    continue
                acc = d["correct"] / d["total"] * 100
                n_fail = d["total"] - d["correct"]
                bg, fg = _acc_color(acc)

                cos_vals = collapse_data[cond][cl]
                if cos_vals:
                    avg_cos = sum(cos_vals) / len(cos_vals)
                    cos_str = f"{avg_cos:.3f}"
                    cbg, cfg = _cos_color(avg_cos)
                else:
                    cos_str = "—"
                    cbg, cfg = "transparent", "var(--text-muted)"

                # eff_dim from layer 27
                ed_vals = []
                for r in cond_records:
                    if r["context_length"] == cl:
                        m = r.get("collapse_metrics", {})
                        if "27" in m:
                            ed_vals.append(m["27"].get("effective_dim", 0))
                ed_str = f"{sum(ed_vals)/len(ed_vals):.1f}" if ed_vals else "—"

                html += f"""            <tr>
              <td>{_cl_label(cl)}</td>
              <td><div class="heatmap-cell" style="background:{bg};color:{fg}">{acc:.1f}%</div></td>
              <td style="color:{'var(--red)' if n_fail > 0 else 'var(--green)'}">{n_fail}/{d['total']}</td>
              <td><div class="heatmap-cell" style="background:{cbg};color:{cfg}">{cos_str}</div></td>
              <td>{ed_str}</td>
            </tr>\n"""

            html += "          </table>\n        </div>\n"

            # Failure mode breakdown for this condition
            if cond_failures:
                cond_mode_counts = defaultdict(int)
                for f in cond_failures:
                    cond_mode_counts[f["failure_mode"]] += 1
                cond_sorted_modes = sorted(cond_mode_counts.items(), key=lambda x: -x[1])
                cond_max = max(cond_mode_counts.values())

                html += """        <div class="card">
          <h4>Failure Modes</h4>
          <div class="bar-chart">\n"""
                for mode, count in cond_sorted_modes:
                    bar_pct = count / cond_max * 100
                    color = FAILURE_MODE_COLORS.get(mode, "#666")
                    mlabel = FAILURE_MODE_LABELS.get(mode, mode)
                    pct = count / cond_fail_count * 100
                    html += f"""            <div class="bar-row">
              <div class="bar-label">{mlabel}</div>
              <div class="bar-track"><div class="bar-fill" style="width:{bar_pct:.1f}%;background:{color}">{count}</div></div>
              <div class="bar-value">{pct:.0f}%</div>
            </div>\n"""
                html += "          </div>\n        </div>\n"

                # Failure mode evolution over context length
                html += """        <div class="card">
          <h4>Failure Modes by Context Length</h4>
          <table>
            <tr><th>Context</th>"""
                # Get all modes present in this condition
                all_cond_modes = [m for m, _ in cond_sorted_modes]
                for mode in all_cond_modes:
                    color = FAILURE_MODE_COLORS.get(mode, "#666")
                    mlabel = FAILURE_MODE_LABELS.get(mode, mode)
                    html += f'<th><span style="color:{color}">{mlabel}</span></th>'
                html += "<th>Total</th></tr>\n"

                for cl in cond_lengths:
                    cl_failures = [f for f in cond_failures if f["context_length"] == cl]
                    if not cl_failures:
                        continue
                    html += f"            <tr><td>{_cl_label(cl)}</td>"
                    for mode in all_cond_modes:
                        mc = sum(1 for f in cl_failures if f["failure_mode"] == mode)
                        if mc > 0:
                            color = FAILURE_MODE_COLORS.get(mode, "#666")
                            html += f'<td style="color:{color};font-weight:600">{mc}</td>'
                        else:
                            html += '<td style="color:var(--text-muted)">0</td>'
                    html += f"<td style='font-weight:600'>{len(cl_failures)}</td></tr>\n"
                html += "          </table>\n        </div>\n"

            html += "      </div>\n"  # end mini-tables

            # Example failures grouped by context length
            if cond_failures:
                html += """      <div class="card">
        <h4>Example Failures</h4>\n"""

                # Group by context length, show up to 4 per length
                for cl in cond_lengths:
                    cl_failures = [f for f in cond_failures if f["context_length"] == cl]
                    if not cl_failures:
                        continue

                    # Sort by failure mode for grouping
                    cl_failures.sort(key=lambda f: f["failure_mode"])

                    # Pick diverse examples (different modes, different questions)
                    seen_modes = set()
                    seen_questions = set()
                    examples = []
                    for f in cl_failures:
                        if len(examples) >= 5:
                            break
                        mode = f["failure_mode"]
                        q = f.get("question", "")
                        if mode not in seen_modes:
                            seen_modes.add(mode)
                            seen_questions.add(q)
                            examples.append(f)
                        elif q not in seen_questions and len(examples) < 5:
                            seen_questions.add(q)
                            examples.append(f)

                    n_more = len(cl_failures) - len(examples)
                    html += f"""        <div class="length-group">
          <div class="length-label">{_cl_label(cl)} tokens <span class="length-count">({len(cl_failures)} failures)</span></div>\n"""

                    for ex in examples:
                        mode = ex["failure_mode"]
                        color = FAILURE_MODE_COLORS.get(mode, "#666")
                        mlabel = FAILURE_MODE_LABELS.get(mode, mode)
                        cat = ex.get("category", "")
                        gen = ex.get("generated_answer", "")
                        if len(gen) > 150:
                            gen = gen[:150] + "..."
                        collapse = get_collapse_summary(ex)

                        html += f"""          <div class="example">
            <div class="example-header">
              <span class="example-meta">{cat} &middot; <span class="badge" style="background:{color}33;color:{color};font-size:11px">{mlabel}</span></span>
              <span class="example-meta">cos_sim: {collapse['cos_sim']} &middot; eff_dim: {collapse['eff_dim']}</span>
            </div>
            <div class="example-q">Q: {_escape(ex.get('question', ''))}</div>
            <div class="example-expected">Expected: {_escape(ex.get('expected_answer', ''))}</div>
            <div class="example-generated">Generated: {_escape(gen)}</div>
          </div>\n"""

                    if n_more > 0:
                        html += f'          <div class="more-link">+ {n_more} more failures at this length (see Example Browser)</div>\n'
                    html += "        </div>\n"

                html += "      </div>\n"

            html += "    </div>\n  </div>\n"  # end condition-body, condition-section

        html += "</div>\n"  # end tab

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

    # ==================== JavaScript + Data ====================
    browser_data = []
    for r in all_records:
        cs = get_collapse_summary(r)
        browser_data.append({
            "cond": get_condition(r),
            "cl": r["context_length"],
            "cat": r.get("category", ""),
            "q": r.get("question", ""),
            "exp": r.get("expected_answer", ""),
            "gen": r.get("generated_answer", ""),
            "ok": r.get("answer_correct", False),
            "lp": round(r.get("answer_log_prob", 0), 3),
            "mode": r.get("failure_mode", ""),
            "cs": cs["cos_sim"],
            "ed": cs["eff_dim"],
        })

    # Build tab index map for JS
    tab_ids = ["overview"] + [g["id"] for g in CONDITION_GROUPS] + ["examples"]
    tab_map_json = json.dumps({tid: i for i, tid in enumerate(tab_ids)})

    html += f"""
<script>
const DATA = {json.dumps(browser_data)};
const COND_LABELS = {json.dumps(CONDITION_LABELS)};
const MODE_LABELS = {json.dumps(FAILURE_MODE_LABELS)};
const MODE_COLORS = {json.dumps(FAILURE_MODE_COLORS)};
const TAB_MAP = {tab_map_json};

function showTab(id) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  const tabs = document.querySelectorAll('#main-tabs .tab');
  const idx = TAB_MAP[id];
  tabs.forEach((t, i) => t.classList.toggle('active', i === idx));
  if (id === 'examples') filterExamples();
}}

function toggleSection(header) {{
  const body = header.nextElementSibling;
  const icon = header.querySelector('.expand-icon');
  const isOpen = body.style.display !== 'none';
  body.style.display = isOpen ? 'none' : 'block';
  icon.innerHTML = isOpen ? '&#9654;' : '&#9660;';
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
document.addEventListener('DOMContentLoaded', () => {{ filterExamples(); }});
</script>
</body>
</html>"""

    return html


CSS_AND_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Failure Mode Dashboard — Representational Collapse Experiments</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --surface2: #21262d; --border: #30363d;
    --text: #e6edf3; --text-muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --orange: #d29922; --purple: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; padding: 24px; max-width: 1400px; margin: 0 auto; }
  h1 { font-size: 28px; margin-bottom: 8px; }
  h2 { font-size: 22px; margin: 32px 0 16px; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 8px; }
  h3 { font-size: 18px; margin: 24px 0 12px; }
  h4 { font-size: 14px; margin: 12px 0 8px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .subtitle { color: var(--text-muted); font-size: 14px; margin-bottom: 24px; }
  .group-desc { color: var(--text-muted); font-size: 14px; margin-bottom: 20px; }

  .tabs { display: flex; gap: 4px; margin-bottom: 24px; border-bottom: 1px solid var(--border); flex-wrap: wrap; }
  .tab { padding: 8px 16px; cursor: pointer; border: 1px solid transparent; border-bottom: none; border-radius: 6px 6px 0 0; background: transparent; color: var(--text-muted); font-size: 14px; transition: all 0.15s; }
  .tab:hover { color: var(--text); background: var(--surface); }
  .tab.active { color: var(--text); background: var(--surface); border-color: var(--border); border-bottom: 2px solid var(--accent); }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 16px; }
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; text-align: center; }
  .stat-value { font-size: 32px; font-weight: 700; }
  .stat-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }

  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 500; white-space: nowrap; }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }
  th { background: var(--surface2); color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; position: sticky; top: 0; }
  tr:hover { background: var(--surface2); }
  .table-scroll { overflow-x: auto; }
  .heatmap-cell { padding: 4px 8px; text-align: center; font-weight: 500; font-size: 13px; border-radius: 4px; display: inline-block; min-width: 50px; }

  /* Condition sections */
  .condition-section { margin-bottom: 24px; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .condition-header { display: flex; justify-content: space-between; align-items: center; padding: 14px 20px; background: var(--surface); cursor: pointer; user-select: none; flex-wrap: wrap; gap: 8px; }
  .condition-header:hover { background: var(--surface2); }
  .condition-header-left { display: flex; align-items: center; gap: 10px; }
  .condition-header-right { display: flex; gap: 8px; flex-wrap: wrap; }
  .expand-icon { font-size: 12px; color: var(--text-muted); width: 16px; }
  .condition-body { padding: 20px; background: var(--bg); }
  .condition-desc { color: var(--text-muted); font-size: 13px; margin-bottom: 16px; }
  .mini-tables { display: grid; grid-template-columns: 1fr; gap: 16px; margin-bottom: 16px; }

  /* Length group */
  .length-group { margin-bottom: 16px; }
  .length-label { font-size: 14px; font-weight: 600; padding: 6px 12px; background: var(--surface2); border-radius: 6px 6px 0 0; border: 1px solid var(--border); border-bottom: none; display: inline-block; }
  .length-count { color: var(--text-muted); font-weight: 400; font-size: 12px; }
  .more-link { color: var(--text-muted); font-size: 12px; padding: 8px 12px; font-style: italic; }

  .example { background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; padding: 14px; margin-bottom: 8px; font-size: 13px; }
  .example-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; flex-wrap: wrap; gap: 6px; }
  .example-meta { color: var(--text-muted); font-size: 11px; }
  .example-q { color: var(--accent); margin-bottom: 4px; }
  .example-expected { color: var(--green); }
  .example-generated { color: var(--red); font-family: 'SFMono-Regular', Consolas, monospace; word-break: break-all; }
  .example-generated.correct { color: var(--green); }

  .bar-chart { margin: 8px 0; }
  .bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .bar-label { width: 180px; font-size: 12px; text-align: right; flex-shrink: 0; }
  .bar-track { flex: 1; height: 24px; background: var(--surface2); border-radius: 4px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; display: flex; align-items: center; padding-left: 8px; font-size: 11px; font-weight: 600; min-width: fit-content; }
  .bar-value { font-size: 12px; width: 50px; text-align: right; flex-shrink: 0; color: var(--text-muted); }

  .filters { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; align-items: center; }
  .filter-group { display: flex; align-items: center; gap: 6px; }
  .filter-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; }
  select { background: var(--surface2); border: 1px solid var(--border); color: var(--text); padding: 4px 8px; border-radius: 4px; font-size: 13px; }
</style>
</head>
<body>
<h1>Failure Mode Dashboard</h1>
<p class="subtitle">Representational Collapse &amp; Knowledge Retrieval &mdash; Qwen2.5-7B-Instruct &mdash; organized by context type</p>
"""


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
