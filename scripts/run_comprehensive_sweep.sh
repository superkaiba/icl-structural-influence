#!/bin/bash
# Safety Collapse Sweep v2: Star-shaped experiment design
#
# Run from project root: bash scripts/run_comprehensive_sweep.sh [sweep_name]
# sweep_name: context_length, model_size, context_type, architecture, all (default)
set -e

SWEEP="${1:-all}"

echo "=========================================="
echo "SAFETY COLLAPSE SWEEP v2"
echo "Sweep: $SWEEP"
echo "=========================================="

# Sweep C: Context types (fastest to run first — single model, 8 lengths)
if [ "$SWEEP" = "context_type" ] || [ "$SWEEP" = "all" ]; then
    echo -e "\n== Sweep C: Context Types (Qwen3.5-9B, 12 types x 8 lengths) =="
    PYTHONPATH=. python -u experiments/core/run_safety_sweep.py --sweep context_type
fi

# Sweep B: Model sizes (Qwen3.5 family)
if [ "$SWEEP" = "model_size" ] || [ "$SWEEP" = "all" ]; then
    echo -e "\n== Sweep B: Model Size (Qwen3.5 0.8B-27B) =="
    PYTHONPATH=. python -u experiments/core/run_safety_sweep.py --sweep model_size
fi

# Sweep D: Architecture comparison
if [ "$SWEEP" = "architecture" ] || [ "$SWEEP" = "all" ]; then
    echo -e "\n== Sweep D: Architecture (Qwen3.5-9B, Llama-3.3-70B, OLMo-3-7B) =="
    PYTHONPATH=. python -u experiments/core/run_safety_sweep.py --sweep architecture
fi

# Sweep A: Context length (if not already covered by B)
if [ "$SWEEP" = "context_length" ] || [ "$SWEEP" = "all" ]; then
    echo -e "\n== Sweep A: Context Length (Qwen3.5-9B, full 18-point range) =="
    PYTHONPATH=. python -u experiments/core/run_safety_sweep.py --sweep context_length
fi

echo -e "\n=========================================="
echo "ALL SWEEPS COMPLETE"
echo "=========================================="

# Generate plots
echo -e "\n== Generating Plots =="
PYTHONPATH=. python experiments/plotting/plot_comprehensive_sweep.py
