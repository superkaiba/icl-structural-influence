#!/bin/bash
# Launch T-030, T-031, T-033 experiments with trajectory tracking
# 4x H200 GPUs, run experiments in parallel

cd /root/projects/icl-structural-influence
export PYTHONPATH=.

SWEEP_C_LENGTHS="0,100,500,2000,10000,50000,150000,262144"
SWEEP_C_LENGTHS_SHORT="0,100,500,2000,10000,50000"
MODEL="Qwen/Qwen3.5-9B"
BASE_DIR="results/safety_collapse_sweep_v2/context_type"

run_experiment() {
    local ctx_type=$1
    local gpu=$2
    local lengths=$3
    local output_dir="${BASE_DIR}/${ctx_type}"
    local log_file="${BASE_DIR}/${ctx_type}.log"

    # Skip if already complete
    if [ -f "${output_dir}/all_results.json" ]; then
        echo "[$(date)] SKIP ${ctx_type} — already has all_results.json"
        return 0
    fi

    echo "[$(date)] Starting ${ctx_type} on GPU ${gpu} -> ${log_file}"

    CUDA_VISIBLE_DEVICES=${gpu} python -u experiments/core/run_safety_collapse_experiment.py \
        --model "${MODEL}" \
        --context-types "${ctx_type}" \
        --wrapping-modes raw \
        --context-lengths "${lengths}" \
        --n-trials 3 \
        --output-dir "${output_dir}" \
        --track-trajectory \
        --max-new-tokens 500 \
        > "${log_file}" 2>&1
    local rc=$?

    echo "[$(date)] Finished ${ctx_type} (exit code: ${rc})"
    return ${rc}
}

# ── Phase 1: T-030 — Small vocab random tokens (6 experiments, 4 GPUs) ──
echo "=== Phase 1: T-030 — Small Vocab Random Tokens ==="
echo "Started at $(date)"

# GPU 0: random_tokens_2, then random_tokens_10
(
    run_experiment "random_tokens_2" 0 "${SWEEP_C_LENGTHS}" || true
    run_experiment "random_tokens_10" 0 "${SWEEP_C_LENGTHS}" || true
) &
PID_GPU0=$!

# GPU 1: random_tokens_3, then random_tokens_12
(
    run_experiment "random_tokens_3" 1 "${SWEEP_C_LENGTHS}" || true
    run_experiment "random_tokens_12" 1 "${SWEEP_C_LENGTHS}" || true
) &
PID_GPU1=$!

# GPU 2: random_tokens_5
(
    run_experiment "random_tokens_5" 2 "${SWEEP_C_LENGTHS}" || true
) &
PID_GPU2=$!

# GPU 3: random_tokens_8
(
    run_experiment "random_tokens_8" 3 "${SWEEP_C_LENGTHS}" || true
) &
PID_GPU3=$!

echo "T-030 PIDs: GPU0=$PID_GPU0, GPU1=$PID_GPU1, GPU2=$PID_GPU2, GPU3=$PID_GPU3"
wait $PID_GPU0 $PID_GPU1 $PID_GPU2 $PID_GPU3
echo "=== T-030 Complete at $(date) ==="

# ── Phase 2: T-031 + T-033 — Adversarial + structure sweep (7 experiments) ──
echo ""
echo "=== Phase 2: T-031 + T-033 — Adversarial + Structure Sweep ==="
echo "Started at $(date)"

# GPU 0: p0, p50
(
    run_experiment "structured_walk_15_p0" 0 "${SWEEP_C_LENGTHS}" || true
    run_experiment "structured_walk_15_p50" 0 "${SWEEP_C_LENGTHS}" || true
) &
PID_GPU0=$!

# GPU 1: p15, p65
(
    run_experiment "structured_walk_15_p15" 1 "${SWEEP_C_LENGTHS}" || true
    run_experiment "structured_walk_15_p65" 1 "${SWEEP_C_LENGTHS}" || true
) &
PID_GPU1=$!

# GPU 2: p30, p95
(
    run_experiment "structured_walk_15_p30" 2 "${SWEEP_C_LENGTHS}" || true
    run_experiment "structured_walk_15_p95" 2 "${SWEEP_C_LENGTHS}" || true
) &
PID_GPU2=$!

# GPU 3: least_probable_tokens (shorter lengths, slower per-token)
(
    run_experiment "least_probable_tokens" 3 "${SWEEP_C_LENGTHS_SHORT}" || true
) &
PID_GPU3=$!

echo "Phase 2 PIDs: GPU0=$PID_GPU0, GPU1=$PID_GPU1, GPU2=$PID_GPU2, GPU3=$PID_GPU3"
wait $PID_GPU0 $PID_GPU1 $PID_GPU2 $PID_GPU3
echo "=== Phase 2 Complete at $(date) ==="

echo ""
echo "All experiments complete at $(date)"

# Summary: check which experiments produced results
echo ""
echo "=== Results Summary ==="
for d in ${BASE_DIR}/random_tokens_{2,3,5,8,10,12} ${BASE_DIR}/structured_walk_15_p{0,15,30,50,65,95} ${BASE_DIR}/least_probable_tokens; do
    name=$(basename $d)
    if [ -f "$d/all_results.json" ]; then
        echo "  OK: $name"
    else
        echo "  FAIL: $name"
    fi
done
