#!/bin/bash
# GPU scheduler v2: properly tracks GPU assignment via PID files
cd /root/projects/icl-structural-influence
export PYTHONPATH=.

SWEEP_C_LENGTHS="0,100,500,2000,10000,50000,150000,262144"
SWEEP_C_LENGTHS_SHORT="0,100,500,2000,10000,50000"
MODEL="Qwen/Qwen3.5-9B"
BASE_DIR="results/safety_collapse_sweep_v2/context_type"
PID_DIR="/tmp/gpu_pids"
mkdir -p "$PID_DIR"

launch_on_gpu() {
    local ctx_type=$1
    local gpu=$2
    local lengths=$3
    local output_dir="${BASE_DIR}/${ctx_type}"
    local log_file="${BASE_DIR}/${ctx_type}.log"

    if [ -f "${output_dir}/all_results.json" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP ${ctx_type} (already complete)"
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] START ${ctx_type} on GPU ${gpu}"

    CUDA_VISIBLE_DEVICES=${gpu} python -u experiments/core/run_safety_collapse_experiment.py \
        --model "${MODEL}" \
        --context-types "${ctx_type}" \
        --wrapping-modes raw \
        --context-lengths "${lengths}" \
        --n-trials 3 \
        --output-dir "${output_dir}" \
        --track-trajectory \
        --max-new-tokens 500 \
        > "${log_file}" 2>&1 &
    local pid=$!
    echo "$pid" > "${PID_DIR}/gpu${gpu}.pid"
    echo "[$(date '+%H:%M:%S')] ${ctx_type} PID=${pid} GPU=${gpu}"
}

gpu_is_free() {
    local gpu=$1
    local pidfile="${PID_DIR}/gpu${gpu}.pid"
    if [ ! -f "$pidfile" ]; then
        return 0  # No PID file = free
    fi
    local pid=$(cat "$pidfile")
    if ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$pidfile"
        return 0  # Process dead = free
    fi
    return 1  # Process alive = busy
}

# Register already-running experiments
# random_tokens_10 on GPU 3 (PID 79035)
echo "79035" > "${PID_DIR}/gpu3.pid"
# random_tokens_12 on GPU 1 (PID 80531)
echo "80531" > "${PID_DIR}/gpu1.pid"

# Queue of remaining experiments
QUEUE=(
    "structured_walk_15_p0:${SWEEP_C_LENGTHS}"
    "structured_walk_15_p15:${SWEEP_C_LENGTHS}"
    "structured_walk_15_p30:${SWEEP_C_LENGTHS}"
    "structured_walk_15_p50:${SWEEP_C_LENGTHS}"
    "structured_walk_15_p65:${SWEEP_C_LENGTHS}"
    "structured_walk_15_p95:${SWEEP_C_LENGTHS}"
    "least_probable_tokens:${SWEEP_C_LENGTHS_SHORT}"
)

echo "=== GPU Scheduler v2 ==="
echo "Already running: random_tokens_10 (GPU 3), random_tokens_12 (GPU 1)"
echo "Queue: ${#QUEUE[@]} experiments"
echo ""

QUEUE_IDX=0

while [ $QUEUE_IDX -lt ${#QUEUE[@]} ]; do
    for gpu in 0 2 1 3; do  # Try free GPUs first (0, 2)
        if [ $QUEUE_IDX -ge ${#QUEUE[@]} ]; then
            break
        fi
        if gpu_is_free "$gpu"; then
            entry="${QUEUE[$QUEUE_IDX]}"
            ctx_type="${entry%%:*}"
            lengths="${entry#*:}"
            launch_on_gpu "$ctx_type" "$gpu" "$lengths"
            QUEUE_IDX=$((QUEUE_IDX + 1))
            sleep 3
        fi
    done

    if [ $QUEUE_IDX -lt ${#QUEUE[@]} ]; then
        sleep 30
    fi
done

echo ""
echo "[$(date '+%H:%M:%S')] All ${#QUEUE[@]} experiments launched. Waiting..."
wait
echo "[$(date '+%H:%M:%S')] All done!"

# Summary
echo ""
echo "=== Results ==="
for d in random_tokens_{2,3,5,8,10,12} structured_walk_15_p{0,15,30,50,65,95} least_probable_tokens; do
    if [ -f "${BASE_DIR}/${d}/all_results.json" ]; then
        echo "  OK: $d"
    else
        echo "  FAIL: $d"
    fi
done
