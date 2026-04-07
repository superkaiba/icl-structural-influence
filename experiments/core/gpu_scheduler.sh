#!/bin/bash
# GPU scheduler: launch experiments on GPUs as they become free
# Monitors GPU usage and immediately fills any idle GPU with the next experiment
cd /root/projects/icl-structural-influence
export PYTHONPATH=.

SWEEP_C_LENGTHS="0,100,500,2000,10000,50000,150000,262144"
SWEEP_C_LENGTHS_SHORT="0,100,500,2000,10000,50000"
MODEL="Qwen/Qwen3.5-9B"
BASE_DIR="results/safety_collapse_sweep_v2/context_type"

# All experiments to run (order = priority)
EXPERIMENTS=(
    # T-030: small vocab (some already running)
    "random_tokens_2:${SWEEP_C_LENGTHS}"
    "random_tokens_3:${SWEEP_C_LENGTHS}"
    "random_tokens_5:${SWEEP_C_LENGTHS}"
    "random_tokens_8:${SWEEP_C_LENGTHS}"
    "random_tokens_10:${SWEEP_C_LENGTHS}"
    "random_tokens_12:${SWEEP_C_LENGTHS}"
    # T-033: structure amount sweep
    "structured_walk_15_p0:${SWEEP_C_LENGTHS}"
    "structured_walk_15_p15:${SWEEP_C_LENGTHS}"
    "structured_walk_15_p30:${SWEEP_C_LENGTHS}"
    "structured_walk_15_p50:${SWEEP_C_LENGTHS}"
    "structured_walk_15_p65:${SWEEP_C_LENGTHS}"
    "structured_walk_15_p95:${SWEEP_C_LENGTHS}"
    # T-031: adversarial
    "least_probable_tokens:${SWEEP_C_LENGTHS_SHORT}"
)

is_complete() {
    local ctx_type=$1
    [ -f "${BASE_DIR}/${ctx_type}/all_results.json" ]
}

is_running() {
    local ctx_type=$1
    ps aux | grep "run_safety_collapse.*${ctx_type}" | grep -v grep > /dev/null 2>&1
}

gpu_has_experiment() {
    local gpu=$1
    # Check if any experiment python process is using this GPU
    # We track by checking CUDA_VISIBLE_DEVICES in /proc/*/environ
    for pid in $(pgrep -f "run_safety_collapse_experiment"); do
        cuda_dev=$(cat /proc/$pid/environ 2>/dev/null | tr '\0' '\n' | grep CUDA_VISIBLE_DEVICES | cut -d= -f2)
        if [ "$cuda_dev" = "$gpu" ]; then
            return 0  # GPU is busy
        fi
    done
    return 1  # GPU is free
}

launch_on_gpu() {
    local ctx_type=$1
    local gpu=$2
    local lengths=$3
    local output_dir="${BASE_DIR}/${ctx_type}"
    local log_file="${BASE_DIR}/${ctx_type}.log"

    echo "[$(date)] Launching ${ctx_type} on GPU ${gpu}"

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

    echo "[$(date)] ${ctx_type} PID=$! on GPU ${gpu}"
}

# Build queue of experiments that need running
QUEUE=()
for entry in "${EXPERIMENTS[@]}"; do
    ctx_type="${entry%%:*}"
    lengths="${entry#*:}"
    if is_complete "$ctx_type"; then
        echo "SKIP: ${ctx_type} (already complete)"
    elif is_running "$ctx_type"; then
        echo "RUNNING: ${ctx_type} (already in progress)"
    else
        QUEUE+=("${entry}")
        echo "QUEUED: ${ctx_type}"
    fi
done

echo ""
echo "Queue: ${#QUEUE[@]} experiments to launch"
echo "Monitoring GPUs 0-3 for availability..."
echo ""

QUEUE_IDX=0
NUM_GPUS=4

while [ $QUEUE_IDX -lt ${#QUEUE[@]} ]; do
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        if [ $QUEUE_IDX -ge ${#QUEUE[@]} ]; then
            break
        fi

        if ! gpu_has_experiment "$gpu"; then
            entry="${QUEUE[$QUEUE_IDX]}"
            ctx_type="${entry%%:*}"
            lengths="${entry#*:}"

            # Double-check not already complete (may have finished while waiting)
            if is_complete "$ctx_type"; then
                echo "SKIP: ${ctx_type} (completed while waiting)"
                QUEUE_IDX=$((QUEUE_IDX + 1))
                continue
            fi

            launch_on_gpu "$ctx_type" "$gpu" "$lengths"
            QUEUE_IDX=$((QUEUE_IDX + 1))
            sleep 5  # Brief pause for process to register
        fi
    done

    if [ $QUEUE_IDX -lt ${#QUEUE[@]} ]; then
        sleep 30  # Check every 30s for free GPUs
    fi
done

echo ""
echo "[$(date)] All experiments launched. Waiting for completion..."
wait
echo "[$(date)] All experiments complete!"

# Summary
echo ""
echo "=== Results Summary ==="
for entry in "${EXPERIMENTS[@]}"; do
    ctx_type="${entry%%:*}"
    if [ -f "${BASE_DIR}/${ctx_type}/all_results.json" ]; then
        echo "  OK: ${ctx_type}"
    else
        echo "  FAIL: ${ctx_type}"
    fi
done
