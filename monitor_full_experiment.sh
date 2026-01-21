#!/bin/bash
# Monitor the full multi-layer LOO experiment progress

OUTPUT_FILE="/tmp/claude/-workspace-research-projects-in-context-representation-influence/tasks/b0d42a3.output"
CHECKPOINT_DIR="results/loo_multilayer"

echo "=== Full Multi-Layer Experiment Monitor ==="
echo "Time: $(date)"
echo ""

# Check if experiment is still running
if ps aux | grep -v grep | grep "run_multilayer_loo_experiment.py" > /dev/null; then
    STATUS="RUNNING"
else
    if grep -q "EXPERIMENT COMPLETED" "$OUTPUT_FILE" 2>/dev/null; then
        STATUS="COMPLETED"
    elif grep -q "EXPERIMENT FAILED" "$OUTPUT_FILE" 2>/dev/null; then
        STATUS="FAILED"
    else
        STATUS="UNKNOWN"
    fi
fi

echo "Status: $STATUS"
echo ""

# Count checkpoints completed
if [ -d "$CHECKPOINT_DIR" ]; then
    SEMANTIC_CHECKPOINTS=$(ls -1 "$CHECKPOINT_DIR"/checkpoint_semantic_*.json 2>/dev/null | wc -l)
    UNRELATED_CHECKPOINTS=$(ls -1 "$CHECKPOINT_DIR"/checkpoint_unrelated_*.json 2>/dev/null | wc -l)
    echo "Checkpoints completed:"
    echo "  - Semantic condition: $SEMANTIC_CHECKPOINTS / 14 context lengths"
    echo "  - Unrelated condition: $UNRELATED_CHECKPOINTS / 14 context lengths"
    echo ""
fi

# Show recent progress
if [ -f "$OUTPUT_FILE" ]; then
    echo "Recent output:"
    tail -20 "$OUTPUT_FILE" | grep -E "Context length|N=|%|Loading|Saving|SEMANTIC|UNRELATED|COMPLETE|FAILED" | tail -10
    echo ""

    echo "Last line:"
    tail -1 "$OUTPUT_FILE"
fi

# Check for OOM
if [ -f "$OUTPUT_FILE" ] && grep -q "Killed" "$OUTPUT_FILE"; then
    echo ""
    echo "⚠️  WARNING: Process was killed (likely OOM)"
fi

# Estimate progress
if [ -f "$OUTPUT_FILE" ]; then
    TOTAL_CHECKPOINTS=28  # 14 context lengths × 2 conditions
    COMPLETED_CHECKPOINTS=$((SEMANTIC_CHECKPOINTS + UNRELATED_CHECKPOINTS))
    if [ $COMPLETED_CHECKPOINTS -gt 0 ]; then
        PROGRESS=$((COMPLETED_CHECKPOINTS * 100 / TOTAL_CHECKPOINTS))
        echo ""
        echo "Overall progress: $COMPLETED_CHECKPOINTS / $TOTAL_CHECKPOINTS checkpoints ($PROGRESS%)"
    fi
fi
