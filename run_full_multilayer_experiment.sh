#!/bin/bash
# Run full multi-layer LOO experiment with 32 layers and 14 context lengths
# Estimated runtime: 7-8 hours

OUTPUT_DIR="results/loo_multilayer"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Starting Full Multi-Layer LOO Experiment"
echo "============================================================"
echo "Configuration:"
echo "  - Layers: All 32 (0-31)"
echo "  - Context lengths: [6,7,8,9,10,12,15,20,25,30,40,50,75,100]"
echo "  - Trials per N: 5"
echo "  - Conditions: semantic + unrelated"
echo "  - Estimated time: 7-8 hours"
echo ""
echo "Checkpoints will be saved after each N to:"
echo "  $OUTPUT_DIR/checkpoint_*.json"
echo ""
echo "Started at: $(date)"
echo "============================================================"
echo ""

# Run the experiment
python run_multilayer_loo_experiment.py

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "EXPERIMENT COMPLETED SUCCESSFULLY"
else
    echo "EXPERIMENT FAILED (exit code: $EXIT_CODE)"
    echo "Check checkpoint files for partial results"
fi
echo "Finished at: $(date)"
echo "============================================================"

exit $EXIT_CODE
