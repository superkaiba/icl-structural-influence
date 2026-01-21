#!/bin/bash
cd /workspace/research/projects/in_context_representation_influence
python run_block_permutation_experiment.py --model gpt2 --context-lengths 10,20,50,100 --n-samples 30 --seed 42
