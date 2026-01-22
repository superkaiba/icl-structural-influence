#!/usr/bin/env python3
"""Direct runner for block permutation experiment - bypasses shell issues."""

import subprocess
import sys

cmd = [
    sys.executable,
    "run_block_permutation_experiment.py",
    "--model", "gpt2",
    "--context-lengths", "10,20,50,100",
    "--n-samples", "30",
    "--seed", "42"
]

print("Starting block permutation experiment...")
print(f"Command: {' '.join(cmd)}")
print()

result = subprocess.run(cmd, capture_output=False, text=True)
sys.exit(result.returncode)
