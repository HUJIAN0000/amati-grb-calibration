"""Fit the three slope models on the frozen MILP sample."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from pathlib import Path
from amati_pipeline.data import Dataset
from amati_pipeline.inference import run
from amati_pipeline.model import MODES, RelativeCalibration

ROOT = Path(__file__).resolve().parent
SAMPLE = "reference"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", choices=(SAMPLE,))
    parser.add_argument("model", choices=MODES)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    data = Dataset.load(ROOT / "inputs" / args.sample)
    model = RelativeCalibration(data, args.model)
    seed = 2026111400 + MODES.index(args.model)
    run(model, ROOT / "runs" / args.sample / args.model,
        dict(walkers=72, steps=12000, burn=3000, max_steps=50000, seed=seed,
             chunk=2000, auto_extend=True), resume=args.resume)

