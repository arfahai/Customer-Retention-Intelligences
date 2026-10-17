"""One command that regenerates the whole, coherent artifact set.

    python src/run_pipeline.py

Order: train_churn -> clv -> score_customers -> validate_model -> make_manifest -> verify.
All outputs (artifacts/ and reports/) are written in the same environment/run so they
are always consistent. Commit them together.
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = [
    "train_churn.py",
    "clv.py",
    "score_customers.py",
    "validate_model.py",
    "make_manifest.py",
    "verify_artifacts.py",
]


def main():
    for step in STEPS:
        print(f"\n=== {step} ===", flush=True)
        subprocess.run([sys.executable, str(SRC / step)], check=True, cwd=SRC)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
