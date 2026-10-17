Generated files - do not edit by hand.

Regenerate the whole coherent set (artifacts/ + reports/) with:

    python src/run_pipeline.py

`manifest.json` records the source-data hash, Python/package versions, generation command and
the sha256 of every output. Check it with `python src/verify_artifacts.py` (add `--full` to also
re-score the full population from the saved model). Commit artifacts/ and reports/ together.
