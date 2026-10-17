"""Write artifacts/manifest.json: data hash, environment, code hashes, output hashes.

The manifest ties every committed artifact to the exact source data, code and package
versions that produced it. Verify later with: python src/verify_artifacts.py
"""
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "online_retail_II.xlsx"
GENERATION_COMMAND = "python src/run_pipeline.py"

# distribution names as pip reports them
PIN_NAMES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-learn": "scikit-learn",
    "lifetimes": "Lifetimes",
    "openpyxl": "openpyxl",
    "matplotlib": "matplotlib",
    "joblib": "joblib",
    "pyarrow": "pyarrow",
    "streamlit": "streamlit",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def requirements_pins():
    pins = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)==([^\s#]+)", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def installed_packages():
    out = {}
    for key, dist in PIN_NAMES.items():
        try:
            out[key] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[key] = None
    return out


def file_entry(path, count_rows=False):
    entry = {"sha256": sha256(path), "bytes": path.stat().st_size}
    if count_rows and path.suffix == ".csv":
        entry["rows"] = int(len(pd.read_csv(path)))
    return entry


def build_manifest():
    pins = requirements_pins()
    installed = installed_packages()
    matches = all(installed.get(k) == pins.get(k) for k in pins if k in installed)

    outputs = {}
    for folder in (ART, REPORTS):
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.name not in {"manifest.json", "README.md"}:
                outputs[f"{folder.name}/{p.name}"] = file_entry(p, count_rows=True)

    code = {
        f"src/{p.name}": sha256(p)
        for p in sorted((ROOT / "src").glob("*.py"))
    }
    code["requirements.txt"] = sha256(ROOT / "requirements.txt")

    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generation_command": GENERATION_COMMAND,
        "random_state": 42,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": installed,
            "matches_requirements_txt": bool(matches),
        },
        "source_data": {"path": "data/online_retail_II.xlsx", **file_entry(DATA)},
        "code": code,
        "outputs": outputs,
    }


def main():
    manifest = build_manifest()
    (ART / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Wrote artifacts/manifest.json")
    if not manifest["environment"]["matches_requirements_txt"]:
        print("WARNING: installed packages do not match requirements.txt pins")


if __name__ == "__main__":
    main()
