"""
Ali zandieh - Interview task
Inspect a local dataset without uploading it.
Put dataset file(s) in ./data, then run:
    python inspect_dataset.py
It prints a compact schema/sample report and writes outputs/dataset_profile.txt.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SUPPORTED = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".jsonl", ".parquet"}

def preview_file(path):
    print("\n" + "=" * 78)
    print(f"FILE: {path.name} | size: {path.stat().st_size / (1024**2):.2f} MB")
    ext = path.suffix.lower()
    try:
        if ext in {".csv", ".tsv", ".txt"}:
            sep = "\t" if ext == ".tsv" else None
            # Read a tiny preview only, so the whole dataset is not loaded into memory.
            sample = None
            last_error = None
            for enc in ("utf-8-sig", "utf-8", "cp1256"):
                try:
                    sample = pd.read_csv(path, encoding=enc, sep=sep, engine="python",
                                         nrows=5, on_bad_lines="warn")
                    break
                except Exception as e:
                    last_error = e
            if sample is None:
                raise last_error
            print("Columns:")
            for c in sample.columns:
                print(f"  - {c!r} | sample dtype={sample[c].dtype}")
            print("\nFirst 3 rows (long cells shortened):")
            print(sample.head(3).to_string(index=False, max_colwidth=100))
            # Count rows and missing values in chunks, without loading everything.
            try:
                encoding = "utf-8-sig"
                sep2 = sep
                totals = {"rows": 0, "missing": {str(c): 0 for c in sample.columns}}
                for chunk in pd.read_csv(path, encoding=encoding, sep=sep2, engine="python",
                                         chunksize=50000, on_bad_lines="warn"):
                    totals["rows"] += len(chunk)
                    for c in chunk.columns:
                        totals["missing"][str(c)] = totals["missing"].get(str(c), 0) + int(chunk[c].isna().sum())
                print(f"\nFull-file row count: {totals['rows']:,}")
                print("Full-file missing values:")
                for c, n in totals["missing"].items():
                    pct = (100*n/totals["rows"]) if totals["rows"] else 0
                    print(f"  - {c}: {n:,} ({pct:.2f}%)")
            except Exception as e:
                print(f"\nFull scan unavailable (preview still valid): {type(e).__name__}: {e}")
        elif ext in {".xlsx", ".xls"}:
            sample = pd.read_excel(path, nrows=5)
            print("Columns:")
            for c in sample.columns:
                print(f"  - {c!r} | sample dtype={sample[c].dtype}")
            print("\nFirst 3 rows (long cells shortened):")
            print(sample.head(3).to_string(index=False, max_colwidth=100))
            print("\nFor Excel, this helper only previews the first rows; full row/missing counts are not scanned.")
        elif ext in {".json", ".jsonl"}:
            try:
                sample = pd.read_json(path, lines=(ext == ".jsonl"), nrows=5)
            except TypeError:
                sample = pd.read_json(path, lines=(ext == ".jsonl")).head(5)
            print("Columns:")
            for c in sample.columns:
                print(f"  - {c!r} | sample dtype={sample[c].dtype}")
            print("\nFirst 3 rows (long cells shortened):")
            print(sample.head(3).to_string(index=False, max_colwidth=100))
        elif ext == ".parquet":
            sample = pd.read_parquet(path).head(5)
            print("Columns:")
            for c in sample.columns:
                print(f"  - {c!r} | sample dtype={sample[c].dtype}")
            print("\nFirst 3 rows (long cells shortened):")
            print(sample.head(3).to_string(index=False, max_colwidth=100))
    except Exception as e:
        print(f"Could not inspect this file: {type(e).__name__}: {e}")

def main():
    files = sorted(p for p in DATA.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"No supported dataset files found in: {DATA}")
        print("Copy the dataset file(s) into the project's data/ folder, then run again.")
        return
    report_lines = []
    import contextlib, io
    for p in files:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            preview_file(p)
        result = buf.getvalue()
        print(result)
        report_lines.append(result)
    report_path = OUT / "dataset_profile.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n" + "=" * 78)
    print(f"Saved compact report to: {report_path}")

if __name__ == "__main__":
    main()
