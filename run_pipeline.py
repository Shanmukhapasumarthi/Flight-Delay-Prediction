"""End-to-end pipeline runner.

    python run_pipeline.py                 # every stage, in order
    python run_pipeline.py --from features # resume partway through
    python run_pipeline.py --only eda      # a single stage
    python run_pipeline.py --list          # show the stages

Each stage is a module with a main(); stages are idempotent and write their
outputs to disk, so a failure part-way through can be resumed rather than
restarted.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time

STAGES: list[tuple[str, str, str]] = [
    ("collect",  "src.data.generate_synthetic", "Data collection (6 raw sources)"),
    ("clean",    "src.data.clean",              "Cleaning, validation, tz alignment"),
    ("ingest",   "src.data.ingest",             "Integration of all sources"),
    ("features", "src.features.build_features", "Feature engineering (6 families)"),
    ("eda",      "src.analysis.eda",            "Exploratory analysis + figures"),
    ("select",   "src.features.selection",      "Feature selection (MI/gain/RFE)"),
    ("train",    "src.models.train",            "Model comparison (5 candidates)"),
    ("tune",     "src.models.tune",             "Optuna + calibration + final model"),
    ("store",    "src.models.feature_store",    "Online feature store for serving"),
    ("explain",  "src.models.explain",          "SHAP global and local explanations"),
    ("insights", "src.insights.business",       "Business insights and economics"),
]


def run(stage: str, module: str, desc: str) -> float:
    print("\n" + "=" * 78)
    print(f"  {stage.upper():<10} {desc}")
    print("=" * 78, flush=True)
    t0 = time.time()
    importlib.import_module(module).main()
    took = time.time() - t0
    print(f"  -- {stage} finished in {took:.1f}s")
    return took


def main(argv=None) -> int:
    names = [s[0] for s in STAGES]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", choices=names, help="resume from this stage")
    ap.add_argument("--to", dest="end", choices=names, help="stop after this stage")
    ap.add_argument("--only", choices=names, help="run exactly one stage")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    args = ap.parse_args(argv)

    if args.list:
        for n, m, d in STAGES:
            print(f"  {n:<10} {d:<45} ({m})")
        return 0

    selected = STAGES
    if args.only:
        selected = [s for s in STAGES if s[0] == args.only]
    else:
        if args.start:
            selected = selected[names.index(args.start):]
        if args.end:
            keep = [s[0] for s in selected]
            selected = selected[:keep.index(args.end) + 1]

    total = 0.0
    for stage, module, desc in selected:
        try:
            total += run(stage, module, desc)
        except Exception as exc:
            print(f"\n!! stage '{stage}' failed: {type(exc).__name__}: {exc}")
            print(f"   resume with:  python run_pipeline.py --from {stage}")
            return 1

    print("\n" + "=" * 78)
    print(f"  pipeline complete in {total/60:.1f} min")
    print("=" * 78)
    print("  next:  uvicorn api.main:app --port 8000")
    print("         streamlit run dashboard/app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
