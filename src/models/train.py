"""Stage 7 - Model Building.

Trains and compares five models on identical splits:

    Logistic Regression   interpretable baseline (one-hot + scaling)
    Random Forest         bagged trees, ordinal-encoded categoricals
    XGBoost               native categorical support, early stopping
    LightGBM              native categorical support, early stopping
    CatBoost              ordered target statistics for categoricals

Selection metric is PR-AUC (average precision) on the validation window:
the target is imbalanced (~21% positive) and the operational question is
"which flights should we act on", which is a precision/recall trade-off,
not an accuracy question.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, log_loss, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline

from src.analysis.style import ACCENT, ACCENT_2, GOOD, INK, MUTED, SEQ, apply_style, despine
from src.config import CFG, FIGURES, MODELS_DIR, REPORTS, SEED
from src.models.dataset import (CATEGORICAL, make_splits, sklearn_preprocessor,
                                to_ordinal)

warnings.filterwarnings("ignore")
apply_style()
MP = CFG["models"]


# --------------------------------------------------------------------------
def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "log_loss": log_loss(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "base_rate": float(np.mean(y_true)),
        "lift_top_decile": _lift(y_true, proba, 0.10),
    }


def _lift(y_true, proba, frac: float) -> float:
    n = max(int(len(proba) * frac), 1)
    idx = np.argsort(proba)[::-1][:n]
    return float(np.mean(np.asarray(y_true)[idx]) / np.mean(y_true))


def best_f1_threshold(y_true, proba) -> tuple[float, float]:
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(t[i]), float(f1[i])


# --------------------------------------------------------------------------
def fit_logistic(sp):
    pipe = Pipeline([("prep", sklearn_preprocessor(sp.features)),
                     ("clf", LogisticRegression(max_iter=2000, C=0.5,
                                                class_weight=None,
                                                random_state=SEED))])
    pipe.fit(sp.X_train, sp.y_train)
    return pipe, lambda X: pipe.predict_proba(X)[:, 1]


def fit_random_forest(sp):
    p = MP["random_forest"]
    clf = RandomForestClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        min_samples_leaf=p["min_samples_leaf"], n_jobs=p["n_jobs"],
        random_state=SEED, class_weight=None)
    clf.fit(to_ordinal(sp.X_train).fillna(-1), sp.y_train)
    return clf, lambda X: clf.predict_proba(to_ordinal(X).fillna(-1))[:, 1]


def fit_xgboost(sp):
    from xgboost import XGBClassifier
    p = MP["xgboost"]
    clf = XGBClassifier(
        n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
        max_depth=p["max_depth"], subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"], enable_categorical=True,
        tree_method="hist", eval_metric="aucpr", early_stopping_rounds=50,
        random_state=SEED, n_jobs=-1)
    clf.fit(sp.X_train, sp.y_train, eval_set=[(sp.X_valid, sp.y_valid)], verbose=False)
    return clf, lambda X: clf.predict_proba(X)[:, 1]


def fit_lightgbm(sp):
    from lightgbm import LGBMClassifier, early_stopping
    p = MP["lightgbm"]
    clf = LGBMClassifier(
        n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
        num_leaves=p["num_leaves"], subsample=p["subsample"], subsample_freq=1,
        colsample_bytree=p["colsample_bytree"], random_state=SEED,
        n_jobs=-1, verbose=-1)
    clf.fit(sp.X_train, sp.y_train, eval_set=[(sp.X_valid, sp.y_valid)],
            eval_metric="average_precision",
            callbacks=[early_stopping(50, verbose=False)])
    return clf, lambda X: clf.predict_proba(X)[:, 1]


def fit_catboost(sp):
    from catboost import CatBoostClassifier, Pool
    p = MP["catboost"]
    cats = [c for c in sp.features if c in CATEGORICAL]

    def prep(X):
        X = X.copy()
        for c in cats:
            X[c] = X[c].astype("object").fillna("missing").astype(str)
        return X

    clf = CatBoostClassifier(
        iterations=p["iterations"], learning_rate=p["learning_rate"],
        depth=p["depth"], eval_metric="PRAUC", random_seed=SEED,
        verbose=False, early_stopping_rounds=50, thread_count=-1)
    clf.fit(Pool(prep(sp.X_train), sp.y_train, cat_features=cats),
            eval_set=Pool(prep(sp.X_valid), sp.y_valid, cat_features=cats))
    return clf, lambda X: clf.predict_proba(prep(X))[:, 1]


MODELS = {
    "logistic_regression": fit_logistic,
    "random_forest": fit_random_forest,
    "xgboost": fit_xgboost,
    "lightgbm": fit_lightgbm,
    "catboost": fit_catboost,
}


# --------------------------------------------------------------------------
def plot_comparison(results: dict, y_valid) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for (name, r), c in zip(results.items(), SEQ):
        fpr, tpr, _ = roc_curve(y_valid, r["proba"])
        axes[0].plot(fpr, tpr, color=c, lw=1.8,
                     label=f"{name} ({r['valid']['roc_auc']:.3f})")
        p, rc, _ = precision_recall_curve(y_valid, r["proba"])
        axes[1].plot(rc, p, color=c, lw=1.8,
                     label=f"{name} ({r['valid']['pr_auc']:.3f})")
        pt, pp = calibration_curve(y_valid, r["proba"], n_bins=12, strategy="quantile")
        axes[2].plot(pp, pt, marker="o", ms=3.5, color=c, lw=1.4, label=name)

    axes[0].plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    axes[0].set(title="ROC curve (validation)", xlabel="false positive rate",
                ylabel="true positive rate")
    axes[0].legend(title="ROC-AUC", loc="lower right")
    axes[1].axhline(np.mean(y_valid), ls="--", color=MUTED, lw=1)
    axes[1].set(title="Precision-Recall curve (validation)", xlabel="recall", ylabel="precision")
    axes[1].legend(title="PR-AUC", loc="upper right")
    axes[2].plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    axes[2].set(title="Calibration (validation)", xlabel="predicted probability",
                ylabel="observed frequency")
    axes[2].legend(loc="upper left")
    for a in axes:
        despine(a)
    fig.savefig(FIGURES / "11_model_comparison.png")
    plt.close(fig)


RUNS = REPORTS / "_model_runs"


def run_one(name: str, sp) -> dict:
    """Fit one model, persist it and its metrics. Resumable across processes."""
    RUNS.mkdir(exist_ok=True)
    meta_p = RUNS / f"{name}.json"
    prob_p = RUNS / f"{name}_valid_proba.npy"
    if meta_p.exists() and prob_p.exists():
        print(f"[train] {name}: cached, skipping")
        return json.loads(meta_p.read_text())

    t0 = time.time()
    print(f"[train] fitting {name} ...", flush=True)
    model, predict = MODELS[name](sp)
    took = time.time() - t0
    pv, pt = predict(sp.X_valid), predict(sp.X_test)
    thr, f1 = best_f1_threshold(sp.y_valid, pv)
    res = {
        "valid": evaluate(sp.y_valid, pv),
        "test": evaluate(sp.y_test, pt),
        "valid_best_f1": {"threshold": thr, "f1": f1},
        "fit_seconds": round(took, 1),
    }
    joblib.dump(model, MODELS_DIR / f"{name}.joblib", compress=3)
    np.save(prob_p, pv.astype("float32"))
    meta_p.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    v = res["valid"]
    print(f"           ROC-AUC {v['roc_auc']:.4f} | PR-AUC {v['pr_auc']:.4f} | "
          f"Brier {v['brier']:.4f} | lift@10% {v['lift_top_decile']:.2f}x | {took:.0f}s")
    del model
    gc.collect()
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS),
                    help="subset of models to fit")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)

    sp = make_splits()
    print(f"[train] {sp.summary()}")
    print(f"[train] {len(sp.features)} features")

    results: dict[str, dict] = {}
    if not args.report_only:
        for name in args.models:
            results[name] = run_one(name, sp)

    for name in MODELS:
        p = RUNS / f"{name}.json"
        if name not in results and p.exists():
            results[name] = json.loads(p.read_text())
    for name in results:
        pp = RUNS / f"{name}_valid_proba.npy"
        if pp.exists():
            results[name]["proba"] = np.load(pp)
    if not results:
        print("[train] nothing to report yet")
        return

    table = pd.DataFrame({
        n: {"valid_roc_auc": r["valid"]["roc_auc"],
            "valid_pr_auc": r["valid"]["pr_auc"],
            "valid_brier": r["valid"]["brier"],
            "valid_logloss": r["valid"]["log_loss"],
            "valid_f1@best": r["valid_best_f1"]["f1"],
            "valid_lift@10%": r["valid"]["lift_top_decile"],
            "test_roc_auc": r["test"]["roc_auc"],
            "test_pr_auc": r["test"]["pr_auc"],
            "fit_seconds": r["fit_seconds"]}
        for n, r in results.items()}).T.sort_values("valid_pr_auc", ascending=False)
    table.to_csv(REPORTS / "model_comparison.csv")
    print("\n[train] model comparison (sorted by validation PR-AUC)")
    print(table.round(4).to_string())

    plot_comparison(results, sp.y_valid)

    best = table.index[0]
    payload = {n: {k: v for k, v in r.items() if k != "proba"} for n, r in results.items()}
    payload["_best_model"] = best
    (REPORTS / "model_metrics.json").write_text(json.dumps(payload, indent=2, default=float),
                                                encoding="utf-8")
    (MODELS_DIR / "best_model_name.txt").write_text(best, encoding="utf-8")
    print(f"\n[train] best by validation PR-AUC: {best}")


if __name__ == "__main__":
    main()
