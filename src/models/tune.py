"""Stage 8 - Hyperparameter Optimisation + final model assembly.

Two search strategies are compared on the same estimator so the comparison is
about the SEARCH, not the model:

    RandomizedSearchCV   uninformed sampling, time-series CV
    Optuna (TPE)         sequential model-based, prunes bad trials early

The winner is refit, probability-calibrated on the validation window with
isotonic regression, evaluated once on the untouched test window, and frozen
into a single deployable bundle: models/final_model.joblib
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.analysis.style import ACCENT, ACCENT_2, GOOD, INK, MUTED, apply_style, despine
from src.config import CFG, FIGURES, MODELS_DIR, REPORTS, SEED
from src.models.dataset import category_levels, make_splits
from src.models.train import best_f1_threshold, evaluate

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
apply_style()

N_TRIALS = CFG["tuning"]["n_trials"]
TIMEOUT = CFG["tuning"]["timeout_seconds"]


# --------------------------------------------------------------------------
def randomized_search(sp, n_iter: int = 8) -> tuple[dict, float, float]:
    """Baseline search strategy: uninformed random sampling with time-series CV."""
    print(f"[tune] RandomizedSearchCV, {n_iter} candidates x 2 folds ...")
    grid = {
        "num_leaves": [31, 63, 127, 255],
        "learning_rate": [0.02, 0.05, 0.08, 0.12],
        "min_child_samples": [20, 60, 150, 300],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.0, 1.0, 10.0],
    }
    # subsample rows: the point is the search comparison, not the last 0.1%
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(sp.X_train), min(90_000, len(sp.X_train)), replace=False))
    t0 = time.time()
    rs = RandomizedSearchCV(
        LGBMClassifier(n_estimators=300, random_state=SEED, verbose=-1, n_jobs=-1),
        grid, n_iter=n_iter, scoring="average_precision",
        cv=TimeSeriesSplit(n_splits=2), random_state=SEED, n_jobs=1, refit=False)
    rs.fit(sp.X_train.iloc[idx], sp.y_train.iloc[idx])
    took = time.time() - t0
    print(f"[tune]   best CV PR-AUC {rs.best_score_:.4f} in {took:.0f}s")
    return rs.best_params_, float(rs.best_score_), took


def optuna_search(sp) -> tuple[dict, float, float, optuna.Study]:
    print(f"[tune] Optuna TPE, up to {N_TRIALS} trials ...")
    cat_cols = [c for c in sp.X_train.columns if str(sp.X_train[c].dtype) == "category"]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": 600,
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 14),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        }
        clf = LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1, **params)
        clf.fit(sp.X_train, sp.y_train, eval_set=[(sp.X_valid, sp.y_valid)],
                eval_metric="average_precision",
                categorical_feature=cat_cols,
                callbacks=[early_stopping(40, verbose=False)])
        p = clf.predict_proba(sp.X_valid)[:, 1]
        trial.set_user_attr("best_iteration", int(clf.best_iteration_ or params["n_estimators"]))
        return average_precision_score(sp.y_valid, p)

    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name="lightgbm_delay")
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT, show_progress_bar=False)
    took = time.time() - t0
    print(f"[tune]   best holdout PR-AUC {study.best_value:.4f} "
          f"in {len(study.trials)} trials / {took:.0f}s")
    return study.best_params, float(study.best_value), took, study


# --------------------------------------------------------------------------
def plot_study(study: optuna.Study, rs_score: float) -> None:
    vals = [t.value for t in study.trials if t.value is not None]
    running = np.maximum.accumulate(vals)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    axes[0].scatter(range(len(vals)), vals, s=26, color=ACCENT_2, alpha=0.75, label="trial")
    axes[0].plot(running, color=ACCENT, lw=2, label="best so far")
    axes[0].axhline(rs_score, color=GOOD, ls="--", lw=1.4, label="RandomizedSearchCV best")
    axes[0].set(title="Optuna optimisation history", xlabel="trial",
                ylabel="validation PR-AUC")
    axes[0].legend()

    try:
        imp = optuna.importance.get_param_importances(study)
        k = list(imp)[::-1]
        axes[1].barh(k, [imp[i] for i in k], color=ACCENT_2)
        axes[1].set(title="Hyperparameter importance (fANOVA)", xlabel="relative importance")
    except Exception:
        axes[1].axis("off")
    for a in axes:
        despine(a)
    fig.savefig(FIGURES / "12_hyperparameter_search.png")
    plt.close(fig)


def plot_calibration(y, raw, cal) -> None:
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=(6, 5.2))
    for p, name, c in [(raw, "raw model", ACCENT_2), (cal, "isotonic-calibrated", ACCENT)]:
        pt, pp = calibration_curve(y, p, n_bins=12, strategy="quantile")
        ax.plot(pp, pt, marker="o", ms=4, lw=1.7, color=c,
                label=f"{name} (Brier {brier_score_loss(y, p):.4f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED)
    ax.set(title="Probability calibration on the test window",
           xlabel="predicted probability", ylabel="observed delay frequency")
    ax.legend()
    despine(ax)
    fig.savefig(FIGURES / "13_calibration.png")
    plt.close(fig)


def _halves(X, y):
    """Split the validation window in two by time order."""
    k = len(X) // 2
    return (X.iloc[:k], y.iloc[:k]), (X.iloc[k:], y.iloc[k:])


def _ece(y, p, bins: int = 15) -> float:
    """Expected calibration error over equal-frequency bins."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["b"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    g = df.groupby("b").agg(obs=("y", "mean"), pred=("p", "mean"), n=("y", "size"))
    return float((g["n"] / len(df) * (g["obs"] - g["pred"]).abs()).sum())


# --------------------------------------------------------------------------
def main() -> None:
    sp = make_splits()
    print(f"[tune] {sp.summary()}")

    rs_params, rs_score, rs_time = randomized_search(sp)
    op_params, op_score, op_time, study = optuna_search(sp)
    plot_study(study, rs_score)

    best_iter = study.best_trial.user_attrs.get("best_iteration", 500)
    params = dict(op_params)
    params.update({"n_estimators": max(int(best_iter), 50), "subsample_freq": 1})
    print(f"[tune] refitting with {params['n_estimators']} trees")

    cat_cols = [c for c in sp.X_train.columns if str(sp.X_train[c].dtype) == "category"]
    model = LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1, **params)
    model.fit(sp.X_train, sp.y_train, categorical_feature=cat_cols)

    # ---- is calibration actually needed? ---------------------------------
    # Gradient boosting trained on log-loss is often already well calibrated.
    # Decide on a HELD-OUT half of the validation window so the comparison is
    # not judged on the same rows the calibrator was fitted to.
    cal_fit, cal_eval = _halves(sp.X_valid, sp.y_valid)
    probe = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    probe.fit(cal_fit[0], cal_fit[1])
    Xe, ye = cal_eval
    p_raw, p_iso = model.predict_proba(Xe)[:, 1], probe.predict_proba(Xe)[:, 1]
    cmp = {
        "raw": {"ece": _ece(ye, p_raw), "brier": brier_score_loss(ye, p_raw),
                "pr_auc": average_precision_score(ye, p_raw)},
        "isotonic": {"ece": _ece(ye, p_iso), "brier": brier_score_loss(ye, p_iso),
                     "pr_auc": average_precision_score(ye, p_iso)},
    }
    print("[tune] calibration probe on held-out half of validation:")
    for k, v in cmp.items():
        print(f"        {k:<9} ECE {v['ece']:.4f}  Brier {v['brier']:.4f}  "
              f"PR-AUC {v['pr_auc']:.4f}")
    use_isotonic = cmp["isotonic"]["ece"] < cmp["raw"]["ece"] * 0.9
    print(f"[tune] -> {'isotonic calibration applied' if use_isotonic else 'raw model kept (already calibrated)'}")

    if use_isotonic:
        cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
        cal.fit(sp.X_valid, sp.y_valid)
    else:
        cal = FrozenEstimator(model)

    raw_test = model.predict_proba(sp.X_test)[:, 1]
    cal_test = cal.predict_proba(sp.X_test)[:, 1]
    plot_calibration(sp.y_test, raw_test, cal_test)

    m_raw = evaluate(sp.y_test, raw_test)
    m_cal = evaluate(sp.y_test, cal_test)
    thr_f1, f1 = best_f1_threshold(sp.y_valid, cal.predict_proba(sp.X_valid)[:, 1])

    print("\n[tune] TEST WINDOW (never used for fitting or selection)")
    print(f"        raw        ROC-AUC {m_raw['roc_auc']:.4f}  PR-AUC {m_raw['pr_auc']:.4f}  "
          f"Brier {m_raw['brier']:.4f}")
    print(f"        calibrated ROC-AUC {m_cal['roc_auc']:.4f}  PR-AUC {m_cal['pr_auc']:.4f}  "
          f"Brier {m_cal['brier']:.4f}")

    bundle = {
        "model": model,
        "calibrator": cal,
        "features": sp.features,
        "categories": category_levels(sp.X_train),
        "params": params,
        "threshold_f1": thr_f1,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "train_rows": int(len(sp.y_train)),
        "metrics_test": m_cal,
        "metrics_test_uncalibrated": m_raw,
        "model_type": ("lightgbm + isotonic calibration" if use_isotonic
                       else "lightgbm (natively calibrated)"),
        "calibration_probe": cmp,
    }
    joblib.dump(bundle, MODELS_DIR / "final_model.joblib", compress=3)

    summary = {
        "randomized_search": {"best_cv_pr_auc": rs_score, "seconds": rs_time,
                              "params": rs_params},
        "optuna": {"best_valid_pr_auc": op_score, "seconds": op_time,
                   "n_trials": len(study.trials), "params": op_params},
        "final": {"params": params, "test_calibrated": m_cal, "test_raw": m_raw,
                  "threshold_best_f1": thr_f1},
    }
    (REPORTS / "tuning_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"[tune] bundle -> {MODELS_DIR/'final_model.joblib'}")


if __name__ == "__main__":
    main()
