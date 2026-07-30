"""Stage 6 - Feature Selection.

Four independent signals are combined into a consensus ranking:

  1. Correlation pruning   drop one of every pair with |r| > 0.95
  2. Mutual information    captures non-linear univariate dependence
  3. LightGBM gain         multivariate, interaction-aware
  4. RFE                   backward elimination with a gradient-boosted base

Selection is fitted on the TRAINING WINDOW ONLY. Using the full data set here
would leak validation/test information into the feature set itself.
"""
from __future__ import annotations

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE, mutual_info_classif

from src.analysis.style import ACCENT, ACCENT_2, INK, MUTED, apply_style, despine
from src.config import CFG, FIGURES, MODELS_DIR, REPORTS, SEED
from src.features.build_features import FEATURE_GROUPS
from src.models.dataset import ALL_FEATURES, make_splits, to_ordinal

warnings.filterwarnings("ignore")
apply_style()

MAX_CORR = CFG["features"]["max_correlation"]
TOP_K = CFG["features"]["top_k"]
GROUP_OF = {f: g for g, cols in FEATURE_GROUPS.items() for f in cols}


def correlation_prune(X: pd.DataFrame, keep_score: pd.Series) -> tuple[list[str], list[tuple]]:
    num = X.select_dtypes("number")
    corr = num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped, pairs = set(), []
    for a in upper.columns:
        for b in upper.index[upper[a] > MAX_CORR]:
            if a in dropped or b in dropped:
                continue
            loser = a if keep_score.get(a, 0) < keep_score.get(b, 0) else b
            winner = b if loser == a else a
            dropped.add(loser)
            pairs.append((winner, loser, float(upper.loc[b, a])))
    return [c for c in X.columns if c not in dropped], pairs


def main() -> None:
    print("[select] building training matrix")
    sp = make_splits(features=ALL_FEATURES)
    X, y = sp.X_train, sp.y_train
    print(f"[select] {sp.summary()}")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(60_000, len(X)), replace=False)
    Xs, ys = to_ordinal(X.iloc[idx]).fillna(-1), y.iloc[idx]

    # ---- 1. mutual information -------------------------------------------
    print("[select] mutual information ...")
    mi = mutual_info_classif(Xs, ys, discrete_features=False, random_state=SEED, n_neighbors=3)
    mi = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    # ---- 2. LightGBM gain ------------------------------------------------
    print("[select] LightGBM gain importance ...")
    lgb = LGBMClassifier(n_estimators=300, learning_rate=0.06, num_leaves=63,
                         random_state=SEED, verbose=-1, n_jobs=-1)
    lgb.fit(X, y, categorical_feature=[c for c in X.columns if str(X[c].dtype) == "category"])
    gain = pd.Series(lgb.booster_.feature_importance("gain"), index=X.columns)
    gain = (gain / gain.sum()).sort_values(ascending=False)

    # ---- 3. correlation pruning -----------------------------------------
    kept, pairs = correlation_prune(X, gain)
    print(f"[select] correlation pruning removed {len(pairs)} redundant features")
    for w, l, r in pairs[:8]:
        print(f"           |r|={r:.3f}  keep {w:<28} drop {l}")

    # ---- 4. RFE ----------------------------------------------------------
    print(f"[select] RFE down to {TOP_K} features ...")
    base = LGBMClassifier(n_estimators=120, learning_rate=0.1, num_leaves=31,
                          random_state=SEED, verbose=-1, n_jobs=-1)
    rfe = RFE(base, n_features_to_select=TOP_K, step=0.12)
    rfe.fit(to_ordinal(X[kept]).fillna(-1), y)
    rfe_rank = pd.Series(rfe.ranking_, index=kept)

    # ---- consensus -------------------------------------------------------
    scores = pd.DataFrame({
        "mutual_information": mi,
        "lgbm_gain": gain,
        "rfe_rank": rfe_rank,
    })
    scores["group"] = [GROUP_OF.get(f, "other") for f in scores.index]
    scores["pruned_by_correlation"] = ~scores.index.isin(kept)
    scores["rank_mi"] = scores["mutual_information"].rank(ascending=False)
    scores["rank_gain"] = scores["lgbm_gain"].rank(ascending=False)
    scores["rank_rfe"] = scores["rfe_rank"].rank(ascending=True)
    scores["consensus_rank"] = scores[["rank_mi", "rank_gain", "rank_rfe"]].mean(axis=1)
    scores.loc[scores["pruned_by_correlation"], "consensus_rank"] += 1000
    scores = scores.sort_values("consensus_rank")

    selected = scores.head(TOP_K).index.tolist()
    scores["selected"] = scores.index.isin(selected)
    scores.to_csv(REPORTS / "feature_selection.csv")

    (MODELS_DIR / "selected_features.json").write_text(json.dumps({
        "selected": selected,
        "n_candidates": len(ALL_FEATURES),
        "n_selected": len(selected),
        "method": "consensus of mutual information, LightGBM gain and RFE, "
                  "after correlation pruning at |r|>%.2f" % MAX_CORR,
        "group_mix": pd.Series([GROUP_OF.get(f, "other") for f in selected])
                       .value_counts().to_dict(),
    }, indent=2), encoding="utf-8")

    # ---- figure ----------------------------------------------------------
    top = scores.head(28).iloc[::-1]
    palette = {"time": "#2f6f8f", "weather": "#4d8ba6", "airport": "#e3b23c",
               "flight": "#d98d3a", "geospatial": "#7aa8ba", "historical": ACCENT}
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.5))
    axes[0].barh(top.index, top["lgbm_gain"],
                 color=[palette.get(g, MUTED) for g in top["group"]])
    axes[0].set(title=f"Top {len(top)} features by LightGBM gain", xlabel="share of total gain")
    axes[1].barh(top.index, top["mutual_information"],
                 color=[palette.get(g, MUTED) for g in top["group"]])
    axes[1].set(title="Same features, mutual information", xlabel="MI (nats)")
    axes[1].set_yticklabels([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()]
    axes[0].legend(handles, palette.keys(), title="feature family", loc="lower right")
    for a in axes:
        despine(a)
    fig.suptitle("Feature selection: engineered features dominate raw columns",
                 y=0.995, fontsize=11, color=MUTED)
    fig.savefig(FIGURES / "10_feature_selection.png")
    plt.close(fig)

    mix = pd.Series([GROUP_OF.get(f, "other") for f in selected]).value_counts()
    print(f"[select] kept {len(selected)}/{len(ALL_FEATURES)} features")
    print("[select] family mix:", dict(mix))
    print("[select] top 12:", selected[:12])


if __name__ == "__main__":
    main()
