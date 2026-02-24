# ============================================================
# ΔGI Driver Discovery Pipeline (Explainability-oriented)
# + Reviewer-friendly figures + Robust metrics (MAE/RMSE/CI/CV)
# ============================================================

import os
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr, pearsonr

from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV, enet_path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import statsmodels.api as sm

# -------------------------
# Journal-style plotting
# -------------------------


# ---- JACMP-friendly global style ----
sns.set_theme(style="whitegrid", context="paper")

plt.rcParams.update({
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "pdf.fonttype": 42,   # embed TrueType fonts (better for AI/PR)
    "ps.fonttype": 42,

    # Text sizes (paper-ready)
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,

    # Line/axis widths (avoid hairlines after downscaling)
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.6,
    "lines.markersize": 5.5,

    # Tick styling
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "xtick.minor.width": 0.7,
    "ytick.minor.width": 0.7,

    # Grid (subtle, journal-like)
    "grid.linewidth": 0.6,
    "grid.alpha": 0.35,

    # Figure export defaults
    "figure.dpi": 300,        # screen
    "savefig.dpi": 600,       # line art (tiff)
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    # Misc
    "axes.unicode_minus": False,
})

# -------------------------
# Config (edit if needed)
# -------------------------
DATA_PATH = r"C:\zq\pytorchlearn\Radiotherapy files\最终数据_with_featuresimputed.csv"
GI_COL = "GI"
PTV_COL = "PTV_volume"   # 你已确认存在
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Missingness filter (train-only)
MISSING_THRESHOLD_PERCENT = 10.0

# ElasticNetCV hyperparam grid
L1_RATIOS = [0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
ALPHAS = np.logspace(-4, -0.5, 20)

# Stability selection
N_BOOT = 200
SUBSAMPLE_FRAC = 0.8
FREQ_THRESHOLD = 0.60

# Correlation pruning
CORR_THRESHOLD = 0.85
FINAL_TOPK = 15

# Visualization helpers
TOP_HEATMAP = 10
TOP4_SHOW = 4
WINSOR_P = (0.01, 0.99)
BIN_Q = 10

# CV for CI (outer repeated CV on training set)
DO_REPEATED_CV = True
KFOLDS = 5
REPEATS = 10         # 觉得慢就改成 5；想更稳就 20
CV_RANDOM_STATE = 42

# Bootstrap CI for test metrics
BOOT_N =5000        # 觉得慢可 1000；想更稳 5000
BOOT_RANDOM_STATE = 42

# Output dir (DO NOT CHANGE)
OUT_DIR = Path(r"C:\zq\pytorchlearn\Radiotherapy files\gpr_driver_outputsnew1")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================
# Utilities
# ============================================================

def safe_read_csv(path: str):
    for enc in ["utf-8-sig", "utf-8", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError("CSV read failed with common encodings.")

def save_fig(filename: str, dpi: int = 600, close: bool = False):
    out = OUT_DIR / filename
    plt.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close()
    print(f"[Saved Figure] {out}")

def summary_stats(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, median=np.nan, q1=np.nan, q3=np.nan, min=np.nan, max=np.nan)
    q1, q3 = np.quantile(x, [0.25, 0.75])
    return dict(n=int(x.size), median=float(np.median(x)), q1=float(q1), q3=float(q3),
                min=float(np.min(x)), max=float(np.max(x)))

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return dict(R2=float(r2), MAE=float(mae), RMSE=float(rmse))

def bootstrap_ci_metrics(y_true, y_pred, n_boot=2000, seed=2026):
    """
    Non-parametric bootstrap CI on (R2, MAE, RMSE).
    Resample pairs (y_true, y_pred) with replacement.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    stats = np.zeros((n_boot, 3), dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = compute_metrics(y_true[idx], y_pred[idx])
        stats[b, 0] = m["R2"]
        stats[b, 1] = m["MAE"]
        stats[b, 2] = m["RMSE"]

    ci = {}
    for j, name in enumerate(["R2", "MAE", "RMSE"]):
        lo, hi = np.quantile(stats[:, j], [0.025, 0.975])
        ci[name] = (float(lo), float(hi))
    return ci

def winsorize_series(s: pd.Series, p_low=0.01, p_high=0.99):
    s = s.astype(float)
    lo, hi = np.nanquantile(s.values, [p_low, p_high])
    return s.clip(lo, hi)

def lowess_xy(x, y, frac=0.35):
    z = sm.nonparametric.lowess(endog=y, exog=x, frac=frac, it=1, return_sorted=True)
    return z[:, 0], z[:, 1]

def binned_mean_ci(x, y, q=10):
    """
    Quantile bins. Return centers, mean, low, up, nbin.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 30:
        return None
    try:
        bins = pd.qcut(x, q=q, duplicates="drop")
    except Exception:
        return None

    dfb = pd.DataFrame({"x": x, "y": y, "bin": bins})
    g = dfb.groupby("bin", observed=True)
    centers = g["x"].median().values
    mean = g["y"].mean().values
    std = g["y"].std(ddof=1).values
    nbin = g.size().values.astype(float)
    se = std / np.sqrt(np.maximum(nbin, 1))
    low = mean - 1.96 * se
    up = mean + 1.96 * se
    return centers, mean, low, up, nbin

def drop_high_missing_train_only(X_train: pd.DataFrame, X_test: pd.DataFrame, thr_pct: float):
    miss_pct = X_train.isnull().mean() * 100
    drop_cols = miss_pct[miss_pct > thr_pct].index.tolist()
    X_train2 = X_train.drop(columns=drop_cols)
    X_test2 = X_test.drop(columns=drop_cols, errors="ignore")
    return X_train2, X_test2, drop_cols, miss_pct.sort_values(ascending=False)

def correlation_prune_spearman(df_features: pd.DataFrame, scores: pd.Series, corr_thr=0.85):
    """
    Greedy prune: keep higher-score feature, drop others with |rho|>=thr to it.
    """
    corr = df_features.corr(method="spearman").abs()
    kept = []
    dropped = set()

    order = scores.sort_values(ascending=False).index.tolist()
    for f in order:
        if f in dropped:
            continue
        kept.append(f)
        highly = corr.index[(corr[f] >= corr_thr)].tolist()
        for g in highly:
            if g != f:
                dropped.add(g)
    return kept, corr

def stability_selection_elasticnet(X, y, feature_names, alpha, l1_ratio, n_boot=200, subsample_frac=0.8, random_state=42):
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    counts = np.zeros(len(feature_names), dtype=int)
    for _ in range(n_boot):
        idx = rng.choice(n, size=int(n * subsample_frac), replace=False)
        m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=50000, random_state=random_state)
        m.fit(X[idx], y[idx])
        counts += (m.coef_ != 0).astype(int)
    freq = counts / n_boot
    out = pd.DataFrame({"feature": feature_names, "select_freq": freq}).sort_values("select_freq", ascending=False)
    return out

def build_feature_matrix(df: pd.DataFrame, y_col: str):
    """
    Build numeric predictor matrix:
    - drop y_col, keep numeric columns only
    - also drop GI itself (if exists) to avoid leakage when outcome is ΔGI
    """
    drop_cols = {y_col}
    # if GI exists, it is part of outcome construction; exclude it from predictors
    if GI_COL in df.columns:
        drop_cols.add(GI_COL)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()
    feature_names = X.columns.tolist()
    return X, feature_names

# ============================================================
# Step 0) Load data, construct ΔGI, split indices (leakage-free)
# ============================================================

df0 = safe_read_csv(DATA_PATH)
assert GI_COL in df0.columns, f"Missing {GI_COL} in data."
assert PTV_COL in df0.columns, f"Missing {PTV_COL} in data."

df0 = df0.loc[df0[GI_COL].notna()].copy()

# Train/test split (fixed)
idx_all = np.arange(len(df0))
idx_train, idx_test = train_test_split(
    idx_all, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

gi_all = df0.loc[:, GI_COL].astype(float).values
gi_tr = df0.loc[idx_train, GI_COL].astype(float).values
gi_te = df0.loc[idx_test, GI_COL].astype(float).values

gi_train_median = float(np.median(gi_tr))
DELTA_GI_COL = "dGI"
df0[DELTA_GI_COL] = df0[GI_COL].astype(float) - gi_train_median

dgi_all = df0[DELTA_GI_COL].astype(float).values
dgi_tr = df0.loc[idx_train, DELTA_GI_COL].astype(float).values
dgi_te = df0.loc[idx_test, DELTA_GI_COL].astype(float).values

# ============================================================
# Fig1: GI and ΔGI distribution (Train vs Test overlay)
# ============================================================

st_gi_all = summary_stats(gi_all)
st_gi_tr  = summary_stats(gi_tr)
st_gi_te  = summary_stats(gi_te)

st_dgi_all = summary_stats(dgi_all)
st_dgi_tr  = summary_stats(dgi_tr)
st_dgi_te  = summary_stats(dgi_te)

print("\n[Fig1 Summary] GI distribution")
print(f"  Overall:  median={st_gi_all['median']:.4f} (IQR {st_gi_all['q1']:.4f}-{st_gi_all['q3']:.4f}), "
      f"range {st_gi_all['min']:.4f}-{st_gi_all['max']:.4f}, n={st_gi_all['n']}")
print(f"  Train:    median={st_gi_tr['median']:.4f} (IQR {st_gi_tr['q1']:.4f}-{st_gi_tr['q3']:.4f}), "
      f"range {st_gi_tr['min']:.4f}-{st_gi_tr['max']:.4f}, n={st_gi_tr['n']}")
print(f"  Test:     median={st_gi_te['median']:.4f} (IQR {st_gi_te['q1']:.4f}-{st_gi_te['q3']:.4f}), "
      f"range {st_gi_te['min']:.4f}-{st_gi_te['max']:.4f}, n={st_gi_te['n']}")

print("\n[Fig1 Summary] ΔGI distribution")
print(f"  Overall:  median={st_dgi_all['median']:.4f} (IQR {st_dgi_all['q1']:.4f}-{st_dgi_all['q3']:.4f}), "
      f"range {st_dgi_all['min']:.4f}-{st_dgi_all['max']:.4f}, n={st_dgi_all['n']}")
print(f"  Train:    median={st_dgi_tr['median']:.4f} (IQR {st_dgi_tr['q1']:.4f}-{st_dgi_tr['q3']:.4f}), "
      f"range {st_dgi_tr['min']:.4f}-{st_dgi_tr['max']:.4f}, n={st_dgi_tr['n']}")
print(f"  Test:     median={st_dgi_te['median']:.4f} (IQR {st_dgi_te['q1']:.4f}-{st_dgi_te['q3']:.4f}), "
      f"range {st_dgi_te['min']:.4f}-{st_dgi_te['max']:.4f}, n={st_dgi_te['n']}")

# 建议：双栏宽 180 mm
W_MM = 180
W_IN = W_MM / 25.4
H_IN = W_IN * 0.50  # 1x2 panels

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W_IN, H_IN), constrained_layout=True)

# ---------------- Panel A: GI ----------------
sns.histplot(gi_tr, bins=30, stat="density", kde=True, element="step",
             fill=False, linewidth=1.0, label="Train", ax=ax1)
sns.histplot(gi_te, bins=30, stat="density", kde=True, element="step",
             fill=False, linewidth=1.0, label="Test", ax=ax1)
ax1.axvline(gi_train_median, linestyle="--", linewidth=1.2, color="k")

# ax1.set_title("GI distribution (Train vs Test)")
ax1.set_xlabel("GI")
ax1.set_ylabel("Density")

# 统计框：放左上角，稍微往下挪一点避免压标题
txt1 = "\n".join([
    f"Overall: M={st_gi_all['median']:.3f}",
    f"IQR {st_gi_all['q1']:.3f}-{st_gi_all['q3']:.3f}",
    f"Range {st_gi_all['min']:.3f}-{st_gi_all['max']:.3f}",
    f"n={st_gi_all['n']}"
])
ax1.text(0.56, 0.95, txt1, transform=ax1.transAxes, va="top", ha="left",
         fontsize=9, bbox=dict(boxstyle="round", facecolor="white",
                               alpha=0.85, linewidth=0.8))

# 用“文字标注”代替把 median 放进 legend（更省空间）
ax1.axvline(gi_train_median, linestyle="--", linewidth=1.2, color="k", label="Train median")

# ---------------- Panel B: ΔGI ----------------
sns.histplot(dgi_tr, bins=30, stat="density", kde=True, element="step",
             fill=False, linewidth=1.0, label="Train", ax=ax2)
sns.histplot(dgi_te, bins=30, stat="density", kde=True, element="step",
             fill=False, linewidth=1.0, label="Test", ax=ax2)
ax2.axvline(0, linestyle="--", linewidth=1.2, color="k")

# ax2.set_title("ΔGI = GI − median(GI_train) (Train vs Test)")
ax2.set_xlabel("ΔGI")
ax2.set_ylabel("Density")

txt2 = "\n".join([
    f"Overall: M={st_dgi_all['median']:.3f}",
    f"IQR {st_dgi_all['q1']:.3f}-{st_dgi_all['q3']:.3f}",
    f"Range {st_dgi_all['min']:.3f}-{st_dgi_all['max']:.3f}",
    f"n={st_dgi_all['n']}"
])
ax2.text(0.56, 0.95, txt2, transform=ax2.transAxes, va="top", ha="left",
         fontsize=9, bbox=dict(boxstyle="round", facecolor="white",
                               alpha=0.85, linewidth=0.8))

# 统一图例：放在两张图上方“轴外”
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True,
           bbox_to_anchor=(0.5, 0.1), fontsize=9)

# 如果你的 save_fig 会 tight_layout，这里不要再 tight_layout；用 bbox_inches 即可
save_fig("Fig1_GI_and_dGI_distribution.pdf", close=False)
save_fig("Fig1_GI_and_dGI_distribution.tiff", close=True)


# ============================================================
# Step 1) Build predictors, split
# ============================================================

y = df0[DELTA_GI_COL].values.astype(float)
X_df, feat_names = build_feature_matrix(df0, y_col=DELTA_GI_COL)
print(f"[X] numeric predictors before cleaning: {X_df.shape[1]}")

# drop constant columns
nunique = X_df.nunique(dropna=False)
const_cols = nunique[nunique <= 1].index.tolist()
if const_cols:
    X_df = X_df.drop(columns=const_cols)
    feat_names = X_df.columns.tolist()
    print(f"[X] dropped constant cols={len(const_cols)}, remaining={len(feat_names)}")

X_train_raw = X_df.iloc[idx_train].copy()
X_test_raw  = X_df.iloc[idx_test].copy()
y_train = y[idx_train]
y_test  = y[idx_test]

P0 = X_train_raw.shape[1]

# ============================================================
# Step 2) Train-only missingness filter + Fig0
# ============================================================

X_train_raw, X_test_raw, dropped_missing, miss_pct_sorted = drop_high_missing_train_only(
    X_train_raw, X_test_raw, MISSING_THRESHOLD_PERCENT
)
feat_names = X_train_raw.columns.tolist()

P_drop = len(dropped_missing)
P1 = X_train_raw.shape[1]

print(f"[Missing filter] Dropped {P_drop} (> {MISSING_THRESHOLD_PERCENT}%) | Remaining={P1}")

pd.DataFrame({"dropped_missing_cols": dropped_missing}).to_csv(
    OUT_DIR / "dropped_missing_cols.csv", index=False, encoding="utf-8-sig"
)

top_miss = miss_pct_sorted.head(min(30, len(miss_pct_sorted))).reset_index()
top_miss.columns = ["feature", "missing_percent"]
plt.figure(figsize=(10, 8))
sns.barplot(data=top_miss, y="feature", x="missing_percent")
plt.axvline(MISSING_THRESHOLD_PERCENT, linestyle="--")
# plt.title("Top Missingness Features (Train only)")
plt.xlabel("Missing (%)")
plt.ylabel("")
save_fig("Fig0_missingness_top30_train.pdf", close=False)
save_fig("Fig0_missingness_top30_train.tiff", close=True)

# ============================================================
# Step 3) Preprocess + ElasticNetCV (fit on train only)
# ============================================================

preproc = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

Xtr = preproc.fit_transform(X_train_raw)
Xte = preproc.transform(X_test_raw)

enet_cv = ElasticNetCV(
    l1_ratio=L1_RATIOS,
    alphas=ALPHAS,
    cv=5,
    max_iter=5000,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
enet_cv.fit(Xtr, y_train)

best_alpha = float(enet_cv.alpha_)
best_l1 = float(enet_cv.l1_ratio_) if np.isscalar(enet_cv.l1_ratio_) else float(enet_cv.l1_ratio_[0])

# Refit final model using selected params
enet = ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=50000, random_state=RANDOM_STATE)
enet.fit(Xtr, y_train)

# Predictions
pred_tr = enet.predict(Xtr)
pred_te = enet.predict(Xte)

m_tr = compute_metrics(y_train, pred_tr)
m_te = compute_metrics(y_test, pred_te)

print(f"\n[ElasticNetCV] alpha={best_alpha:.6g}, l1_ratio={best_l1:.2f}")
print(f"[Perf] Train: R2={m_tr['R2']:.3f}, MAE={m_tr['MAE']:.3f}, RMSE={m_tr['RMSE']:.3f}")
print(f"[Perf] Test : R2={m_te['R2']:.3f}, MAE={m_te['MAE']:.3f}, RMSE={m_te['RMSE']:.3f}")

# Bootstrap CI on TEST metrics (recommended for manuscript)
ci_te = bootstrap_ci_metrics(y_test, pred_te, n_boot=BOOT_N, seed=BOOT_RANDOM_STATE)
print("\n[Perf CI] Test bootstrap 95% CI")
print(f"  R2  : {ci_te['R2'][0]:.3f} to {ci_te['R2'][1]:.3f}")
print(f"  MAE : {ci_te['MAE'][0]:.3f} to {ci_te['MAE'][1]:.3f}")
print(f"  RMSE: {ci_te['RMSE'][0]:.3f} to {ci_te['RMSE'][1]:.3f}")

# Save performance table
perf_df = pd.DataFrame([
    {"set": "Train", **m_tr, "R2_CI_low": np.nan, "R2_CI_high": np.nan,
     "MAE_CI_low": np.nan, "MAE_CI_high": np.nan, "RMSE_CI_low": np.nan, "RMSE_CI_high": np.nan},
    {"set": "Test",  **m_te, "R2_CI_low": ci_te["R2"][0], "R2_CI_high": ci_te["R2"][1],
     "MAE_CI_low": ci_te["MAE"][0], "MAE_CI_high": ci_te["MAE"][1],
     "RMSE_CI_low": ci_te["RMSE"][0], "RMSE_CI_high": ci_te["RMSE"][1]},
])
perf_df.to_csv(OUT_DIR / "model_performance_metrics.csv", index=False, encoding="utf-8-sig")
print(f"[Saved] {(OUT_DIR / 'model_performance_metrics.csv').resolve()}")

coef = pd.Series(enet.coef_, index=feat_names)
coef_abs = coef.abs()
coef_abs_norm = coef_abs / (coef_abs.max() + 1e-12)
P_enet = int((coef != 0).sum())

# ============================================================
# -------------------------

# ---- fixed physical size (single-column recommended)
# Fig2A: ElasticNetCV curve (single panel for AI layout)
# -------------------------
mse_path = enet_cv.mse_path_
alphas = np.asarray(enet_cv.alphas_)

l1_list = np.atleast_1d(enet_cv.l1_ratio_).astype(float)

# ---- fixed physical size (single-column recommended)

mse_path = enet_cv.mse_path_
mean_mse = mse_path.mean(axis=2)  # (n_l1, n_alpha)
alphas_used = enet_cv.alphas_

# align with provided L1_RATIOS
n_l1 = mean_mse.shape[0]
l1_to_plot = L1_RATIOS[:n_l1]

# ---- single-column friendly size ----
W_MM = 90
W_IN = W_MM / 25.4
H_IN = W_IN * 0.75
fig, ax = plt.subplots(1, 1, figsize=(W_IN, H_IN), constrained_layout=False)

# ---- plot: de-emphasize non-best curves ----
# find best row
i_best = l1_to_plot.index(best_l1) if best_l1 in l1_to_plot else int(np.argmin(mean_mse.min(axis=1)))

for i, l1 in enumerate(l1_to_plot):
    if i == i_best:
        # highlight best l1 curve
        ax.plot(
            alphas_used, mean_mse[i],
            linewidth=2.0,
            label=f"best l1={l1:.2f}"
        )
    else:
        # light gray reference curves (no legend)
        ax.plot(
            alphas_used, mean_mse[i],
            linewidth=1.0,
            alpha=0.25,
            color="gray"
        )

# best alpha vertical line (in legend)
ax.axvline(best_alpha, linestyle="--", linewidth=1.2, label="best α")

# mark best point on best curve
j_best = int(np.argmin(np.abs(alphas_used - best_alpha)))
ax.scatter(best_alpha, mean_mse[i_best, j_best], s=35, zorder=6)

# ---- axes formatting ----
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("alpha (log scale)")
ax.set_ylabel("Mean CV-MSE")
ax.margins(x=0.02, y=0.08)

# ---- small annotation ----
ax.text(
    0.02, 0.98,
    f"best α={best_alpha:.3g}, best l1={l1_to_plot[i_best]:.2f}",
    transform=ax.transAxes,
    va="top", ha="left",
    fontsize=8,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, linewidth=0.6)
)

# ---- legend: minimal and inside (now it won't clash) ----
ax.legend(
    loc="upper right",
    frameon=False,
    fontsize=8,
    handlelength=1.6
)

save_fig("Fig2A_ElasticNetCV_curve.pdf", close=False)
save_fig("Fig2A_ElasticNetCV_curve.tiff", close=True)



# ============================================================
# Fig2A_path: Coefficient path at best l1_ratio
# ============================================================

alphas_path, coefs_path, _ = enet_path(
    Xtr, y_train,
    l1_ratio=best_l1,
    alphas=np.sort(ALPHAS)[::-1],   # stronger reg first
)
# coefs_path: (n_features, n_alphas)

# choose top features by maximum absolute coefficient along the path
max_abs = np.max(np.abs(coefs_path), axis=1)
TOP_SHOW = 25
top_idx = np.argsort(max_abs)[::-1][:min(TOP_SHOW, len(max_abs))]

# ---- export mapping table for the caption / supplement ----
mapping = pd.DataFrame({
    "rank": np.arange(1, len(top_idx) + 1),
    "feature": [feat_names[j] for j in top_idx],
    "max_abs_coef_path": max_abs[top_idx]
})
mapping.to_csv(OUT_DIR / "Fig2A_path_feature_legend_mapping.csv",
               index=False, encoding="utf-8-sig")

# ---- figure size (MUST NOT change) ----
W_MM = 90
W_IN = W_MM / 25.4
H_IN = W_IN * 0.75
fig, ax = plt.subplots(1, 1, figsize=(W_IN, H_IN), constrained_layout=False)

# ---- plot curves with numeric labels ----
for rank, j in enumerate(top_idx, start=1):
    ax.plot(alphas_path, coefs_path[j, :], linewidth=1.1, label=str(rank))

# selected alpha
ax.axvline(best_alpha, linestyle="--", linewidth=1.3, label="selected α")

# mark best point on path (optional): pick the curve with largest |coef| at best_alpha
j_best = int(np.argmin(np.abs(alphas_path - best_alpha)))
# find which feature has largest abs coef at best_alpha point
abs_at_best = np.abs(coefs_path[:, j_best])
k_star = int(np.argmax(abs_at_best))
ax.scatter(best_alpha, coefs_path[k_star, j_best], s=20, zorder=6)

# axes
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("alpha (log scale)")
ax.set_ylabel("Coefficient (standardized X)")
ax.margins(x=0.02, y=0.08)

# small annotation
# ax.text(
#     0.02, 0.98,
#     f"l1={best_l1:.2f}  (Top{len(top_idx)} paths labeled by rank; see mapping table)",
#     transform=ax.transAxes, va="top", ha="left",
#     fontsize=7.5,
#     bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, linewidth=0.6)
# )

# ---- legend: numeric labels only (compact) ----
# Put legend above the plot to avoid shrinking data region
ax.legend(
    loc="lower left",
    bbox_to_anchor=(0.0, 1.02, 1.0, 0.2),
    mode="expand",
    ncol=7,               # 25个编号：7列基本两行
    frameon=False,
    fontsize=7,
    handlelength=1.0,
    columnspacing=0.5,
    borderaxespad=0.0
)

fig.subplots_adjust(top=0.78)

save_fig("Fig2A_path_CoefficientPath.pdf", close=False)
save_fig("Fig2A_path_CoefficientPath.tiff", close=True)

# ============================================================
# Optional: Repeated outer CV on training (gives distribution + CI-like)
# ============================================================

if DO_REPEATED_CV:
    rkf = RepeatedKFold(n_splits=KFOLDS, n_repeats=REPEATS, random_state=CV_RANDOM_STATE)
    cv_rows = []
    fold_id = 0

    for tr_idx, va_idx in rkf.split(X_train_raw):
        fold_id += 1
        X_tr_f = X_train_raw.iloc[tr_idx].copy()
        X_va_f = X_train_raw.iloc[va_idx].copy()
        y_tr_f = y_train[tr_idx]
        y_va_f = y_train[va_idx]

        # train-only missingness is already done globally; here just standard preproc
        pre_f = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])

        Xtr_f = pre_f.fit_transform(X_tr_f)
        Xva_f = pre_f.transform(X_va_f)

        # inner CV tuning within the fold
        enetcv_f = ElasticNetCV(
            l1_ratio=L1_RATIOS,
            alphas=ALPHAS,
            cv=5,
            max_iter=50000,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        enetcv_f.fit(Xtr_f, y_tr_f)

        a_f = float(enetcv_f.alpha_)
        l1_f = float(enetcv_f.l1_ratio_) if np.isscalar(enetcv_f.l1_ratio_) else float(enetcv_f.l1_ratio_[0])

        m_f = ElasticNet(alpha=a_f, l1_ratio=l1_f, max_iter=50000, random_state=RANDOM_STATE)
        m_f.fit(Xtr_f, y_tr_f)
        pred_va = m_f.predict(Xva_f)

        met = compute_metrics(y_va_f, pred_va)
        cv_rows.append({
            "fold": fold_id,
            "alpha": a_f,
            "l1_ratio": l1_f,
            "R2": met["R2"],
            "MAE": met["MAE"],
            "RMSE": met["RMSE"],
        })

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(OUT_DIR / "repeated_outer_cv_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"[Saved] {(OUT_DIR / 'repeated_outer_cv_metrics.csv').resolve()}")

    # summarize (percentile interval)
    cv_sum = {}
    for k in ["R2", "MAE", "RMSE"]:
        cv_sum[k] = {
            "mean": float(cv_df[k].mean()),
            "p2.5": float(np.quantile(cv_df[k], 0.025)),
            "p97.5": float(np.quantile(cv_df[k], 0.975))
        }
    cv_sum_df = pd.DataFrame(cv_sum).T.reset_index().rename(columns={"index": "metric"})
    cv_sum_df.to_csv(OUT_DIR / "repeated_outer_cv_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[Saved] {(OUT_DIR / 'repeated_outer_cv_summary.csv').resolve()}")

# ============================================================
# Step 4) Stability selection (train only)
# ============================================================

stab = stability_selection_elasticnet(
    X=Xtr, y=y_train, feature_names=feat_names,
    alpha=best_alpha, l1_ratio=best_l1,
    n_boot=N_BOOT, subsample_frac=SUBSAMPLE_FRAC,
    random_state=RANDOM_STATE
)
stab.to_csv(OUT_DIR / "stability_frequency.csv", index=False, encoding="utf-8-sig")
print("[Stability] saved: stability_frequency.csv")

stable_candidates = stab.loc[stab["select_freq"] >= FREQ_THRESHOLD, "feature"].tolist()
P_stable = len(stable_candidates)
print(f"[Stability] freq >= {FREQ_THRESHOLD}: {P_stable} candidates")

if P_stable == 0:
    raise RuntimeError("No stable candidates. Lower FREQ_THRESHOLD (0.4–0.5) or increase N_BOOT.")

# ============================================================
# Fig2B: Stability frequency distribution (key figure)
# ============================================================

W_MM = 180
W_IN = W_MM / 25.4
H_IN = W_IN * 0.50  # 1x2 panels

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W_IN, H_IN), constrained_layout=True)
ax1 = plt.subplot(1, 2, 1)
sns.histplot(stab["select_freq"], bins=25, ax=ax1)
plt.axvline(FREQ_THRESHOLD, linestyle="--")
plt.title("Stability Selection Frequency")
plt.xlabel("Selection frequency")
plt.ylabel("Count")

ax2 = plt.subplot(1, 2, 2)
topN = min(20, len(stab))
top_stab = stab.head(topN).copy()
sns.barplot(data=top_stab, y="feature", x="select_freq", ax=ax2)
plt.axvline(FREQ_THRESHOLD, linestyle="--")
plt.title(f"Top {topN} Selection Frequencies")
plt.xlabel("Selection frequency")
plt.ylabel("")

save_fig("Fig2B_StabilityFrequency.pdf", close=False)
save_fig("Fig2B_StabilityFrequency.tiff", close=True)

# ============================================================
# Step 5) Correlation pruning among stable candidates (train median impute only)
# ============================================================

X_train_med = X_train_raw.copy()
med_train = X_train_med.median(numeric_only=True)
X_train_med = X_train_med.fillna(med_train)

cand_df = X_train_med[stable_candidates].copy()
stab_scores = stab.set_index("feature")["select_freq"]

combined_score = (0.8 * stab_scores.reindex(stable_candidates).fillna(0.0) +
                  0.2 * coef_abs_norm.reindex(stable_candidates).fillna(0.0))

kept, corr_mat = correlation_prune_spearman(cand_df, combined_score, corr_thr=CORR_THRESHOLD)
P_final = len(kept)

print(f"[Corr prune] |rho| >= {CORR_THRESHOLD}: {len(stable_candidates)} -> {P_final}")

final_rank = pd.DataFrame({
    "feature": kept,
    "stability_freq": stab_scores.reindex(kept).values,
    "abs_coef_train": coef_abs.reindex(kept).values,
    "combined_score": combined_score.reindex(kept).values
}).sort_values("combined_score", ascending=False)

final_rank.to_csv(OUT_DIR / "final_driver_rank.csv", index=False, encoding="utf-8-sig")
final_top = final_rank.head(FINAL_TOPK).copy()
final_top.to_csv(OUT_DIR / f"final_top{FINAL_TOPK}_drivers.csv", index=False, encoding="utf-8-sig")

print("\n========== FINAL TOP DRIVERS ==========")
print(final_top.to_string(index=False))

# ============================================================
# ============================================================
# Fig2C: Feature reduction across pipeline (single-column)
# ============================================================
# Fig2C: Feature reduction across pipeline (single-column, compact labels)
# X-axis uses P0..P5 only; full definitions moved to caption.
# Style: step plot + markers + value labels (journal-friendly)
# ============================================================

# map stages to compact codes
stage_code = ["P0", "P1", "P2", "P3", "P4", "P5"]

# keep your counts (edit variable names if your script differs)
counts = [
    P0,                          # numeric features
    P1,                          # after missingness filter
    P_enet,                      # Elastic Net non-zero
    P_stable,                    # stability freq threshold
    P_final,                     # correlation pruning
    min(FINAL_TOPK, P_final)     # TopK
]

reduction_df = pd.DataFrame({
    "stage": stage_code,
    "n_features": counts
})

# --- single-column sizing (90 mm width) ---
W_MM = 90
W_IN = W_MM / 25.4
H_IN = W_IN * 0.62  # slightly shorter for a compact single-column panel

fig, ax = plt.subplots(1, 1, figsize=(W_IN, H_IN), constrained_layout=True)

# Step/stairstep line (more "process" feel than a plain line)
ax.step(
    reduction_df["stage"],
    reduction_df["n_features"],
    where="mid",
    linewidth=1.8
)

# Overlay markers
ax.plot(
    reduction_df["stage"],
    reduction_df["n_features"],
    marker="o",
    linewidth=0,
    markersize=5.5
)

# Axis formatting (no title, caption will describe)
ax.set_xlabel("")
ax.set_ylabel("Number of features")

# Y grid only (subtle)
ax.grid(True, axis="y", linewidth=0.6, alpha=0.6)
ax.grid(False, axis="x")

# Annotate counts (slight offset, avoid overlap at top)
ymax = max(reduction_df["n_features"])
for x, yv in zip(reduction_df["stage"], reduction_df["n_features"]):
    dy = 0.03 * ymax
    ax.text(x, yv + dy, f"{int(yv)}", ha="center", va="bottom", fontsize=8)

# Tight y-range for aesthetics
ax.set_ylim(0, ymax * 1.18)

# Clean spines
sns.despine(ax=ax, top=True, right=True)

# Make x tick labels minimal and clean
ax.tick_params(axis="x", labelsize=9)
ax.tick_params(axis="y", labelsize=9)

save_fig("Fig2C_FeatureReductionAcrossPipeline.pdf", close=False)
save_fig("Fig2C_FeatureReductionAcrossPipeline.tiff", close=True)

# ============================================================
# Step 6) Prepare full matrix (all samples, impute by train medians)
# ============================================================

# ============================================================
# Fig3: Top10 Spearman correlation heatmap (two-column, cleaner)
# ============================================================
X_all = X_df.copy() 
for c in X_all.columns: 
    if c in med_train.index: X_all[c] = X_all[c].fillna(med_train[c]) 
    else: X_all[c] = X_all[c].fillna(X_all[c].median()) 
y_all = df0[DELTA_GI_COL].values.astype(float)
top10 = final_top["feature"].head(min(TOP_HEATMAP, len(final_top))).tolist()
corr_top10 = X_all[top10].corr(method="spearman")

# --- reorder by hierarchical clustering to highlight correlation blocks ---
# (use 1-|rho| as distance; clip for numerical stability)
import scipy.cluster.hierarchy as sch
dist = 1.0 - corr_top10.abs()
dist = dist.clip(lower=0.0, upper=2.0)

# linkage expects condensed distance matrix
link = sch.linkage(sch.distance.squareform(dist.values, checks=False), method="average")
order = sch.leaves_list(link)
corr_ord = corr_top10.iloc[order, order]

# --- mask upper triangle (keep lower triangle only) ---
mask = np.triu(np.ones_like(corr_ord, dtype=bool), k=1)

# --- annotations: only show |rho| >= threshold to avoid clutter ---
ANNOT_THR = 0.40
annot_mat = corr_ord.copy()
annot_mat = annot_mat.where(~mask)  # keep only lower triangle
annot_txt = annot_mat.applymap(lambda v: f"{v:.2f}" if pd.notna(v) and abs(v) >= ANNOT_THR else "")

# --- two-column sizing ---
W_MM = 180
W_IN = W_MM / 25.4
H_IN = W_IN * 0.62  

fig, ax = plt.subplots(1, 1, figsize=(W_IN, H_IN), constrained_layout=False)

sns.heatmap(
    corr_ord,
    mask=mask,
    vmin=-1, vmax=1, center=0,
    cmap="RdBu_r",
    square=True,
    linewidths=0.4,
    cbar_kws={"label": "Spearman ρ", "shrink": 0.85, "pad": 0.02},
    annot=annot_txt, fmt="",
    annot_kws={"fontsize": 8},
    ax=ax
)

# no in-figure title; move to caption
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(axis="x", rotation=35)
for t in ax.get_xticklabels():
    t.set_horizontalalignment("right")

# slightly smaller tick labels for 10 vars
ax.tick_params(axis="both", labelsize=9)

save_fig("Fig3_top10_spearman_heatmap.pdf", close=False)
save_fig("Fig3_top10_spearman_heatmap.tiff", close=True)


# ============================================================
# Fig4: Top4 scatter + LOWESS (chosen by |Spearman(ΔGI, X)|)
# ============================================================

from scipy.stats import spearmanr

# ---- choose Top4 by |Spearman(ΔGI, X)| ----
cand_for_show = final_top["feature"].tolist()
spearman_list = []
for f in cand_for_show:
    x = X_all[f].astype(float).values
    rho, p = spearmanr(x, y_all)
    spearman_list.append((f, rho, p, abs(rho)))

sp_df = (pd.DataFrame(spearman_list, columns=["feature", "rho", "p", "abs_rho"])
         .sort_values("abs_rho", ascending=False))

show4 = sp_df["feature"].head(min(TOP4_SHOW, len(sp_df))).tolist()

print("\n[Fig4] Top4 by |Spearman(ΔGI,X)|:")
print(sp_df.head(min(10, len(sp_df))).to_string(index=False))


# ============================================================
# Fig4: Top4 drivers (2×2), two-column figure, compact style
# ============================================================

# --- two-column sizing (JACMP-like): 180 mm width ---
W_MM = 180
W_IN = W_MM / 25.4

# 2×2 layout: choose a compact height ratio
H_IN = W_IN * 0.62  # ~11 cm when W=18 cm, good for 2×2
fig, axes = plt.subplots(2, 2, figsize=(W_IN, H_IN), constrained_layout=True)
axes = np.array(axes).reshape(-1)

# Panel labels
panel_labels = ["A", "B", "C", "D"]

# To avoid per-axes legends exploding: collect handles from the first axis only
global_handles = None
global_labels = None

for i, feat in enumerate(show4):
    ax = axes[i]

    # x preprocessing
    x_raw = X_all[feat].astype(float)
    xw = winsorize_series(x_raw, WINSOR_P[0], WINSOR_P[1]).values

    # density background
    hb = ax.hexbin(xw, y_all, gridsize=35, mincnt=1, cmap="Blues")

    # LOWESS
    xs, ys = lowess_xy(xw, y_all, frac=0.35)
    l1, = ax.plot(xs, ys, linewidth=2.0, label="LOWESS")

    # binned mean + CI
    bm = binned_mean_ci(xw, y_all, q=BIN_Q)
    if bm is not None:
        centers, mean, low, up, nbin = bm
        l2, = ax.plot(centers, mean, marker="o", markersize=3.5, linewidth=1.4,
                      label=f"Binned mean (q={BIN_Q})")
        ax.fill_between(centers, low, up, alpha=0.18, label="95% CI")

    # reference line ΔGI=0
    ax.axhline(0, linestyle="--", linewidth=1.0)

    # stats text + panel label (no long title)
    rho = float(sp_df.loc[sp_df["feature"] == feat, "rho"].values[0])
    p = float(sp_df.loc[sp_df["feature"] == feat, "p"].values[0])

    txt = ( f"Spearman ρ={rho:.3f}, p={p:.3g}")
    txt1=(f"{panel_labels[i]}")
    ax.text(0.05, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, linewidth=0.6))
    ax.text(-0.1, 0.97, txt1, transform=ax.transAxes, va="top", ha="left",
            fontsize=9,fontweight='bold',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, linewidth=1.0))
    # axis labels (keep short; full meaning goes to caption)
    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel("ΔGI", fontsize=9)

    # collect handles/labels from first subplot only for a global legend
    if i == 0:
        global_handles, global_labels = ax.get_legend_handles_labels()

# remove unused axes if <4
for j in range(len(show4), 4):
    fig.delaxes(axes[j])

# global legend (single, clean)
if global_handles and global_labels:
    fig.legend(global_handles, global_labels,
               loc="lower center", ncol=3, frameon=True,
               fontsize=9, bbox_to_anchor=(0.5, -0.08))

# colorbar (single for all) — compact
cbar = fig.colorbar(hb, ax=axes[:len(show4)], shrink=0.85, pad=0.01)
cbar.set_label("Counts", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# Save (same output path)
out_pdf = OUT_DIR / "Fig4_top4_scatter_lowess.pdf"
out_tif = OUT_DIR / "Fig4_top4_scatter_lowess.tiff"
fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
fig.savefig(out_tif, dpi=600, bbox_inches="tight")
plt.close(fig)
print(f"[Saved Figure] {out_pdf.resolve()}")
# ============================================================
# FigS3: Partial correlation / adjusted relationship (control PTV_volume_cc)
# ============================================================

def residualize_linear(z: np.ndarray, cov: np.ndarray):
    z = np.asarray(z).astype(float).reshape(-1)
    cov = np.asarray(cov).astype(float)
    cov = sm.add_constant(cov, has_constant="add")
    model = sm.OLS(z, cov, missing="drop").fit()
    return model.resid

def partial_corr_residuals(x, y, cov):
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(cov), axis=1)
    x2, y2, cov2 = x[m], y[m], cov[m]
    if len(x2) < 30:
        return np.nan, np.nan, np.nan, np.nan, None, None
    rx = residualize_linear(x2, cov2)
    ry = residualize_linear(y2, cov2)
    sr, sp = spearmanr(rx, ry)
    pr, pp = pearsonr(rx, ry)
    return float(sr), float(sp), float(pr), float(pp), rx, ry

top15 = final_top["feature"].tolist()
cov = X_all[[PTV_COL]].astype(float).values
y_vec = y_all.astype(float)

rows = []
for f in top15:
    x_vec = X_all[f].astype(float).values
    sr, sp, pr, pp, _, _ = partial_corr_residuals(x_vec, y_vec, cov)
    rows.append({
        "feature": f,
        "partial_spearman_rho": sr,
        "partial_spearman_p": sp,
        "partial_pearson_r": pr,
        "partial_pearson_p": pp
    })

pc_df = pd.DataFrame(rows).sort_values("partial_spearman_rho", key=lambda s: s.abs(), ascending=False)
pc_df.to_csv(OUT_DIR / "partial_corr_top15.csv", index=False, encoding="utf-8-sig")
print(f"[Saved] {(OUT_DIR / 'partial_corr_top15.csv').resolve()}")

pc_top4 = pc_df["feature"].head(min(4, len(pc_df))).tolist()
print("\n[FigS3] Top4 by |partial Spearman| (adjusted for PTV):")
print(pc_df.head(10).to_string(index=False))

# ============================================================
# FigS3: Top4 by |partial Spearman| (adjusted for PTV) — 2×2, two-column
# ============================================================

# residualize y once (against PTV)
ry_all = residualize_linear(y_vec, cov)

# --- two-column sizing (JACMP-like): 180 mm width ---
W_MM = 180
W_IN = W_MM / 25.4
H_IN = W_IN * 0.62  # compact for 2×2

fig, axes = plt.subplots(2, 2, figsize=(W_IN, H_IN), constrained_layout=True)
axes = np.array(axes).reshape(-1)

panel_labels = ["A", "B", "C", "D"]
global_handles, global_labels = None, None
hb_last = None

for i, f in enumerate(pc_top4[:4]):
    ax = axes[i]

    x_vec = X_all[f].astype(float).values
    rx_all = residualize_linear(x_vec, cov)

    # density background (hexbin)
    hb = ax.hexbin(rx_all, ry_all, gridsize=35, mincnt=1, cmap="Blues")
    hb_last = hb

    # LOWESS
    z = sm.nonparametric.lowess(endog=ry_all, exog=rx_all, frac=0.35, it=1, return_sorted=True)
    ax.plot(z[:, 0], z[:, 1], linewidth=2.0, label="LOWESS")

    # binned mean + CI (q=BIN_Q, e.g. 10 deciles)
    bm = binned_mean_ci(rx_all, ry_all, q=BIN_Q)
    if bm is not None:
        centers, mean, low, up, nbin = bm
        ax.plot(centers, mean, marker="o", markersize=3.5, linewidth=1.4,
                label=f"Binned mean (q={BIN_Q})")
        ax.fill_between(centers, low, up, alpha=0.18, label="95% CI")

    # correlations on residuals (partial)
    sr, sp = spearmanr(rx_all, ry_all)
    pr, pp = pearsonr(rx_all, ry_all)

    # reference lines at 0 (both axes)
    ax.axhline(0, linestyle="--", linewidth=1.0)
    ax.axvline(0, linestyle="--", linewidth=1.0)

    # panel label + stats (no long title)
    txt = (
           f"Partial Spearman ρ={sr:.3f}, p={sp:.3g}\n"
           f"Partial Pearson r={pr:.3f}, p={pp:.3g}")
    text2=(f"{panel_labels[i]}")
    ax.text(0.05, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, linewidth=0.6))
    ax.text(-0.12, 0.97, text2, transform=ax.transAxes, va="top", ha="left",
            fontsize=9,fontweight='bold',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, linewidth=1))
    # axis labels (short; explain in caption)
    ax.set_xlabel(f"Residualized {f}", fontsize=9)
    ax.set_ylabel("Residualized ΔGI", fontsize=9)

    if i == 0:
        global_handles, global_labels = ax.get_legend_handles_labels()

# remove unused axes if <4
for j in range(len(pc_top4[:4]), 4):
    fig.delaxes(axes[j])

# global legend (single)
if global_handles and global_labels:
    fig.legend(global_handles, global_labels,
               loc="lower center", ncol=3, frameon=True,
               fontsize=9, bbox_to_anchor=(0.5, -0.08))

# global colorbar (single)
if hb_last is not None:
    cbar = fig.colorbar(hb_last, ax=axes[:len(pc_top4[:4])], shrink=0.85, pad=0.01)
    cbar.set_label("Counts", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

# Save (same output filenames)
out_pdf = OUT_DIR / "FigS3_top4_partialcorr_residual_plots.pdf"
out_tif = OUT_DIR / "FigS3_top4_partialcorr_residual_plots.tiff"
fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
fig.savefig(out_tif, dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"[Saved Figure] {out_pdf.resolve()}")
# Optional: partial Spearman heatmap (Top10)
# ============================================================
# FigS3: Partial Spearman heatmap (Top10) — two-column, clean
# ============================================================

top10_pc = pc_df["feature"].head(min(10, len(pc_df))).tolist()
pc_vals = pc_df.set_index("feature").loc[top10_pc, "partial_spearman_rho"].astype(float)

# sort by magnitude for readability (optional; comment out if you want original order)
pc_vals = pc_vals.reindex(pc_vals.abs().sort_values(ascending=False).index)

hm_df = pd.DataFrame([pc_vals.values], columns=pc_vals.index.tolist(),
                     index=[f"Partial Spearman (adjusted for {PTV_COL})"])

# --- two-column sizing ---
W_MM = 180
W_IN = W_MM / 25.4
H_IN = W_IN * 0.18  # slim single-row heatmap

fig, ax = plt.subplots(1, 1, figsize=(W_IN, H_IN), constrained_layout=True)

sns.heatmap(
    hm_df,
    vmin=-1, vmax=1, center=0,
    cmap="RdBu_r",
    annot=True, fmt=".2f",
    linewidths=0.4,
    cbar_kws={"label": "Partial Spearman ρ", "shrink": 0.95, "pad": 0.01},
    annot_kws={"fontsize": 8},
    ax=ax
)

# no in-figure title; move to caption
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(axis="x", rotation=35, labelsize=9)
for t in ax.get_xticklabels():
    t.set_horizontalalignment("right")
ax.tick_params(axis="y", labelsize=9)

# IMPORTANT: do not call plt.tight_layout() because constrained_layout is used
fig.savefig(OUT_DIR / "FigS4_partialcorr_heatmap_top10.pdf", dpi=600, bbox_inches="tight")
fig.savefig(OUT_DIR / "FigS4_partialcorr_heatmap_top10.tiff", dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"[Saved Figure] {(OUT_DIR / 'FigS4_partialcorr_heatmap_top10.pdf').resolve()}")
print("\n[DONE] All figures + metrics (MAE/RMSE/CI/CV) generated successfully.")
