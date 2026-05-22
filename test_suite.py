"""
Heat Loss Model — Test Suite
=============================
Runs a proper 80/20 stratified train/test split, computes full diagnostics,
and saves a 10-panel visualisation to model_test_results.png.

Usage (from heatloss-backend/):
    python test_suite.py

Outputs:
    model_test_results.png  — full visual report
    model_test_results.txt  — plain-text metrics summary
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from production_model import HeatlossProductionModel
from preprocess import preprocess

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_FILE    = 'heatlossdata.csv'
OUTPUT_DIR   = '.'
TARGET       = 'ashp_survey_total_property_heatloss_w'
AREA_COL     = 'ashp_survey_total_floor_area_sqm'
AGE_COL      = 'property_age'
WALL_COL     = 'walls_construction_type'
RISK_THRESH  = 15_000   # Watts — "unserviceable" boundary
TEST_SIZE    = 0.20
RANDOM_SEED  = 42
AREA_MIN     = 30       # below = individual rooms / data quality issues

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG  = '#0f172a'
CARD_BG  = '#1e293b'
GRID_COL = '#334155'
TEXT_COL = '#f1f5f9'
DIM_COL  = '#94a3b8'

WALL_PAL = {'CAVITY': '#3b82f6', 'SOLID': '#ef4444', 'TIMBER_FRAME': '#22c55e'}
ERA_PAL  = {
    'Pre-1960':  '#7c3aed',
    '1960-2000': '#f59e0b',
    '2000-08':   '#06b6d4',
    'Post-2008': '#10b981',
}
ERA_ORDER = ['Pre-1960', '1960-2000', '2000-08', 'Post-2008']

def era_short(v):
    return {
        'HEAT_PUMP_SURVEY_PROPERTY_AGE_PRE_1960':             'Pre-1960',
        'HEAT_PUMP_SURVEY_PROPERTY_AGE_BETWEEN_1960_2000':    '1960-2000',
        'HEAT_PUMP_SURVEY_PROPERTY_AGE_BETWEEN_2000_2008':    '2000-08',
        'HEAT_PUMP_SURVEY_PROPERTY_AGE_POST_2008':            'Post-2008',
    }.get(str(v), str(v).replace('HEAT_PUMP_SURVEY_PROPERTY_AGE_', ''))

def wall_short(v):
    return str(v).replace('HEAT_PUMP_SURVEY_MATERIAL_WALLS_CONSTRUCTION_TYPE_', '')

def style(ax, title=None):
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=DIM_COL, labelsize=8)
    ax.xaxis.label.set_color(DIM_COL); ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_color(DIM_COL); ax.yaxis.label.set_fontsize(9)
    if title:
        ax.set_title(title, color=TEXT_COL, fontsize=10, pad=6)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.5, alpha=0.6)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & SPLIT
# ─────────────────────────────────────────────────────────────────────────────

print("Loading data ...")
raw = pd.read_csv(DATA_FILE)
print(f"  {len(raw):,} rows loaded")
df = preprocess(raw)

strat = df[AGE_COL].fillna('UNKNOWN')
train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=strat
)
print(f"  Train: {len(train_df):,}   Test: {len(test_df):,}")

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET].values
X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET].values


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAIN ON 80%
# ─────────────────────────────────────────────────────────────────────────────

print("Training model on 80% split (this takes ~60-90 seconds) ...")
model = HeatlossProductionModel(random_state=RANDOM_SEED)
model.fit(X_train, y_train)
print("  Done.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREDICT ON 20% HOLDOUT
# ─────────────────────────────────────────────────────────────────────────────

print("Running predictions on test set ...")
preds = model.predict(X_test)
y_pred       = preds['predicted_heatloss'].values
y_safety     = preds['safety_estimate'].values
is_risk_pred = preds['is_unserviceable_risk'].values


# ─────────────────────────────────────────────────────────────────────────────
# 4. CORE METRICS
# ─────────────────────────────────────────────────────────────────────────────

residuals  = y_pred - y_test
abs_err    = np.abs(residuals)
pct_err    = abs_err / y_test * 100

mae        = mean_absolute_error(y_test, y_pred)
rmse       = np.sqrt(mean_squared_error(y_test, y_pred))
r2         = r2_score(y_test, y_pred)
mape       = np.mean(pct_err)
med_ape    = np.median(pct_err)

# Bias
bias_w     = np.mean(residuals)          # + = overestimate
bias_pct   = bias_w / np.mean(y_test) * 100

# Within-band accuracy
within_10  = np.mean(pct_err <= 10) * 100
within_20  = np.mean(pct_err <= 20) * 100

# Risk classification (>15 kW)
actual_risk  = (y_test >= RISK_THRESH)
tp = np.sum( is_risk_pred &  actual_risk)
fp = np.sum( is_risk_pred & ~actual_risk)
fn = np.sum(~is_risk_pred &  actual_risk)
tn = np.sum(~is_risk_pred & ~actual_risk)
recall    = tp / (tp + fn + 1e-9)
precision = tp / (tp + fp + 1e-9)
f1        = 2 * recall * precision / (recall + precision + 1e-9)

# Also show what recall/precision would be using main prediction only (no safety boost)
is_risk_main_only = (y_pred > RISK_THRESH)
tp_m = np.sum( is_risk_main_only &  actual_risk)
fp_m = np.sum( is_risk_main_only & ~actual_risk)
fn_m = np.sum(~is_risk_main_only &  actual_risk)
recall_main    = tp_m / (tp_m + fn_m + 1e-9)
precision_main = tp_m / (tp_m + fp_m + 1e-9)

# Confidence interval coverage (safety_estimate is 80th-percentile)
ci_coverage = np.mean(y_test <= y_safety) * 100

# ─────────────────────────────────────────────────────────────────────────────
# 5. ANNOTATED TEST DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────

res = test_df.copy()
res['y_actual']  = y_test
res['y_pred']    = y_pred
res['y_safety']  = y_safety
res['residual']  = residuals
res['abs_err']   = abs_err
res['pct_err']   = pct_err
res['wall_type'] = res[WALL_COL].apply(wall_short)
res['era_s']     = res[AGE_COL].apply(era_short)
res['hl_band']   = pd.cut(y_test,
    bins=[0, 4000, 7000, 10000, 15000, 50000],
    labels=['<4 kW', '4–7 kW', '7–10 kW', '10–15 kW', '>15 kW'])

# ─────────────────────────────────────────────────────────────────────────────
# 6. PER-CATEGORY BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────

def cat_metrics(grp):
    return pd.Series({
        'n':        len(grp),
        'MAE_W':    grp['abs_err'].mean(),
        'MAPE_%':   grp['pct_err'].mean(),
        'Med_APE%': grp['pct_err'].median(),
        'Bias_W':   grp['residual'].mean(),
        'R2':       r2_score(grp['y_actual'], grp['y_pred']) if len(grp) > 5 else np.nan,
    })

wall_stats = res.groupby('wall_type').apply(cat_metrics).round(2)
era_stats  = res.groupby('era_s').apply(cat_metrics).round(2)
band_stats = res.groupby('hl_band', observed=True).apply(cat_metrics).round(2)

# ─────────────────────────────────────────────────────────────────────────────
# 7. CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

lines = []
lines.append("="*62)
lines.append(f"HEAT LOSS MODEL — TEST RESULTS   (80/20 split, seed={RANDOM_SEED})")
lines.append(f"Train: {len(y_train):,}   Test: {len(y_test):,}")
lines.append("="*62)
lines.append(f"  R²                 {r2:.4f}")
lines.append(f"  MAE                {mae:,.0f} W  ({mae/1000:.2f} kW)")
lines.append(f"  RMSE               {rmse:,.0f} W")
lines.append(f"  MAPE               {mape:.1f}%")
lines.append(f"  Median APE         {med_ape:.1f}%")
lines.append(f"  Bias               {bias_w:+,.0f} W  ({bias_pct:+.1f}%)")
lines.append(f"  Within ±10%        {within_10:.1f}%")
lines.append(f"  Within ±20%        {within_20:.1f}%")
lines.append("-"*62)
lines.append(f"  >15 kW Recall      {recall*100:.1f}%  ({tp}/{tp+fn} caught)  [main only: {recall_main*100:.1f}%]")
lines.append(f"  >15 kW Precision   {precision*100:.1f}%  [main only: {precision_main*100:.1f}%]")
lines.append(f"  >15 kW F1          {f1:.3f}")
lines.append(f"  Safety CI covers   {ci_coverage:.1f}% of actuals  (target ≈ 80%)")
lines.append("="*62)
lines.append("\nBY WALL TYPE")
lines.append(wall_stats[['n','MAE_W','MAPE_%','Bias_W']].to_string())
lines.append("\nBY ERA")
lines.append(era_stats[['n','MAE_W','MAPE_%','Bias_W']].to_string())
lines.append("\nBY HEAT LOSS BAND")
lines.append(band_stats[['n','MAE_W','MAPE_%','Bias_W']].to_string())

# Best / worst
lines.append("\n\n10 BEST PREDICTIONS (lowest % error)")
best10 = res.nsmallest(10, 'pct_err')[[
    'y_actual','y_pred','abs_err','pct_err','wall_type','era_s',AREA_COL]]
best10.columns = ['Actual_W','Pred_W','AbsErr_W','PctErr%','Wall','Era','Area_m2']
lines.append(best10.round(1).to_string(index=False))

lines.append("\n\n10 WORST PREDICTIONS (highest % error)")
worst10 = res.nlargest(10, 'pct_err')[[
    'y_actual','y_pred','abs_err','pct_err','wall_type','era_s',AREA_COL]]
worst10.columns = ['Actual_W','Pred_W','AbsErr_W','PctErr%','Wall','Era','Area_m2']
lines.append(worst10.round(1).to_string(index=False))

# ── Large-property validation ─────────────────────────────────────────────────
# Targeted check for the element-area proxy improvement. If MAPE for >200m²
# properties hasn't improved vs the expected baseline (~30%+), the v2 proxy
# features are not pulling their weight and should be removed.
LARGE_THRESH = 200
large_mask = res[AREA_COL] > LARGE_THRESH
large = res[large_mask]
lines.append("\n" + "="*62)
lines.append(f"LARGE PROPERTY CHECK  (>{LARGE_THRESH} m²,  n={len(large)})")
lines.append("="*62)
if len(large) > 0:
    lg_mape = large['pct_err'].mean()
    lg_mae  = large['abs_err'].mean()
    lg_bias = large['residual'].mean()
    lg_r2   = r2_score(large['y_actual'], large['y_pred']) if len(large) > 5 else float('nan')
    lines.append(f"  MAPE   {lg_mape:.1f}%   (overall: {mape:.1f}%)")
    lines.append(f"  MAE    {lg_mae:,.0f} W")
    lines.append(f"  Bias   {lg_bias:+,.0f} W  ({'overestimate' if lg_bias > 0 else 'underestimate'})")
    lines.append(f"  R²     {lg_r2:.3f}")
    lines.append("")
    lines.append("  Worst 5 large-property predictions:")
    worst_large = large.nlargest(5, 'pct_err')[[
        'y_actual','y_pred','pct_err','wall_type','era_s',AREA_COL]]
    worst_large.columns = ['Actual_W','Pred_W','PctErr%','Wall','Era','Area_m2']
    lines.append(worst_large.round(0).to_string(index=False))
    verdict = ("PASS — element-area proxy appears effective for large properties"
               if lg_mape < mape * 1.5 else
               "REVIEW — large-property MAPE is >1.5× overall; consider removing v2 proxy features")
    lines.append(f"\n  Verdict: {verdict}")
else:
    lines.append("  No properties above threshold in test set.")

summary_text = "\n".join(lines)
print("\n" + summary_text)

txt_path = os.path.join(OUTPUT_DIR, 'model_test_results.txt')
with open(txt_path, 'w') as f:
    f.write(summary_text)
print(f"\nText summary saved → {txt_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

print("Building visualisation ...")

fig = plt.figure(figsize=(20, 28))
fig.patch.set_facecolor(DARK_BG)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.48, wspace=0.34,
                       top=0.95, bottom=0.04)

# ── PANEL 1: Predicted vs Actual (coloured by wall type) ────────────────────
ax1 = fig.add_subplot(gs[0, 0:2])

lim = max(y_test.max(), y_pred.max()) * 1.05
for wt, c in WALL_PAL.items():
    m = res['wall_type'] == wt
    ax1.scatter(res.loc[m, 'y_actual']/1000, res.loc[m, 'y_pred']/1000,
                c=c, s=4, alpha=0.35, label=wt.replace('_', ' ').title(), rasterized=True)

ax1.plot([0, lim/1000], [0, lim/1000], '--', color='#f59e0b', lw=1.4, label='Perfect')
ax1.fill_between([0, lim/1000], [0, lim*0.8/1000], [0, lim*1.2/1000],
                  alpha=0.07, color='#f59e0b', label='±20% band')
ax1.set_xlim(0, lim/1000); ax1.set_ylim(0, lim/1000)
ax1.set_xlabel('Actual Heat Loss (kW)')
ax1.set_ylabel('Predicted Heat Loss (kW)')
ax1.legend(fontsize=8, facecolor=CARD_BG, labelcolor=DIM_COL,
           edgecolor=GRID_COL, markerscale=3)
ax1.text(0.97, 0.04,
         f'n = {len(y_test):,} | R² = {r2:.3f} | MAE = {mae/1000:.2f} kW',
         transform=ax1.transAxes, ha='right', va='bottom',
         fontsize=8.5, color=TEXT_COL,
         bbox=dict(fc=GRID_COL, ec='none', pad=4, alpha=0.8))
style(ax1, f'Fig 1 — Predicted vs Actual')

# ── PANEL 2: Residual distribution ──────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])

ax2.hist(residuals/1000, bins=70, color='#3b82f6', alpha=0.75, edgecolor='none')
ax2.axvline(0,                     color='#f59e0b', lw=1.6, ls='--', label='Zero')
ax2.axvline(np.median(residuals)/1000, color='#ef4444', lw=1.4, ls=':',
            label=f'Median {np.median(residuals)/1000:+.1f} kW')
ax2.axvline(bias_w/1000,           color='#22c55e', lw=1.2, ls='-.',
            label=f'Mean {bias_w/1000:+.1f} kW')
ax2.set_xlabel('Residual  (kW)   +ve = overestimate')
ax2.set_ylabel('Count')
ax2.legend(fontsize=7.5, facecolor=CARD_BG, labelcolor=DIM_COL, edgecolor=GRID_COL)
ax2.text(0.97, 0.95,
         f'MAPE   {mape:.1f}%\nMed APE {med_ape:.1f}%\nBias   {bias_pct:+.1f}%',
         transform=ax2.transAxes, ha='right', va='top',
         fontsize=8, color=TEXT_COL,
         bbox=dict(fc=GRID_COL, ec='none', pad=4, alpha=0.8))
style(ax2, 'Fig 2 — Residual Distribution')

# ── PANEL 3: % Error by wall type ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])

wall_order = ['CAVITY', 'SOLID', 'TIMBER_FRAME']
data3 = [res[res['wall_type'] == w]['pct_err'].values for w in wall_order]
bp3 = ax3.boxplot(data3, patch_artist=True, notch=False,
                  medianprops=dict(color='white', lw=2),
                  whiskerprops=dict(color=DIM_COL),
                  capprops=dict(color=DIM_COL),
                  flierprops=dict(marker='.', ms=2, color=GRID_COL))
for patch, wt in zip(bp3['boxes'], wall_order):
    patch.set_facecolor(WALL_PAL[wt]); patch.set_alpha(0.8)
ax3.set_xticklabels(['Cavity', 'Solid', 'Timber'], fontsize=9)
ax3.set_ylabel('Absolute % Error')
all3 = np.concatenate(data3)
ax3.set_ylim(0, min(80, np.percentile(all3, 96)))
for i, d in enumerate(data3, 1):
    if len(d):
        ax3.text(i, np.median(d) + 0.4,
                 f'Med {np.median(d):.1f}%\nn={len(d):,}',
                 ha='center', fontsize=7.5, color='white')
style(ax3, 'Fig 3 — % Error by Wall Type')

# ── PANEL 4: % Error by era ─────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])

data4 = [res[res['era_s'] == e]['pct_err'].values for e in ERA_ORDER]
bp4 = ax4.boxplot(data4, patch_artist=True, notch=False,
                  medianprops=dict(color='white', lw=2),
                  whiskerprops=dict(color=DIM_COL),
                  capprops=dict(color=DIM_COL),
                  flierprops=dict(marker='.', ms=2, color=GRID_COL))
for patch, era in zip(bp4['boxes'], ERA_ORDER):
    patch.set_facecolor(ERA_PAL[era]); patch.set_alpha(0.8)
ax4.set_xticklabels(ERA_ORDER, fontsize=8, rotation=15, ha='right')
ax4.set_ylabel('Absolute % Error')
valid4 = [d for d in data4 if len(d) > 0]
ax4.set_ylim(0, min(80, np.percentile(np.concatenate(valid4), 96)))
for i, (e, d) in enumerate(zip(ERA_ORDER, data4), 1):
    if len(d):
        ax4.text(i, np.median(d) + 0.4,
                 f'Med {np.median(d):.1f}%',
                 ha='center', fontsize=7.5, color='white')
style(ax4, 'Fig 4 — % Error by Era')

# ── PANEL 5: % Error by heat loss band ──────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])

band_labels = ['<4 kW', '4–7 kW', '7–10 kW', '10–15 kW', '>15 kW']
band_colors = ['#22c55e', '#84cc16', '#f59e0b', '#f97316', '#ef4444']
data5 = [res[res['hl_band'] == l]['pct_err'].values for l in band_labels]
bp5 = ax5.boxplot(data5, patch_artist=True, notch=False,
                  medianprops=dict(color='white', lw=2),
                  whiskerprops=dict(color=DIM_COL),
                  capprops=dict(color=DIM_COL),
                  flierprops=dict(marker='.', ms=2, color=GRID_COL))
for patch, c in zip(bp5['boxes'], band_colors):
    patch.set_facecolor(c); patch.set_alpha(0.8)
ax5.set_xticklabels(band_labels, fontsize=8, rotation=15, ha='right')
ax5.set_ylabel('Absolute % Error')
valid5 = [d for d in data5 if len(d) > 0]
ax5.set_ylim(0, min(80, np.percentile(np.concatenate(valid5), 96)))
for i, d in enumerate(data5, 1):
    if len(d):
        ax5.text(i, np.median(d) + 0.4,
                 f'Med {np.median(d):.1f}%\nn={len(d):,}',
                 ha='center', fontsize=7, color='white')
style(ax5, 'Fig 5 — % Error by Heat Loss Band\n(does model degrade at extremes?)')

# ── PANEL 6: Residuals vs Predicted (heteroscedasticity check) ──────────────
ax6 = fig.add_subplot(gs[2, 0:2])

sc = ax6.scatter(y_pred/1000, residuals/1000,
                 c=pct_err, cmap='RdYlGn_r', vmin=0, vmax=30,
                 s=3, alpha=0.3, rasterized=True)
ax6.axhline(0,          color='#f59e0b', lw=1.4, ls='--', label='Zero error')
ax6.axhline( mae/1000,  color=DIM_COL,   lw=0.8, ls=':', alpha=0.7)
ax6.axhline(-mae/1000,  color=DIM_COL,   lw=0.8, ls=':', alpha=0.7,
            label=f'±MAE  ({mae/1000:.1f} kW)')
ax6.axvline(RISK_THRESH/1000, color='#ef4444', lw=1.2, ls='--', alpha=0.7,
            label=f'{RISK_THRESH/1000:.0f} kW threshold')

cb = fig.colorbar(sc, ax=ax6, pad=0.01, fraction=0.015)
cb.set_label('% Error', color=DIM_COL, fontsize=8)
cb.ax.yaxis.set_tick_params(color=DIM_COL, labelcolor=DIM_COL, labelsize=7)

ax6.set_xlabel('Predicted Heat Loss (kW)')
ax6.set_ylabel('Residual (kW)   +ve = overestimate')
ax6.legend(fontsize=8, facecolor=CARD_BG, labelcolor=DIM_COL, edgecolor=GRID_COL)
style(ax6, 'Fig 6 — Residuals vs Predicted  (should be flat random band; fanning = scale bias)')

# ── PANEL 7: Risk detection confusion matrix ─────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')
ax7.set_facecolor(CARD_BG)
for sp in ax7.spines.values(): sp.set_edgecolor(GRID_COL)

cm_vals = np.array([[tn, fp], [fn, tp]])
cm_labels = [['True Neg\n(Low → Low)', 'False Pos\n(Low → High)'],
             ['False Neg\n(High → Low)', 'True Pos\n(High → High)']]
cm_colors = [['#1e3a5f', '#7c3aed'], ['#7f1d1d', '#15803d']]

for i in range(2):
    for j in range(2):
        x0, y0 = j * 0.5, (1 - i) * 0.48 + 0.02
        ax7.add_patch(plt.Rectangle((x0, y0), 0.48, 0.44,
                                    fc=cm_colors[i][j], ec=GRID_COL, lw=1.5,
                                    transform=ax7.transAxes))
        ax7.text(x0 + 0.24, y0 + 0.26, f'{cm_vals[i,j]:,}',
                 transform=ax7.transAxes, ha='center', va='center',
                 fontsize=16, color='white', fontweight='bold')
        ax7.text(x0 + 0.24, y0 + 0.11, cm_labels[i][j],
                 transform=ax7.transAxes, ha='center', va='center',
                 fontsize=7.5, color='#cbd5e1')

ax7.text(0.5, 0.97,
         f'Recall {recall*100:.1f}%   Precision {precision*100:.1f}%   F1 {f1:.2f}',
         transform=ax7.transAxes, ha='center', va='top',
         fontsize=9, color=TEXT_COL)
ax7.set_title(f'Fig 7 — >15 kW Risk Detection', color=TEXT_COL, fontsize=10, pad=6)

# ── PANEL 8: Confidence interval coverage ───────────────────────────────────
ax8 = fig.add_subplot(gs[3, 0])

# Sort worst 300 by absolute error for visibility
w300 = res.nlargest(300, 'abs_err').reset_index(drop=True)
idx  = range(len(w300))
ax8.fill_between(idx,
                 w300['y_pred'].values / 1000,
                 w300['y_safety'].values / 1000,
                 alpha=0.4, color='#f59e0b', label='Safety range')
ax8.scatter(idx, w300['y_actual'].values / 1000,
            s=5, color='#ef4444', alpha=0.7, label='Actual', zorder=3)
ax8.scatter(idx, w300['y_pred'].values / 1000,
            s=5, color='#3b82f6', alpha=0.7, label='Predicted', zorder=3)
ax8.set_xlabel('Cases ranked by error')
ax8.set_ylabel('Heat Loss (kW)')
ax8.legend(fontsize=7.5, facecolor=CARD_BG, labelcolor=DIM_COL, edgecolor=GRID_COL)
ax8.text(0.97, 0.97,
         f'Safety CI covers {ci_coverage:.1f}%\nof all test actuals\n(target ≈ 80%)',
         transform=ax8.transAxes, ha='right', va='top',
         fontsize=8, color=TEXT_COL,
         bbox=dict(fc=GRID_COL, ec='none', pad=4, alpha=0.8))
style(ax8, 'Fig 8 — Worst 300: Actual vs Predicted\nwith Safety Estimate (80th pct) band')

# ── PANEL 9: Wall × Era error heatmap ───────────────────────────────────────
ax9 = fig.add_subplot(gs[3, 1])

pivot = res.pivot_table(values='pct_err', index='wall_type', columns='era_s',
                        aggfunc='median')[ERA_ORDER].reindex(
    ['CAVITY', 'SOLID', 'TIMBER_FRAME'])

im = ax9.imshow(pivot.values, cmap='RdYlGn_r', vmin=0, vmax=30, aspect='auto')
ax9.set_xticks(range(4)); ax9.set_xticklabels(ERA_ORDER, fontsize=8, rotation=15, ha='right', color=DIM_COL)
ax9.set_yticks(range(3)); ax9.set_yticklabels(['Cavity', 'Solid', 'Timber'], fontsize=9, color=DIM_COL)
for i in range(3):
    for j in range(4):
        val = pivot.values[i, j]
        if not np.isnan(val):
            n = len(res[(res['wall_type'] == pivot.index[i]) &
                        (res['era_s'] == ERA_ORDER[j])])
            ax9.text(j, i, f'{val:.1f}%\n(n={n})',
                     ha='center', va='center', fontsize=7.5,
                     color='white' if val > 15 else '#1e293b', fontweight='bold')
cb9 = fig.colorbar(im, ax=ax9, pad=0.01, fraction=0.04)
cb9.set_label('Median % Error', color=DIM_COL, fontsize=7)
cb9.ax.yaxis.set_tick_params(color=DIM_COL, labelcolor=DIM_COL, labelsize=6)
ax9.set_facecolor(CARD_BG)
for sp in ax9.spines.values(): sp.set_edgecolor(GRID_COL)
ax9.set_title('Fig 9 — Median % Error: Wall × Era Heatmap\n(red = worst, green = best)',
              color=TEXT_COL, fontsize=10, pad=6)
ax9.grid(False)

# ── PANEL 10: Summary metrics card ──────────────────────────────────────────
ax10 = fig.add_subplot(gs[3, 2])
ax10.axis('off')
ax10.set_facecolor(CARD_BG)
for sp in ax10.spines.values(): sp.set_edgecolor(GRID_COL)

def metric_row(ax, y, label, value, color=DIM_COL, big=False):
    fs = 11 if big else 9.5
    ax.text(0.04, y, label,  transform=ax.transAxes, fontsize=fs,
            color=DIM_COL, fontfamily='monospace')
    ax.text(0.96, y, value,  transform=ax.transAxes, fontsize=fs,
            color=color, fontfamily='monospace', ha='right',
            fontweight='bold' if color != DIM_COL else 'normal')

def sep(ax, y):
    ax.axhline(y, xmin=0.02, xmax=0.98, color=GRID_COL, lw=0.7,
               transform=ax.transAxes)

# Use transAxes-compatible line drawing
rows = [
    (0.94, 'R²',              f'{r2:.4f}',           '#22c55e' if r2 > 0.85 else '#f59e0b'),
    (0.86, 'MAE',             f'{mae/1000:.2f} kW',  '#22c55e' if mae < 1500 else '#f59e0b'),
    (0.79, 'RMSE',            f'{rmse/1000:.2f} kW', DIM_COL),
    (0.72, 'MAPE',            f'{mape:.1f}%',        '#22c55e' if mape < 15 else '#f59e0b'),
    (0.65, 'Median APE',      f'{med_ape:.1f}%',     '#22c55e' if med_ape < 12 else '#f59e0b'),
    (0.58, 'Bias',            f'{bias_w/1000:+.2f} kW ({bias_pct:+.1f}%)', DIM_COL),
    (0.51, 'Within ±10%',     f'{within_10:.1f}%',   '#22c55e' if within_10 > 55 else '#f59e0b'),
    (0.44, 'Within ±20%',     f'{within_20:.1f}%',   '#22c55e' if within_20 > 80 else '#f59e0b'),
    (0.34, '>15kW Recall',    f'{recall*100:.1f}%   ({tp}/{tp+fn})', '#22c55e' if recall > 0.80 else '#ef4444'),
    (0.27, '>15kW Precision', f'{precision*100:.1f}%',               DIM_COL),
    (0.20, '>15kW F1',        f'{f1:.3f}',                           '#22c55e' if f1 > 0.75 else '#f59e0b'),
    (0.10, 'Safety CI',       f'{ci_coverage:.1f}% covered',         DIM_COL),
]
for y, lbl, val, col in rows:
    metric_row(ax10, y, lbl, val, col)
ax10.axhline(0.41, xmin=0.02, xmax=0.98, color=GRID_COL, lw=0.7)
ax10.axhline(0.56, xmin=0.02, xmax=0.98, color=GRID_COL, lw=0.7)
ax10.set_title('Fig 10 — Summary Metrics\n(green = good, amber = needs attention)',
               color=TEXT_COL, fontsize=10, pad=6)

# ── Global title ────────────────────────────────────────────────────────────
fig.text(0.5, 0.975,
         'Heat Loss Model — Test Suite',
         ha='center', va='top', fontsize=17, fontweight='bold', color=TEXT_COL)
fig.text(0.5, 0.960,
         f'Physics-Hybrid-V3  |  80/20 stratified split  |  area >={AREA_MIN} m²  |  '
         f'Train {len(y_train):,}  Test {len(y_test):,}  |  seed={RANDOM_SEED}',
         ha='center', va='top', fontsize=9, color=DIM_COL)

# ─────────────────────────────────────────────────────────────────────────────
# 9. SAVE
# ─────────────────────────────────────────────────────────────────────────────

png_path = os.path.join(OUTPUT_DIR, 'model_test_results.png')
plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
print(f"Chart saved  → {png_path}")
print(f"Text saved   → {txt_path}")
print("\nDone.")
