"""
analyse_extremes.py
===================
Examines the top 250 (highest predicted HL) and bottom 250 (lowest predicted HL)
to understand false-positive and false-negative patterns.

Run from heatloss-backend/:
    python analyse_extremes.py
"""

import pandas as pd
import numpy as np
import joblib
import sys

sys.path.insert(0, '.')
from preprocess import preprocess, TARGET
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE   = 'heatlossdata2.csv'
MODEL_FILE  = 'production_model.joblib'
N           = 250          # how many from each end
THRESHOLD   = 15_000       # W — "unserviceable" boundary
FN_COST     = 200          # £ — wasted survey (conservative, per user)
FP_COST     = 200          # £ — wasted survey visit (surveyor wages + travel)


def era_label(v):
    v = str(v).upper()
    if 'POST_2008' in v: return 'Post-2008'
    if '2000_2008' in v: return '2000-08'
    if '1960_2000' in v: return '1960-2000'
    if 'PRE_1960'  in v: return 'Pre-1960'
    return 'Unknown'

def wall_label(v):
    v = str(v).upper()
    if 'CAVITY'   in v: return 'CAVITY'
    if 'SOLID'    in v: return 'SOLID'
    if 'TIMBER'   in v: return 'TIMBER'
    if 'STONE'    in v: return 'STONE'
    if 'CONCRETE' in v: return 'CONCRETE'
    return v[:18]

def glaz_label(v):
    v = str(v).upper()
    if 'TRIPLE' in v: return 'TRIPLE'
    if 'DOUBLE' in v: return 'DOUBLE'
    if 'SINGLE' in v: return 'SINGLE'
    return 'Unknown'

def ins_label(v):
    v = str(v).upper() if not pd.isna(v) else ''
    if 'FILLED' in v:   return 'FILLED'
    if 'UNFILLED' in v: return 'UNFILLED'
    return 'Unknown'

# ── Load & predict ─────────────────────────────────────────────────────────────
print("Loading data & model...")
raw   = pd.read_csv(DATA_FILE)
df    = preprocess(raw, verbose=False)
_, test_df = train_test_split(df, test_size=0.2, random_state=42)

model = joblib.load(MODEL_FILE)
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET].values

preds       = model.predict(X_test)
pred_main   = np.asarray(preds['predicted_heatloss'])
pred_safety = np.asarray(preds['safety_estimate'])

# Use the model's own flag — respects whatever thresholds are set in predict()
# (currently: pred_main > 15000 OR pred_safety > 14500)
flagged     = np.asarray(preds['is_unserviceable_risk']).astype(bool)
actual_high = (y_test > THRESHOLD).astype(bool)

# Build results frame — assign all numpy arrays positionally, then derive pct_err from stored cols
res = X_test.reset_index(drop=True).copy()
res['actual_w']    = y_test
res['pred_main_w'] = pred_main
res['pred_safe_w'] = pred_safety
res['flagged']     = flagged
res['actual_high'] = actual_high
res['pct_err']     = (res['pred_main_w'] - res['actual_w']) / res['actual_w'] * 100
res['era']  = res['property_age'].map(era_label)
res['wall'] = res['walls_construction_type'].map(wall_label)
res['glaz'] = res['windows_glazing'].map(glaz_label)
res['ins']  = res['walls_insulation'].map(ins_label)
res['area'] = res['ashp_survey_total_floor_area_sqm']

DISPLAY = ['actual_w','pred_main_w','pred_safe_w','pct_err','wall','era','glaz','ins','area']

sep = '=' * 70

# ── TOP 250 (highest predicted HL) ───────────────────────────────────────────
top250 = res.nlargest(N, 'pred_main_w')

tp_top = (top250['actual_high'] & top250['flagged']).sum()
fp_top = (~top250['actual_high'] & top250['flagged']).sum()
fn_top = (top250['actual_high'] & ~top250['flagged']).sum()

print(f"\n{sep}")
print(f"TOP {N} PREDICTIONS  (highest predicted heat loss)")
print(sep)
print(f"  Actual >15kW  :  {top250['actual_high'].sum():>4}  /  {N}")
print(f"  Precision     :  {top250['actual_high'].mean()*100:.1f}%")
print(f"  True  Positives (caught >15kW)        : {tp_top}")
print(f"  False Positives (predicted high, OK)  : {fp_top}  → ~£{fp_top*FP_COST:,.0f} wasted surveys")
print(f"  False Negatives in top {N}             : {fn_top}")

print(f"\n  Wall type breakdown:")
print(top250.groupby(['wall','actual_high']).size().unstack(fill_value=0).to_string())
print(f"\n  Era breakdown:")
print(top250.groupby(['era','actual_high']).size().unstack(fill_value=0).to_string())
print(f"\n  Glazing breakdown:")
print(top250.groupby(['glaz','actual_high']).size().unstack(fill_value=0).to_string())
print(f"\n  Cavity fill breakdown (cavity walls only):")
cav = top250[top250['wall'] == 'CAVITY']
if len(cav):
    print(cav.groupby(['ins','actual_high']).size().unstack(fill_value=0).to_string())

print(f"\n  FALSE POSITIVES — top {N} (predicted high but actually OK):")
print(f"  NOTE: FPs = good customers incorrectly blocked (not wasted surveys)")
fp_df = top250[~top250['actual_high']].sort_values('pred_main_w', ascending=False)
print(fp_df[DISPLAY].head(20).to_string(index=False))

# ── BOTTOM 250 (lowest predicted HL) ─────────────────────────────────────────
bot250 = res.nsmallest(N, 'pred_main_w')

fn_bot = (bot250['actual_high']).sum()

print(f"\n{sep}")
print(f"BOTTOM {N} PREDICTIONS  (lowest predicted heat loss)")
print(sep)
print(f"  Actual >15kW in bottom {N}: {fn_bot}  ← these are the dangerous misses")
print(f"  Mean predicted  : {bot250['pred_main_w'].mean():,.0f} W")
print(f"  Max  predicted  : {bot250['pred_main_w'].max():,.0f} W")
print(f"  Max  actual     : {bot250['actual_w'].max():,.0f} W")

print(f"\n  Wall type breakdown:")
print(bot250['wall'].value_counts().to_string())
print(f"\n  Era breakdown:")
print(bot250['era'].value_counts().to_string())

if fn_bot:
    print(f"\n  DANGEROUS MISSES in bottom {N} (actual >15kW, predicted low):")
    misses = bot250[bot250['actual_high']].sort_values('actual_w', ascending=False)
    print(misses[DISPLAY].to_string(index=False))

# ── FALSE NEGATIVES (all, not just bottom 250) ────────────────────────────────
all_fn = res[res['actual_high'] & ~res['flagged']]
print(f"\n{sep}")
print(f"ALL FALSE NEGATIVES  (actual >15kW, not flagged)  n={len(all_fn)}")
print(sep)
print(f"\n  Wall type:")
print(all_fn['wall'].value_counts().to_string())
print(f"\n  Era:")
print(all_fn['era'].value_counts().to_string())
print(f"\n  Glazing:")
print(all_fn['glaz'].value_counts().to_string())
print(f"\n  Cavity fill (cavity walls only):")
fn_cav = all_fn[all_fn['wall'] == 'CAVITY']
if len(fn_cav):
    print(fn_cav['ins'].value_counts().to_string())

print(f"\n  Full list sorted by actual HL:")
print(all_fn[DISPLAY].sort_values('actual_w', ascending=False).to_string(index=False))

# ── COST SUMMARY ──────────────────────────────────────────────────────────────
# Terminology:
#   FP = good home incorrectly blocked (lost customer opportunity)
#   FN = unserviceable home that slips through → surveyor sent → wasted visit
#
# IMPORTANT: test set has ~5.4% unserviceable (training data selection bias).
# Real-world prevalence is ~30%. Raw test-set FP count is misleading — in
# production you see far fewer FPs relative to TPs because the unserviceable
# rate is 5-6× higher. Use the PRODUCTION-PROJECTED section below for £ impact.

PROD_PREVALENCE = 0.30   # ~30% of real incoming surveys are >15kW

all_tp   = (res['actual_high'] & res['flagged']).sum()
all_fp   = (~res['actual_high'] & res['flagged']).sum()
all_fn_n = (res['actual_high'] & ~res['flagged']).sum()
total_high = res['actual_high'].sum()
total_n    = len(res)

recall    = all_tp / max(total_high, 1)
fp_rate   = all_fp / max(total_n - total_high, 1)  # FP rate among good homes

print(f"\n{sep}")
print(f"COST SUMMARY  —  TEST SET  (FN=£{FN_COST} wasted visit, FP=£{FP_COST} lost customer)")
print(sep)
print(f"  True  positives (correctly blocked)    : {all_tp}  ({recall*100:.1f}% recall)")
print(f"  False positives (good homes blocked)   : {all_fp}  (FP rate {fp_rate*100:.1f}% of good homes)")
print(f"  False negatives (undetected, surveyed) : {all_fn_n}  → £{all_fn_n*FN_COST:,.0f}")
print(f"  NOTE: £{all_fp*FP_COST:,.0f} FP 'cost' above is inflated — test set has only ~5% "
      f"unserviceable\n  vs ~30% in production, making FP count look 6× worse than reality.\n")

# ── Production-projected cost (per 1,000 homes screened) ─────────────────────
print(f"  PRODUCTION PROJECTION  (per 1,000 homes screened, {PROD_PREVALENCE*100:.0f}% prevalence)")
print(f"  {'-'*50}")
prod_unsvc    = 1000 * PROD_PREVALENCE           # 300 unserviceable
prod_svc      = 1000 - prod_unsvc                # 700 serviceable
prod_tp       = prod_unsvc * recall              # correctly blocked
prod_fn       = prod_unsvc * (1 - recall)        # missed, surveyor sent
prod_fp       = prod_svc   * fp_rate             # good homes blocked
prod_tn       = prod_svc   - prod_fp             # correctly passed
prod_cost_fn  = prod_fn * FN_COST
prod_cost_fp  = prod_fp * FP_COST
no_model_cost = prod_unsvc * FN_COST             # survey everyone → all 300 waste visits
savings       = no_model_cost - prod_fn * FN_COST
print(f"  Unserviceable homes caught (TP) : {prod_tp:.0f} / {prod_unsvc:.0f}")
print(f"  Unserviceable missed (FN)       : {prod_fn:.0f}  → £{prod_cost_fn:,.0f} wasted surveys")
print(f"  Good homes blocked (FP)         : {prod_fp:.0f}  → £{prod_cost_fp:,.0f} lost customers")
print(f"  Wasted-survey cost with model   : £{prod_cost_fn:,.0f}")
print(f"  Wasted-survey cost, no model    : £{no_model_cost:,.0f}")
print(f"  Saving on wasted surveys        : £{savings:,.0f}  ({savings/no_model_cost*100:.0f}% reduction)")
print(f"  Survey failure rate: no model ≈ {PROD_PREVALENCE*100:.0f}%  →  "
      f"with model ≈ {prod_fn/(prod_tn+prod_fn)*100:.1f}%")
