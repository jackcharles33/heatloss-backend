"""
preprocess.py  —  Survey data cleaning pipeline
================================================
Called by train_deploy.py before training.  Can also be run standalone:

    python preprocess.py "heatlossdata2.csv"

Steps
-----
1. Drop rows with null target (safety net — SQL already filters but CSV may not)
2. Area filter  >= AREA_MIN (30 m²) — below this are individual rooms / data errors
3. Drop rows with null walls_construction_type (our primary feature, useless without it)
4. Normalize  walls_insulation  →  FILLED / UNFILLED / None
5. Extract    wall_insulation_mm  →  float  (mm of IWI/EWI on solid walls)
6. Normalize  final_walls_depth  — safety-net remap of any stray BETWEEN_290_310 values
"""

import re
import sys
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET   = 'ashp_survey_total_property_heatloss_w'
AREA_COL = 'ashp_survey_total_floor_area_sqm'
AREA_MIN = 30.0


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalize_walls_insulation(series: pd.Series) -> pd.Series:
    """
    Map the walls_insulation column (COALESCE of 6 survey sources) to one of:
      'FILLED'   — cavity is confirmed filled / insulated
      'UNFILLED' — cavity confirmed unfilled / open
       None      — ambiguous, partial, or not recorded

    The 6 source question keys produce varied value formats:
      Full enum keys:   HEAT_PUMP_SURVEY_MATERIAL_WALLS_INSULATION_TYPE_FILLED
      Short strings:    FILLED, UNFILLED, PARTIAL
      Boolean strings:  true / false (from V2 yes/no questions)
      EPC impact:       SIGNIFICANT (= not insulated), NONE (= already done / N/A)
    """

    # Tokens that unambiguously mean "filled"
    FILLED_TOKENS   = ('_FILLED', 'FILLED_', '_INSULATED', '_YES', '_TRUE')
    UNFILLED_TOKENS = ('_UNFILLED', 'UNFILLED_', '_NOT_INSULATED', '_NO', '_FALSE',
                       'NO_INSULATION', 'NONE_INSTALLED', 'OPEN_CAVITY')

    def _norm(val):
        if pd.isna(val):
            return None
        s = str(val).upper().strip()

        # --- Stray numeric values (mm or other numeric leakage) → unknown ---
        # Values like '2393', '835' are not insulation status signals.
        try:
            float(s)
            return None
        except ValueError:
            pass

        # --- Check UNFILLED first so "UNFILLED" isn't mis-matched by FILLED ---
        if 'UNFILLED' in s or 'NOT_FILLED' in s or 'NO_FILL' in s:
            return 'UNFILLED'

        # --- Open cavity = unfilled ---
        if 'OPEN_CAVITY' in s or 'OPEN CAVITY' in s:
            return 'UNFILLED'

        # --- Explicit filled tokens ---
        if any(tok in s for tok in FILLED_TOKENS):
            return 'FILLED'
        if s in ('TRUE', '1', 'YES', 'Y'):
            return 'FILLED'
        if s in ('FALSE', '0', 'NO', 'N'):
            return 'UNFILLED'

        # --- EPC impact signal -----------------------------------------------
        # A "significant" or "high" EPC recommendation for cavity insulation means
        # it is currently UNFILLED (there is room for improvement).
        # A "low" / "minimal" / "none" impact means it is already done or N/A.
        if 'SIGNIFICANT' in s or 'HIGH_IMPACT' in s or 'LARGE_IMPACT' in s:
            return 'UNFILLED'
        if 'LOW_IMPACT' in s or 'MINIMAL' in s or 'NONE' in s:
            # "NONE" EPC impact = either already insulated or solid wall — treat
            # as unknown here; the wall construction type will carry the real signal.
            return None

        # --- Partial / unknown → don't force into either bucket ---
        if 'PARTIAL' in s or 'UNKNOWN' in s or 'NOT_APPLICABLE' in s:
            return None

        # Anything else we can't classify safely
        return None

    return series.map(_norm)


def extract_wall_insulation_mm(series: pd.Series) -> pd.Series:
    """
    Extract a numeric mm thickness from wall_insulation_mm.

    Source values can be:
      'HEAT_PUMP_SURVEY_V2_INTERNAL_OR_EXTERNAL_WALL_INSULATION_MM_50'
      '50', '100', '75.0', None
    The trailing digits after the last underscore or the whole value are extracted.
    """

    def _extract(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        # Prefer trailing number (handles both enum keys and plain strings)
        m = re.search(r'(\d+(?:\.\d+)?)$', s)
        if m:
            return float(m.group(1))
        try:
            return float(s)
        except ValueError:
            return np.nan

    return series.map(_extract)


def normalize_depth(series: pd.Series) -> pd.Series:
    """
    Normalise final_walls_depth — named enum keys only.

    Raw mm enum keys (HEAT_PUMP_SURVEY_MATERIAL_WALLS_DEPTH_228 etc.) are left
    as-is so production_model.py can extract the actual numeric mm value via
    digit filtering. The cavity wall logic in production_model.py uses numeric d
    as a fallback when no GT_290/LT_290 string is present — so solid/stone/concrete
    walls keep their precise mm depth and cavity walls still get the GT/LT signal.
    """
    replacements = {
        'HEAT_PUMP_SURVEY_MATERIAL_WALLS_DEPTH_BETWEEN_290_310': 'WALLS_DEPTH_GT_290',
        'BETWEEN_290_310':                                        'WALLS_DEPTH_GT_290',
        'WALLS_DEPTH_BETWEEN_290_310':                            'WALLS_DEPTH_GT_290',
        'HEAT_PUMP_SURVEY_MATERIAL_WALLS_DEPTH_MORE_THAN_290':    'WALLS_DEPTH_GT_290',
        'HEAT_PUMP_SURVEY_MATERIAL_WALLS_DEPTH_MORE_THAN_295':    'WALLS_DEPTH_GT_290',
        'HEAT_PUMP_SURVEY_MATERIAL_WALLS_DEPTH_MORE_THAN_310':    'WALLS_DEPTH_GT_290',
        'HEAT_PUMP_SURVEY_MATERIAL_WALLS_DEPTH_LESS_THAN_290':    'WALLS_DEPTH_LT_290',
        'HEAT_PUMP_SURVEY_MATERIAL_WALLS_DEPTH_LESS_THAN_295':    'WALLS_DEPTH_LT_290',
    }
    return series.replace(replacements)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame, area_min: float = AREA_MIN,
               verbose: bool = True) -> pd.DataFrame:
    """
    Clean and normalise a raw survey DataFrame for model training.

    Parameters
    ----------
    df        : raw DataFrame as loaded from CSV
    area_min  : minimum floor area (m²) — rows below this are excluded
    verbose   : print cleaning summary to stdout

    Returns
    -------
    Cleaned DataFrame ready for HeatlossProductionModel.fit()
    """

    def _log(msg):
        if verbose:
            print(msg)

    n0 = len(df)
    _log(f"\n{'─'*55}")
    _log(f"  Preprocessing  — {n0:,} raw rows")
    _log(f"{'─'*55}")

    # ── 1. Target nulls ─────────────────────────────────────────────────────
    if TARGET in df.columns:
        n = len(df)
        df = df.dropna(subset=[TARGET])
        if n - len(df):
            _log(f"  [1] Drop null target:          {len(df):>7,}  ({n - len(df):,} removed)")
        else:
            _log(f"  [1] Target nulls:              none found")

    # ── 2. Area filter ───────────────────────────────────────────────────────
    if AREA_COL in df.columns:
        n = len(df)
        df = df[df[AREA_COL] >= area_min].copy()
        _log(f"  [2] Area filter (>={area_min:.0f} m²):      {len(df):>7,}  ({n - len(df):,} removed)")
    else:
        _log(f"  [2] Area column not found — skipping area filter")

    # ── 3. Wall construction type ────────────────────────────────────────────
    if 'walls_construction_type' in df.columns:
        n = len(df)
        df = df.dropna(subset=['walls_construction_type'])
        _log(f"  [3] Drop null wall type:       {len(df):>7,}  ({n - len(df):,} removed)")
    else:
        _log(f"  [3] walls_construction_type column not found")

    # ── 4. Normalize walls_insulation ────────────────────────────────────────
    if 'walls_insulation' in df.columns:
        raw_nulls = df['walls_insulation'].isna().sum()
        df['walls_insulation'] = normalize_walls_insulation(df['walls_insulation'])
        filled    = (df['walls_insulation'] == 'FILLED').sum()
        unfilled  = (df['walls_insulation'] == 'UNFILLED').sum()
        remaining = df['walls_insulation'].isna().sum()
        _log(f"\n  [4] walls_insulation normalization:")
        _log(f"      FILLED     : {filled:>7,}")
        _log(f"      UNFILLED   : {unfilled:>7,}")
        _log(f"      null/unkn  : {remaining:>7,}  (was {raw_nulls:,} raw nulls)")
    else:
        _log(f"  [4] walls_insulation column not present — "
             f"model will use depth-only fill signal")

    # ── 5. Extract wall_insulation_mm ────────────────────────────────────────
    if 'wall_insulation_mm' in df.columns:
        df['wall_insulation_mm'] = extract_wall_insulation_mm(df['wall_insulation_mm'])
        present = df['wall_insulation_mm'].notna().sum()
        med     = df['wall_insulation_mm'].median()
        _log(f"\n  [5] wall_insulation_mm: {present:,} non-null  "
             f"(median {med:.0f} mm where present)")
    else:
        _log(f"  [5] wall_insulation_mm column not present — skipping")

    # ── 6. Depth normalization (safety net) ──────────────────────────────────
    if 'final_walls_depth' in df.columns:
        before_vals = df['final_walls_depth'].value_counts()
        df['final_walls_depth'] = normalize_depth(df['final_walls_depth'])
        after_vals  = df['final_walls_depth'].value_counts()
        named_remapped = int(
            before_vals.get('HEAT_PUMP_SURVEY_MATERIAL_WALLS_DEPTH_BETWEEN_290_310', 0) +
            before_vals.get('BETWEEN_290_310', 0) +
            before_vals.get('WALLS_DEPTH_BETWEEN_290_310', 0)
        )
        if named_remapped:
            _log(f"\n  [6] Remapped {named_remapped:,} BETWEEN_290_310 → GT_290")
        else:
            _log(f"  [6] Depth normalization: no stray BETWEEN_290_310 found")
    else:
        _log(f"  [6] final_walls_depth column not found — skipping")

    _log(f"\n  {'─'*45}")
    # ── 7. Remove implausible HL/area outliers ───────────────────────────────
    # Heat loss > 200 W/m² is physically impossible for standard UK housing and
    # indicates survey errors or data corruption (e.g. 32,744W for a 218m²
    # cavity home = 150 W/m²). These distort the loss surface for all models.
    if TARGET in df.columns and AREA_COL in df.columns:
        hl_per_sqm = df[TARGET] / df[AREA_COL].clip(lower=1)
        n = len(df)
        df = df[hl_per_sqm <= 200].copy()
        removed = n - len(df)
        if removed:
            _log(f"  [7] Outlier cap (<=200 W/m²):  {len(df):>7,}  ({removed:,} removed)")
        else:
            _log(f"  [7] Outlier cap: no implausible W/m² rows found")

    _log(f"\n  {'─'*45}")
    _log(f"  Final usable rows: {len(df):,}  "
         f"(removed {n0 - len(df):,} / {(n0 - len(df)) / max(n0, 1) * 100:.1f}%)")
    _log(f"  {'─'*45}\n")

    return df


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    fname = sys.argv[1] if len(sys.argv) > 1 else 'heatlossdata.csv'
    print(f"Loading {fname}...")
    raw = pd.read_csv(fname)
    clean = preprocess(raw)
    out = fname.replace('.csv', '_clean.csv')
    clean.to_csv(out, index=False)
    print(f"Saved cleaned data → {out}")
