"""
Production Heatloss Prediction Model - v2
==========================================
Key improvements over v1:
- Era-specific U-values that correctly model post-2008 Building Regs (Part L 2010/2013)
- Modern timber frame corrected from flat 0.43 to era-bracketed 0.17-0.50
- Infiltration reduced for modern builds (0.3 ACH vs 0.5 in v1)
- Explicit era flags (is_post_2008, is_modern) added as ML features so the
  gradient boosters can learn residuals the physics proxy misses
- BETWEEN_2000_2008 treated as its own era (previously lumped with 1960-2000)
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')


# =============================================================================
# ERA HELPERS
# =============================================================================

def _era(row):
    """Return normalised era string from property_age field."""
    age = str(row.get('property_age', '')).upper()
    if 'POST_2008' in age or 'POST2008' in age:
        return 'POST_2008'
    if '2000_2008' in age or '2000-2008' in age:
        return '2000_2008'
    if '1960_2000' in age or '1960-2000' in age:
        return '1960_2000'
    if 'PRE_1960' in age or 'PRE-1960' in age:
        return 'PRE_1960'
    return 'UNKNOWN'


# =============================================================================
# PHYSICS KNOWLEDGE BASE  (corrected U-values, W/m²K)
# =============================================================================

class UValueMapper:

    @staticmethod
    def get_wall_u_value(row):
        cons  = str(row.get('walls_construction_type', '')).upper()
        depth = str(row.get('final_walls_depth', ''))
        era   = _era(row)

        # Parse wall depth (mm)
        try:
            d = float(''.join(filter(str.isdigit, depth))) if depth else 250
        except Exception:
            d = 250

        # --- Stone ---
        if 'STONE' in cons:
            if d <= 350: return 2.78
            if d <= 500: return 2.23
            return 1.68

        # --- Solid brick ---
        if 'SOLID' in cons:
            if d <= 150: return 2.97
            if d <= 280: return 2.11
            return 1.64

        # --- Timber frame (era-specific) ---
        # v1 used a flat 0.43 for all eras – badly wrong for modern builds.
        # UK Building Regs progressively tightened: Part L 2006, 2010, 2013, 2021.
        if 'TIMBER' in cons:
            if era == 'POST_2008':   return 0.17   # Part L 2010 compliant
            if era == '2000_2008':   return 0.25   # Part L 2006 era
            if era == '1960_2000':   return 0.40   # older timber kits
            return 0.50                             # PRE_1960 / unknown

        # --- Cavity wall (era-specific) ---
        # v1 used 0.42 (filled) / 0.77 (unfilled) for all non-pre-1960 cavity.
        # Post-2008 Part L requires max 0.18 W/m²K for new walls.
        if 'CAVITY' in cons:
            is_filled = ('FILLED' in cons or
                         'INSULATED' in str(row.get('walls_insulation', '')).upper())

            if era == 'PRE_1960':
                return 0.56 if is_filled else 1.37

            if era == '1960_2000':
                return 0.35 if is_filled else 0.60

            if era == '2000_2008':
                return 0.25 if is_filled else 0.35   # Part L 2006 tightened regs

            if era == 'POST_2008':
                return 0.18 if is_filled else 0.25   # Part L 2010 / modern new-build

            # Unknown era – conservative mid-range
            return 0.30 if is_filled else 0.55

        # Fallback
        return 1.5

    @staticmethod
    def get_window_u_value(row):
        glazing = str(row.get('windows_glazing', '')).upper()
        if 'TRIPLE' in glazing:
            return 1.6
        if 'SINGLE' in glazing:
            return 4.8
        if 'LOW_E' in glazing or 'LOW-E' in glazing or 'LOW E' in glazing:
            return 1.8   # double low-e (was 2.3 in v1)
        return 2.8        # standard double

    @staticmethod
    def get_roof_u_value(row):
        roof = str(row.get('roof_type', '')).upper()
        ins  = str(row.get('roof_insulation_thickness', ''))
        era  = _era(row)

        try:
            d = float(''.join(filter(str.isdigit, ins))) if ins else 100
        except Exception:
            d = 100

        is_flat = 'FLAT' in roof

        if is_flat:
            if d >= 200: return 0.17
            if d >= 100: return 0.32
            if d >= 50:  return 0.53
            return 1.69
        else:   # pitched
            # Post-2008 with deep insulation – Part L requires ~0.11-0.16
            if d >= 300:
                return 0.11 if era == 'POST_2008' else 0.13
            if d >= 200:
                return 0.15 if era == 'POST_2008' else 0.18
            if d >= 100: return 0.34
            if d >= 50:  return 0.60
            return 2.51

    @staticmethod
    def get_floor_u_value(row):
        ins = str(row.get('final_floor_insulation_type', ''))
        try:
            d = float(''.join(filter(str.isdigit, ins))) if ins else 50
        except Exception:
            d = 50

        if d >= 100: return 0.22
        if d >= 75:  return 0.28
        if d >= 50:  return 0.40
        if d >= 25:  return 0.55
        return 0.70

    @staticmethod
    def get_infiltration(row):
        """
        Air change rate (ACH). v1 used 0.5 for all post-2008.
        Part L 2010+ requires air permeability ≤10 m³/h·m², good modern builds ≤5.
        """
        era = _era(row)
        if era == 'PRE_1960':   return 1.5
        if era == '1960_2000':  return 1.0
        if era == '2000_2008':  return 0.6
        if era == 'POST_2008':  return 0.3   # was 0.5 in v1
        return 0.8


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def calculate_physics_features(df):
    df = df.copy()

    # U-values
    df['u_wall']       = df.apply(UValueMapper.get_wall_u_value,  axis=1)
    df['u_window']     = df.apply(UValueMapper.get_window_u_value, axis=1)
    df['u_roof']       = df.apply(UValueMapper.get_roof_u_value,   axis=1)
    df['u_floor']      = df.apply(UValueMapper.get_floor_u_value,  axis=1)
    df['infiltration'] = df.apply(UValueMapper.get_infiltration,   axis=1)

    # Floor area (handle column name variants)
    area_col = ('ashp_survey_total_floor_area_sqm'
                if 'ashp_survey_total_floor_area_sqm' in df.columns
                else 'ashp_survey_total_area_sqm')
    df['floor_area'] = df[area_col].fillna(df[area_col].median())

    # --- Era flags (explicit features so the ML can learn residuals) ---
    df['era'] = df.apply(_era, axis=1)
    df['is_post_2008']   = (df['era'] == 'POST_2008').astype(int)
    df['is_2000_2008']   = (df['era'] == '2000_2008').astype(int)
    df['is_modern']      = ((df['era'] == 'POST_2008') |
                             (df['era'] == '2000_2008')).astype(int)
    df['is_pre_1960']    = (df['era'] == 'PRE_1960').astype(int)

    # --- Physics aggregates ---
    df['total_u_sum']  = (df['u_wall'] + df['u_window'] +
                          df['u_roof'] + df['u_floor'] + df['infiltration'])

    DELTA_T = 24.2   # standard UK design day (21°C internal, −3.2°C external)
    df['physics_heatloss_proxy'] = df['floor_area'] * df['total_u_sum'] * DELTA_T

    # Interaction: does floor area × era make sense?
    df['area_x_post2008']  = df['floor_area'] * df['is_post_2008']
    df['area_x_modern']    = df['floor_area'] * df['is_modern']
    df['area_x_pre1960']   = df['floor_area'] * df['is_pre_1960']

    # Log transforms
    df['log_physics_proxy'] = np.log1p(df['physics_heatloss_proxy'])
    df['log_floor_area']    = np.log1p(df['floor_area'])

    return df


physics_transformer = FunctionTransformer(calculate_physics_features)


# =============================================================================
# MODEL
# =============================================================================

NUMERIC_FEATURES = [
    'floor_area', 'u_wall', 'u_window', 'u_roof', 'u_floor', 'infiltration',
    'physics_heatloss_proxy', 'total_u_sum',
    'log_physics_proxy', 'log_floor_area',
    # era flags
    'is_post_2008', 'is_2000_2008', 'is_modern', 'is_pre_1960',
    # interactions
    'area_x_post2008', 'area_x_modern', 'area_x_pre1960',
]


class HeatlossProductionModel(BaseEstimator):
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.main_pipe    = None
        self.safety_pipe  = None

    def _build_pipeline(self):
        xgb  = XGBRegressor(
            n_estimators=1200, learning_rate=0.04, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=self.random_state, n_jobs=-1, verbosity=0)

        cat  = CatBoostRegressor(
            iterations=1000, learning_rate=0.05, depth=6,
            random_state=self.random_state, verbose=0)

        lgbm = LGBMRegressor(
            n_estimators=1200, learning_rate=0.04, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=self.random_state, verbose=-1)

        ensemble = VotingRegressor(estimators=[
            ('xgb', xgb), ('cat', cat), ('lgbm', lgbm)
        ])

        quantile = LGBMRegressor(
            objective='quantile', alpha=0.8,
            n_estimators=600, learning_rate=0.04, max_depth=5,
            random_state=self.random_state, verbose=-1)

        prep = ColumnTransformer([
            ('num', SimpleImputer(strategy='median'), NUMERIC_FEATURES)
        ])

        main_pipe = Pipeline([
            ('physics', physics_transformer),
            ('prep',    prep),
            ('model',   ensemble),
        ])

        safety_pipe = Pipeline([
            ('physics', physics_transformer),
            ('prep',    prep),
            ('model',   quantile),
        ])

        return main_pipe, safety_pipe

    def fit(self, X, y):
        self.main_pipe, self.safety_pipe = self._build_pipeline()
        self.main_pipe.fit(X, y)
        self.safety_pipe.fit(X, y)
        return self

    def predict(self, X):
        pred_main   = self.main_pipe.predict(X)
        pred_safety = self.safety_pipe.predict(X)
        return pd.DataFrame({
            'predicted_heatloss':    pred_main,
            'safety_estimate':       pred_safety,
            'is_unserviceable_risk': (pred_main > 15000) | (pred_safety > 15000),
        })


if __name__ == '__main__':
    print("HeatlossProductionModel v2 — import and use HeatlossProductionModel class.")