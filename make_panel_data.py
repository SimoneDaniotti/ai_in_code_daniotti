#!/usr/bin/env python3
"""
process_data.py — Prepares user-quarter panel data for regression analysis.

This script processes raw GitHub commit and AI usage data to produce a 
user-quarter panel suitable for fixed-effects regression. It computes:
  - AI exposure measures (quarterly averages and moving averages)
  - Commit counts (total, multi-file, with-imports)
  - Library combination counts (various granularities)
  - User experience and gender attributes

Inputs (in inputs/ folder):
  - functions.csv: Function-level AI share data (user_hashed, date, ai_share)
  - panel_with_libs_coarse.csv: User-project-quarter commit and library data
  - user_gender_experience.csv: User demographics (gender, year_started)

Output (in outputs/ folder):
  - panel_uq.parquet: User-quarter panel ready for regression analysis

Usage:
  python process_data.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Configuration
IN_DIR = Path("inputs")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Moving average windows for AI exposure measurement
MA_WINDOWS = [4, 8, 16, 32]

# Minimum functions per quarter to compute quarterly AI average
MIN_FUNCTIONS_PER_QUARTER = 10

# Quarter midpoints for interpolating AI measures
MIDPOINT_STRS = [
    "14feb2019", "16may2019", "15aug2019", "15nov2019",
    "15feb2020", "16may2020", "15aug2020", "15nov2020",
    "14feb2021", "16may2021", "15aug2021", "15nov2021",
    "14feb2022", "16may2022", "15aug2022", "15nov2022",
    "14feb2023", "16may2023", "15aug2023", "15nov2023",
    "15feb2024", "16may2024", "15aug2024", "15nov2024",
]
MIDPOINTS = pd.to_datetime(MIDPOINT_STRS, format="%d%b%Y")


def qcode(ts: pd.Series) -> pd.Series:
    """Convert timestamp to quarter code YYYYQ (e.g., 20231 for Q1 2023)."""
    return ts.dt.year * 10 + ts.dt.quarter


def compute_ma_full_window(gdf: pd.DataFrame, w: int) -> pd.Series:
    """
    Compute strict moving average over exactly w function rows.
    
    Uses w//2 lags and ceil(w/2)-1 leads. Returns NA if any value in
    window is missing or if window spans more than 184 days.
    """
    w1 = int(np.ceil(w / 2) - 1)  # leads
    w2 = w // 2  # lags

    val_parts = [gdf['AIf'].shift(-k) for k in range(-w2, w1 + 1)]
    t_parts = [gdf['t'].shift(-k) for k in range(-w2, w1 + 1)]

    val_mat = pd.concat(val_parts, axis=1)
    t_mat = pd.concat(t_parts, axis=1)

    out = val_mat.mean(axis=1, skipna=False)

    # Mask windows with missing data or spanning > 184 days
    complete = val_mat.notna().all(axis=1) & t_mat.notna().all(axis=1)
    span_days = (t_mat.max(axis=1) - t_mat.min(axis=1)).dt.days
    out = out.mask(complete & (span_days > 184))
    
    return out


def process_ai_shares():
    """
    Process function-level AI data to create user-quarter AI exposure measures.
    
    Returns:
        tuple: (user_mapping_df, ai_shares_df)
            - user_mapping_df: IDu_hash to IDu mapping
            - ai_shares_df: User-quarter AI measures (AIav, AIma{w}_i)
    """
    print("Processing AI shares from functions.csv...")
    
    df = pd.read_csv(IN_DIR / "functions.csv")
    df = df.rename(columns={
        'user_hashed': 'IDu_hash',
        'function_modified_names': 'IDf',
        'ai_share': 'AIf',
        'date': 'date_str',
    })
    
    df['t'] = pd.to_datetime(df['date_str'], dayfirst=True, errors='coerce')
    df = df.sort_values(['IDu_hash', 't', 'IDf']).reset_index(drop=True)
    df['q'] = qcode(df['t'])
    
    # Create user ID mapping
    users = df[['IDu_hash']].drop_duplicates().sort_values('IDu_hash').reset_index(drop=True)
    users['IDu'] = np.arange(1, len(users) + 1, dtype=int)
    
    dff = df.merge(users, on='IDu_hash', how='left')
    dff = dff.sort_values(['IDu', 't', 'IDf']).reset_index(drop=True)
    
    # Compute function-level moving averages
    for w in MA_WINDOWS:
        dff[f'AIma{w}'] = dff.groupby('IDu', group_keys=False).apply(
            lambda g: compute_ma_full_window(g, w)
        )
    
    # Create quarter midpoint grid
    mp = pd.DataFrame({'t': MIDPOINTS}).sort_values('t').reset_index(drop=True)
    mp['q'] = qcode(mp['t'])
    mp['key'] = 1
    
    uk = users.copy()
    uk['key'] = 1
    uq = mp.merge(uk, on='key').drop(columns=['key', 'IDu_hash'])[['IDu', 't', 'q']]
    
    # Restrict to user's active quarter range
    q_bounds = dff.groupby('IDu')['q'].agg(q_min='min', q_max='max').reset_index()
    uq = uq.merge(q_bounds, on='IDu', how='left')
    uq = uq[(uq['q'] >= uq['q_min']) & (uq['q'] <= uq['q_max'])]
    uq = uq.drop(columns=['q_min', 'q_max']).sort_values(['IDu', 't']).reset_index(drop=True)
    
    # Compute quarterly AI averages (AIav)
    g = dff.groupby(['IDu', 'q'])['AIf'].agg(
        AIav='mean', 
        Nav=lambda s: s.notna().sum()
    ).reset_index()
    g.loc[g['Nav'] < MIN_FUNCTIONS_PER_QUARTER, 'AIav'] = np.nan
    uq = uq.merge(g[['IDu', 'q', 'AIav']], on=['IDu', 'q'], how='left')
    
    # Interpolate MA columns to quarter midpoints
    print("Interpolating AI moving averages to quarter midpoints...")
    ma_quarter = []
    
    for idu, gfun in dff.groupby('IDu', sort=True):
        gf = gfun.copy()
        gf['rank_in_day'] = gf.groupby('t').cumcount()
        gf['tnew'] = gf['t'] + pd.to_timedelta(gf['rank_in_day'], unit='ms')
        gf['is_fun'] = True

        gu = uq[uq['IDu'] == idu][['t', 'q']].copy()
        if gu.empty:
            continue
        gu['tnew'] = gu['t']
        gu['is_fun'] = False

        ai_cols = [f'AIma{w}' for w in MA_WINDOWS]
        grid = pd.concat(
            [gf[['t', 'tnew', 'is_fun'] + ai_cols], gu[['t', 'tnew', 'is_fun']]],
            ignore_index=True, sort=False
        ).sort_values('tnew').set_index('tnew')

        # Interpolate each MA series
        for w in MA_WINDOWS:
            col = f'AIma{w}'
            if col in grid.columns:
                grid[f'{col}_i'] = grid[[col]].interpolate(method='time', limit_area='inside')

        # Apply 184-day gap mask
        grid['t_fun'] = grid['t'].where(grid['is_fun'])
        grid['prev_fun'] = grid['t_fun'].ffill()
        grid['next_fun'] = grid['t_fun'].bfill()
        span_days = (grid['next_fun'] - grid['prev_fun']).dt.days
        gap_mask = (span_days > 184) | span_days.isna()
        
        for w in MA_WINDOWS:
            ci = f'AIma{w}_i'
            if ci in grid.columns:
                grid.loc[gap_mask, ci] = np.nan

        out = grid.reset_index()
        out = out[out['is_fun'] == False].copy()
        out = out.merge(gu[['t', 'tnew', 'q']], on=['t', 'tnew'], how='left')

        keep = ['q'] + [c for c in out.columns if c.endswith('_i')]
        out = out[keep].copy()
        out['IDu'] = idu
        ma_quarter.append(out)

    if ma_quarter:
        ma_q = pd.concat(ma_quarter, ignore_index=True)
        uq = uq.merge(ma_q, on=['IDu', 'q'], how='left')
    
    # Keep only rows with some AI information
    ai_cols = ['AIav'] + [f'AIma{w}_i' for w in MA_WINDOWS]
    keep_mask = uq[ai_cols].notna().any(axis=1)
    ai_shares = uq.loc[keep_mask, ['IDu', 'q', 't'] + ai_cols].copy()
    ai_shares = ai_shares.sort_values(['IDu', 'q']).reset_index(drop=True)
    
    print(f"  Created AI shares for {ai_shares['IDu'].nunique()} users")
    return users[['IDu_hash', 'IDu']], ai_shares


def process_commits(user_map: pd.DataFrame):
    """
    Process commit data to create user-quarter outcome variables.
    
    Args:
        user_map: DataFrame with IDu_hash to IDu mapping
        
    Returns:
        DataFrame: User-quarter commit and library counts
    """
    print("Processing commits from panel_with_libs_coarse.csv...")
    
    df = pd.read_csv(IN_DIR / "panel_with_libs_coarse.csv")
    
    # Column renaming - only keep what's needed for regressions
    rename_map = {
        "user_hashed": "IDu_hash",
        "n_commits": "Cupq",
        "n_commits_multi_files": "Cupq_mfiles",
        "n_commits_with_import": "Cupq_wimprt",
        
        # Library counts used in regressions
        "unique_import_lists": "libC_all",
        "unique_import_entries": "libE_all",
        "new_unique_import_lists": "libC_new_u",
        "new_unique_import_entries": "libE_new_u",
        
        # Library combos
        "unique_commit_combos_simone": "libQC_all",
        "new_unique_commit_combos_simone": "libQC_new_u",
        
        # Top-k library combos
        "unique_import_library_combos_commit_topk": "libkQC_all",
        "new_unique_import_library_combos_commit_topk": "libkQC_new_u",
        
        # Community combos (fine)
        "unique_import_community_combos_fine": "libLQ_all",
        "new_unique_import_community_combos_fine": "libLQ_new_u",
        
        # Additional library vars for log transforms
        "unique_topk_libs": "libkE_all",
        "new_unique_topk_libs": "libkE_new_u",
        "unique_import_communities_coarse": "libSE_all",
        "new_unique_import_communities_coarse": "libSE_new_u",
    }
    
    # Only rename columns that exist
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    
    # Parse quarter
    y = pd.to_numeric(df["year_quarter"].str.slice(0, 4), errors='coerce')
    qtr = pd.to_numeric(df["year_quarter"].str[-1:], errors='coerce')
    df["q"] = (y * 10 + qtr).astype("Int64")
    
    # Map to numeric user IDs
    df = df.merge(user_map, on="IDu_hash", how="inner")
    
    # Bot filter: drop users with >10k total commits or >2k in any quarter
    print("  Applying bot filter...")
    tmp = df[['IDu', 'q', 'Cupq']].copy()
    tmp['Cupq'] = pd.to_numeric(tmp['Cupq'], errors='coerce').fillna(0)
    upq = tmp.groupby(['IDu', 'q'], as_index=False)['Cupq'].sum()
    
    tot = upq.groupby('IDu')['Cupq'].sum()
    mx = upq.groupby('IDu')['Cupq'].max()
    bots = set(tot[tot > 10000].index) | set(mx[mx > 2000].index)
    
    if bots:
        df = df[~df['IDu'].isin(bots)].copy()
        print(f"  Dropped {len(bots)} bot users")
    
    # Aggregate to user-quarter level
    sum_cols = [c for c in df.columns if c.startswith('Cupq') or c.startswith('lib')]
    sum_cols = [c for c in sum_cols if c in df.columns]
    
    agg_map = {c: 'sum' for c in sum_cols}
    agg_map['IDu_hash'] = 'first'
    
    uq = df.groupby(['IDu', 'q'], as_index=False).agg(agg_map)
    
    # Rename Cupq -> Cuq at user-quarter level
    rename_uq = {c: c.replace('Cupq', 'Cuq') for c in uq.columns if c.startswith('Cupq')}
    uq = uq.rename(columns=rename_uq)
    
    print(f"  Created commit panel with {uq['IDu'].nunique()} users, {len(uq)} obs")
    return uq


def process_demographics(user_map: pd.DataFrame):
    """
    Process user demographics (gender, experience start year).
    
    Args:
        user_map: DataFrame with IDu_hash to IDu mapping
        
    Returns:
        DataFrame: User-level demographics
    """
    print("Processing demographics from user_gender_experience.csv...")
    
    df = pd.read_csv(IN_DIR / "user_gender_experience.csv")
    df = df[["user_hashed", "gender", "year_started"]].rename(columns={
        "user_hashed": "IDu_hash",
        "year_started": "yr0_GH"
    })
    
    # Take first gender, min start year per user
    g = df.groupby("IDu_hash", as_index=False).agg(
        yr0_GH=("yr0_GH", "min"),
        gender=("gender", "first"),
    )
    
    # Map to numeric IDs
    g = g.merge(user_map, on="IDu_hash", how="inner")
    g = g[['IDu', 'gender', 'yr0_GH']]
    
    print(f"  Loaded demographics for {len(g)} users")
    return g


def rectangularize_panel(commits_df: pd.DataFrame):
    """
    Fill in missing quarters for each user (within their active range).
    
    Args:
        commits_df: User-quarter commit data
        
    Returns:
        DataFrame: Rectangularized panel
    """
    print("Rectangularizing panel...")
    
    mins = commits_df.groupby("IDu")["q"].min()
    maxs = commits_df.groupby("IDu")["q"].max()
    
    frames = []
    for idu in commits_df["IDu"].unique():
        qmin, qmax = int(mins.loc[idu]), int(maxs.loc[idu])
        years = range(qmin // 10, qmax // 10 + 1)
        qs = [y * 10 + qtr for y in years for qtr in [1, 2, 3, 4] 
              if qmin <= y * 10 + qtr <= qmax]
        frames.append(pd.DataFrame({"IDu": idu, "q": qs}))
    
    full = pd.concat(frames, ignore_index=True)
    merged = full.merge(commits_df, on=["IDu", "q"], how="left")
    
    # Fill missing counts with 0
    count_cols = [c for c in merged.columns if c.startswith('Cuq') or c.startswith('lib')]
    for col in count_cols:
        merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0)
    
    return merged


def create_derived_variables(df: pd.DataFrame):
    """
    Create log transforms, forward-fills, and lags needed for regressions.
    
    Args:
        df: Panel DataFrame with AI and commit variables
        
    Returns:
        DataFrame: Panel with all derived variables
    """
    print("Creating derived variables...")
    
    df = df.sort_values(['IDu', 'q']).reset_index(drop=True)
    
    # Log transforms for count variables
    count_cols = [c for c in df.columns if c.startswith('Cuq') or c.startswith('lib')]
    for col in count_cols:
        df[f'log_{col}'] = np.log(pd.to_numeric(df[col], errors='coerce').fillna(0) + 1)
    
    # Forward-fill AI variables (limit=2 quarters)
    ai_vars = ['AIav'] + [f'AIma{w}_i' for w in MA_WINDOWS]
    
    for col in ai_vars:
        if col in df.columns:
            # Forward-fill
            df[f'{col}F'] = df.groupby('IDu', sort=False)[col].ffill(limit=2)
            # Create lags of forward-filled version
            df[f'L_{col}F'] = df.groupby('IDu', sort=False)[f'{col}F'].shift(1)
            df[f'L2_{col}F'] = df.groupby('IDu', sort=False)[f'{col}F'].shift(2)
            df[f'L3_{col}F'] = df.groupby('IDu', sort=False)[f'{col}F'].shift(3)
    
    return df


def main():
    """Main processing pipeline."""
    print("=" * 60)
    print("Data Processing Pipeline")
    print("=" * 60)
    
    # Step 1: Process AI exposure data
    user_map, ai_shares = process_ai_shares()
    
    # Step 2: Process commit/library data
    commits = process_commits(user_map)
    
    # Step 3: Process demographics
    demographics = process_demographics(user_map)
    
    # Step 4: Rectangularize panel
    panel = rectangularize_panel(commits)
    
    # Step 5: Merge AI shares
    panel = panel.merge(ai_shares.drop(columns=['t'], errors='ignore'), 
                        on=['IDu', 'q'], how='left')
    
    # Step 6: Merge demographics
    panel = panel.merge(demographics, on='IDu', how='left')
    
    # Step 7: Compute experience
    panel['experience'] = (
        np.floor(pd.to_numeric(panel['q'], errors='coerce') / 10).astype('Int64') 
        - pd.to_numeric(panel['yr0_GH'], errors='coerce').astype('Int64')
    )
    
    # Step 8: Create derived variables (logs, forward-fills, lags)
    panel = create_derived_variables(panel)
    
    # Step 9: Save output
    output_path = OUT_DIR / "panel_uq.parquet"
    panel.to_parquet(output_path, index=False)
    
    print("=" * 60)
    print(f"Output written to: {output_path}")
    print(f"  Users: {panel['IDu'].nunique()}")
    print(f"  Observations: {len(panel)}")
    print(f"  Quarters: {panel['q'].min()} to {panel['q'].max()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
