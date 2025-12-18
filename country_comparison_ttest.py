#!/usr/bin/env python3
"""country_comparison_ttest.py

Reads:
  inputs/country_functions.csv
  inputs/functions.csv

Writes:
  outputs/AI_country_comparison_stats.csv
  outputs/AI_country_comparison_ttests.csv
  outputs/AI_country_comparison_pvalues.tex

The script:
  1) loads function-level observations,
  2) computes country-year summary statistics,
  3) runs pairwise Welch t-tests by year,
  4) exports a LaTeX p-value matrix (one block per year).
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy import stats


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _standardize_ai_share_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the AI share column is named 'AIf'."""
    if "AIf" in df.columns:
        return df
    if "ai_share" in df.columns:
        return df.rename(columns={"ai_share": "AIf"})
    raise KeyError("Expected an AI share column named 'AIf' or 'ai_share'.")


def load_data() -> pd.DataFrame:
    """Load and clean inputs into columns: country, year, AIf."""
    # Non-US
    cf = pd.read_csv(INPUT_DIR / "country_functions.csv", index_col=0)
    cf = _standardize_ai_share_column(cf)

    cf["year"] = pd.to_numeric(cf.get("year"), errors="coerce")
    cf = cf[cf["year"].notna()].copy()
    cf["year"] = np.floor(cf["year"]).astype(int)
    cf = cf[cf["year"] < 2025].copy()

    # US
    us = pd.read_csv(INPUT_DIR / "functions.csv", index_col=0)
    us = _standardize_ai_share_column(us)
    us["country"] = "unitedstates"

    if "year" not in us.columns:
        if "quarter" not in us.columns:
            raise KeyError("functions.csv must contain 'year' or 'quarter'.")
        us["year"] = us["quarter"].astype("string").str.slice(0, 4)

    us["year"] = pd.to_numeric(us["year"], errors="coerce")
    us = us[us["year"].notna()].copy()
    us["year"] = us["year"].astype(int)
    us = us[us["year"] < 2025].copy()

    df = pd.concat([cf, us], ignore_index=True, sort=False)

    for col in ("country", "year", "AIf"):
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    df["AIf"] = pd.to_numeric(df["AIf"], errors="coerce")
    df = df.dropna(subset=["country", "year", "AIf"]).copy()

    df["country"] = df["country"].astype(str)
    df["year"] = df["year"].astype(int)

    return df[["country", "year", "AIf"]]


def country_year_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute country-year mean, sd, n, sem and 95% CI for AIf."""
    g = df.groupby(["country", "year"], as_index=False)["AIf"]
    out = g.agg(AIav_cy="mean", AIsd_cy="std", AIn_cy="count")

    out["AIsem_cy"] = out["AIsd_cy"] / np.sqrt(out["AIn_cy"])
    out["AIlb_cy"] = out["AIav_cy"] - 1.96 * out["AIsem_cy"]
    out["AIub_cy"] = out["AIav_cy"] + 1.96 * out["AIsem_cy"]

    return out


def welch_t_ci(a: np.ndarray, b: np.ndarray, alpha: float = 0.05):
    """Welch two-sample t-test with CI for mean(a) - mean(b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    n1, n2 = a.size, b.size
    m1, m2 = float(a.mean()), float(b.mean())
    v1 = float(a.var(ddof=1)) if n1 > 1 else 0.0
    v2 = float(b.var(ddof=1)) if n2 > 1 else 0.0

    diff = m1 - m2
    se_sq = (v1 / n1 if n1 > 0 else 0.0) + (v2 / n2 if n2 > 0 else 0.0)
    if se_sq <= 0:
        return diff, float("nan"), float("nan"), float("nan")

    se = math.sqrt(se_sq)

    # Welch–Satterthwaite df
    df_num = se_sq**2
    df_den = 0.0
    if n1 > 1 and v1 > 0:
        df_den += (v1**2) / (n1**2 * (n1 - 1))
    if n2 > 1 and v2 > 0:
        df_den += (v2**2) / (n2**2 * (n2 - 1))
    if df_den == 0:
        return diff, float("nan"), float("nan"), float("nan")

    dof = df_num / df_den

    t_stat = diff / se
    p = 2 * stats.t.sf(abs(t_stat), dof)

    t_crit = stats.t.ppf(1 - alpha / 2, dof)
    lb = diff - t_crit * se
    ub = diff + t_crit * se

    return diff, lb, ub, p


def run_ttests(df: pd.DataFrame) -> pd.DataFrame:
    """Welch t-tests of AIf for every ordered country pair, within each year."""
    countries = sorted(df["country"].unique().tolist())
    years = sorted(df["year"].unique().tolist())

    rows: list[dict] = []

    for y in years:
        sub_y = df[df["year"] == y]
        if sub_y.empty:
            continue

        for c1 in countries:
            a = sub_y.loc[sub_y["country"] == c1, "AIf"].to_numpy()
            if a.size == 0:
                continue

            for c2 in countries:
                if c1 == c2:
                    continue

                b = sub_y.loc[sub_y["country"] == c2, "AIf"].to_numpy()
                if b.size == 0:
                    continue

                mu_diff, lb_diff, ub_diff, p = welch_t_ci(a, b)

                rows.append(
                    {
                        "country1": c1,
                        "country2": c2,
                        "year": int(y),
                        "n1": int(a.size),
                        "n2": int(b.size),
                        "mean1": float(np.mean(a)),
                        "mean2": float(np.mean(b)),
                        "mu_diff": float(mu_diff),
                        "lb_diff": float(lb_diff),
                        "ub_diff": float(ub_diff),
                        "p": float(p),
                    }
                )

    return pd.DataFrame(rows)


def _format_p_cell(mu_diff: float, p: float) -> str:
    if not np.isfinite(p):
        return ""

    p_str = f"{p:.3f}"

    if p < 0.01:
        p_str = f"\\textbf{{{p_str}}}"

    if mu_diff > 0:
        return f"\\textcolor{{green}}{{{p_str}}}"
    if mu_diff < 0:
        return f"\\textcolor{{red}}{{({p_str})}}"

    return p_str


def export_latex_pvalue_tables(ttests_df: pd.DataFrame, path: Path) -> None:
    """One LaTeX table with one block per year."""
    years = sorted(ttests_df["year"].unique().tolist())
    countries = sorted(set(ttests_df["country1"]).union(ttests_df["country2"]))

    lines: list[str] = []
    lines.append("% Auto-generated by country_comparison_ttest.py")
    lines.append("% Requires: \\usepackage{xcolor}")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Pairwise p-values for differences in AI adoption rates}")
    lines.append("\\label{tab:country_pvalues}")
    lines.append("\\scriptsize")

    for y in years:
        sub = ttests_df[ttests_df["year"] == y]

        align = "l" + "c" * len(countries)
        lines.append(f"\\vspace{{0.5em}}% year {y}")
        lines.append(f"\\begin{{tabular}}{{{align}}}")
        lines.append("\\hline")
        lines.append(" & " + " & ".join(countries) + " \\\\")
        lines.append("\\hline")

        for r in countries:
            row_cells = [r]
            for c in countries:
                if r == c:
                    row_cells.append("--")
                    continue

                match = sub[(sub["country1"] == r) & (sub["country2"] == c)]
                if match.empty:
                    row_cells.append("")
                    continue

                mu_diff = float(match["mu_diff"].iloc[0])
                p = float(match["p"].iloc[0])
                row_cells.append(_format_p_cell(mu_diff, p))

            lines.append(" & ".join(row_cells) + " \\\\")

        lines.append("\\hline")
        lines.append("\\end{tabular}\\\\")
        lines.append(f"% End of year {y}")

    lines.append("\\end{table}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_data()

    stats_df = country_year_stats(df)
    stats_df.to_csv(OUTPUT_DIR / "AI_country_comparison_stats.csv", index=False)

    ttests_df = run_ttests(df)
    ttests_df.to_csv(OUTPUT_DIR / "AI_country_comparison_ttests.csv", index=False)

    export_latex_pvalue_tables(ttests_df, OUTPUT_DIR / "AI_country_comparison_pvalues.tex")

    print("Wrote:")
    print(" - outputs/AI_country_comparison_stats.csv")
    print(" - outputs/AI_country_comparison_ttests.csv")
    print(" - outputs/AI_country_comparison_pvalues.tex")


if __name__ == "__main__":
    main()
