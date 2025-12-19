# AI in Code — Reproduction Files
Code for reproducing figures and statistics for **“Who is using AI to code? Global diffusion and impact of generative AI”** (https://arxiv.org/abs/2506.08945).

## Data & environment
- Primary inputs live under `./final_data/` (not tracked here); some scripts expect `inputs/` and write to `outputs/`.
- Python 3.10+ recommended. Key packages: `pandas`, `numpy`, `matplotlib`, `pyfixest`, `marginaleffects`, `pyarrow`, `duckdb`, `networkx`, `tqdm`, `scipy`, `statsmodels`, `seaborn`, `pyreadstat`.

## Panel construction
- `make_panel_data.py` — CLI script that reads `inputs/functions.csv`, `inputs/panel_with_libs_coarse.csv`, and `inputs/user_gender_experience.csv`, computes AI exposure moving averages, commit/library counts, demographics, and writes `outputs/panel_uq.parquet` with forward-filled AI lags and experience measures.


## Statistical scripts
- `country_comparison_ttest.py` — Welch t-tests of AI function shares across countries by year. Inputs: `inputs/country_functions.csv`, `inputs/functions.csv`. Outputs: CSV stats and LaTeX p-value matrices in `outputs/`.
- `verbosity_fp_analysis.py` — Feature engineering and analysis helpers for verbosity-driven false positives: token/structure features, composite verbosity indices, Spearman correlations, FP-by-decile tables, and logistic regressions with odds ratios.

## Figure notebooks
- `fig2_diffusion.ipynb` — Figure 2 (Diffusion): reads `final_data/raw_data_encrypted_final.csv.zip`, computes quarterly mean AI shares, and plots smoothed country-level adoption using `final_data/full_countries.csv`.
- `fig3_gender_experience.ipynb` — Figure 3 (Gender & Experience): filters post-2024Q1 users, runs user-level gender regressions, and estimates experience effects by year using `final_data/raw_data_encrypted_final.csv.zip` and `final_data/AI_010_data_uq.parquet`.
- `fig3_c_d.ipynb` — Figure 3c/d regressions: consumes `outputs/panel_uq.parquet`, runs user and quarter fixed-effects models (commits and library metrics), interaction models by experience, placebo pre-2022 runs, moving-average extrapolations, and exports coefficient plots and LaTeX tables.
- `fig_s2.ipynb` — Supplementary S2 (Classifier diagnostics): plots saved histogram bins from `final_data/hist_data.npz` and training/validation loss curves.
- `fig_s3.ipynb` — Supplementary S3 (Verbosity false positives): loads `final_data/pyfunctions_ai_classified.parquet`, calls `analyze_verbosity_core`, tabulates correlations/deciles, fits baseline vs. verbosity/templatedness logits, and plots FP rates with CIs.
- `fig_s4_s5.ipynb` — Supplementary S4 & S5: iterates over pickled evaluation scores in `final_data/wild_data/` and `final_data/newmodels_data/`, reporting per-model means (WildChat vs. newer model variants).
- `fig_s7/S7.ipynb` — Supplementary S7: rebuilds the library co-occurrence network from `final_data/s7_datasets`, computes significant PMI edges via `library_cooccurrence.py`/`CoLoc_class.py`, thresholds by z-score, and prepares the main connected component for export.
- `fig_s8/step1_S8.ipynb` — Supplementary S8 step 1: processes O*NET task statements/ratings (`final_data/s8_salary_data/`), derives programming-intensity weights, task-level working time allocations, and aligns with BLS wages and employment counts.
- `fig_s8/step2_S17.ipynb` — Supplementary S17/Salary uncertainty: draws Monte Carlo samples over task-time vectors and programming scores with reported standard errors, saving simulated salary impacts and confidence intervals.

## Other artifacts
- `reg_stargazer.tex` — Stored LaTeX regression output referenced in the supplementary material.
