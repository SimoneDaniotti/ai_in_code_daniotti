# AI in Code — Reproduction Files
Code for reproducing the figures in the paper **“Who is using AI to code? Global diffusion and impact of generative AI”** (https://arxiv.org/abs/2506.08945).

## Data
- All inputs are expected under `./final_data/` (not tracked here). 

## Environment
- Python 3.10+ recommended. Typical dependencies: `pandas`, `numpy`, `matplotlib`, `pyarrow`, `pyfixest`, `duckdb`, `networkx`, `tqdm`, `scipy`, `pyreadstat`.


## Figure-by-figure notebooks
- `fig2_diffusion.ipynb` — Recreates **Figure 2 (Diffusion)**: reads the raw dataset, constructs quarterly adoption shares, and plots global and country-level diffusion curves.
- `fig3_gender_experience.ipynb` — Recreates **Figure 3 (Gender & Experience heterogeneity)**: filters the main panel, runs user-level regressions (via `pyfixest`), and plots adoption differences by gender and experience over time.
- `fig_s2.ipynb` — Recreates **Supplementary Figure S2 (Classifier diagnostics)**: plots histogram comparisons of model scores and training/validation loss trajectories.
- `fig_s3.ipynb` — Recreates **Supplementary Figure S3 (Verbosity-driven false positives)**: loads labeled code snippets, computes verbosity and templating features (`verbosity_fp_analysis.py`), and plots false-positive rates by decile with confidence intervals.
- `fig_s4_s5.ipynb` — Recreates **Supplementary Figures S4 & S5 (Model and WildChat comparisons)**: loads pickled evaluation outputs for WildChat and newer model variants, aggregates scores, and plots performance differences.
- `fig_s7/S7.ipynb` — Recreates **Supplementary Figure S7 (Library co-occurrence network)**: uses the accompanying `CoLoc_class.py` and `library_cooccurrence.py` utilities to build significant PMI edges between libraries and render the community-level network from `s7_datasets`.
- `fig_s8/step1_S8.ipynb` — First step for **Supplementary Figure S8**: loads task embeddings, LLM prompt ratings, and BLS working-hour data to compute programming-intensity weights and working-time adjustments.
- `fig_s8/step2_S17.ipynb` — Second step for **Supplementary Figure S8 (and S17 salary uncertainty)**: runs Monte Carlo sampling over task-level programming scores and salary distributions to derive confidence intervals for income impacts.

## Supporting code
- `generate_user_quarter_project_panel_refactored.ipynb` constructs the user × project × quarter panel used across analyses (commit aggregation, novelty metrics, and clustering enrichments). Point it to the raw CSVs in `final_data/` to regenerate the panel.
- `verbosity_fp_analysis.py` holds the feature engineering and plotting helpers used in `fig_s3.ipynb`.
- `reg_stargazer.tex` contains LaTeX output for regression tables referenced in the supplementary material.
