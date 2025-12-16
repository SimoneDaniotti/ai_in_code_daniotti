
"""
verbosity_fp_analysis.py (minimal)

Computes:
1) Spearman correlations between FP (humans) and: individual features, VS, VC, templatedness
2) FP rate by verbosity decile (VS, VC)
3) Logistic regressions: baseline (tokens, complexity, experience), +VS, +VS+templatedness+interaction

No auto-report generation. Import this in a notebook and build your own tables.
"""

import ast
import io
import math
import textwrap
import tokenize
from collections import Counter
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# ----------------------------
# Feature extraction
# ----------------------------

def _safe_parse(code: str):
    try:
        return ast.parse(code)
    except SyntaxError:
        try:
            return ast.parse(textwrap.dedent(code))
        except Exception:
            return None

def _first_function_node(tree):
    if tree is None:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None

def _leading_spaces(s: str) -> int:
    return len(s) - len(s.lstrip(" \t"))

def _shannon_entropy_normalized(items: List[str]) -> float:
    if not items:
        return 0.0
    counts = Counter(items)
    total = float(len(items))
    probs = [c/total for c in counts.values()]
    H = -sum(p * math.log(p + 1e-12) for p in probs)
    maxH = math.log(len(counts)) if len(counts) > 0 else 1.0
    return float(H / (maxH + 1e-12)) if maxH > 0 else 0.0

def _normalize_token(tok) -> str:
    import token as _token
    t = tok.type
    s = tok.string
    if t == _token.NAME:
        return "ID"
    if t == _token.NUMBER:
        return "NUMBER"
    if t == _token.STRING:
        return "STRING"
    if t == _token.OP:
        return s
    return tokenize.tok_name.get(t, "OTHER")

def _count_branching_nodes(tree) -> int:
    if tree is None:
        return 0
    branches = 0
    branch_types = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.IfExp,
                    ast.BoolOp, ast.comprehension, ast.Match)
    for node in ast.walk(tree):
        if isinstance(node, branch_types):
            if isinstance(node, ast.BoolOp):
                branches += max(0, len(getattr(node, "values", [])) - 1)
            elif isinstance(node, ast.comprehension):
                branches += 1
            else:
                branches += 1
    return branches

def _typehint_ratio(fn_node) -> float:
    if fn_node is None:
        return 0.0
    total = 0
    annotated = 0
    args = fn_node.args
    all_args = []
    all_args.extend(getattr(args, "posonlyargs", []))
    all_args.extend(getattr(args, "args", []))
    if getattr(args, "vararg", None) is not None:
        all_args.append(args.vararg)
    all_args.extend(getattr(args, "kwonlyargs", []))
    if getattr(args, "kwarg", None) is not None:
        all_args.append(args.kwarg)
    for a in all_args:
        total += 1
        if getattr(a, "annotation", None) is not None:
            annotated += 1
    if total == 0:
        return 0.0
    return float(annotated) / float(total)

def _avg_identifier_length(tree) -> float:
    if tree is None:
        return 0.0
    names = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
    if not names:
        return 0.0
    cleaned = [n.strip("_") for n in names if n]
    lengths = [len(n) for n in cleaned if n]
    return float(np.mean(lengths)) if lengths else 0.0

def extract_python_function_features(code: str) -> Dict[str, Any]:
    if not isinstance(code, str):
        return {}

    code_str = code
    lines = code_str.splitlines()
    n_lines = len(lines)
    n_chars = len(code_str)
    blank_lines = sum(1 for l in lines if l.strip() == "")
    comment_lines = sum(1 for l in lines if "#" in l and l.strip().startswith("#"))
    avg_line_len = (n_chars / n_lines) if n_lines > 0 else 0.0

    indents = [(_leading_spaces(l.expandtabs(4))) for l in lines if l.strip() != ""]
    indent_mean = float(np.mean(indents)) if indents else 0.0
    indent_std = float(np.std(indents)) if indents else 0.0
    indent_max = max(indents) if indents else 0

    import token as _token
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code_str).readline))
    except Exception:
        toks = []

    STRUCT = {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
              tokenize.ENCODING, tokenize.ENDMARKER}
    effective_tokens = [t for t in toks if t.type not in STRUCT]
    n_tokens = len(effective_tokens)

    n_comment_tokens = sum(1 for t in toks if t.type == tokenize.COMMENT)
    n_name_tokens = sum(1 for t in toks if t.type == _token.NAME)
    n_string_tokens = sum(1 for t in toks if t.type == _token.STRING)
    n_number_tokens = sum(1 for t in toks if t.type == _token.NUMBER)
    n_op_tokens = sum(1 for t in toks if t.type == _token.OP)

    norm_tokens = [_normalize_token(t) for t in effective_tokens]
    unique_token_ratio = (len(set(norm_tokens)) / n_tokens) if n_tokens > 0 else 0.0
    token_entropy_norm = _shannon_entropy_normalized(norm_tokens)
    templatedness = 1.0 - token_entropy_norm

    tree = _safe_parse(code_str)
    fn = _first_function_node(tree)

    docstr_len = 0
    has_docstring = 0
    if fn is not None:
        ds = ast.get_docstring(fn)
        if isinstance(ds, str):
            has_docstring = 1
            docstr_len = len(ds)

    ast_nodes = sum(1 for _ in ast.walk(tree)) if tree is not None else 0
    calls = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call)) if tree is not None else 0
    literals = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Constant)) if tree is not None else 0

    binops = sum(1 for n in ast.walk(tree) if isinstance(n, ast.BinOp)) if tree is not None else 0
    unops = sum(1 for n in ast.walk(tree) if isinstance(n, ast.UnaryOp)) if tree is not None else 0
    boolops = sum(1 for n in ast.walk(tree) if isinstance(n, ast.BoolOp)) if tree is not None else 0
    compares = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Compare)) if tree is not None else 0
    operators_ast = binops + unops + boolops + compares

    branches = _count_branching_nodes(tree)
    complexity = 1 + branches

    typehint_ratio = _typehint_ratio(fn)
    avg_id_len = _avg_identifier_length(tree)

    comment_ratio = (comment_lines / n_lines) if n_lines > 0 else 0.0
    blank_ratio = (blank_lines / n_lines) if n_lines > 0 else 0.0
    op_token_ratio = (n_op_tokens / n_tokens) if n_tokens > 0 else 0.0
    string_token_ratio = (n_string_tokens / n_tokens) if n_tokens > 0 else 0.0

    raw = dict(
        n_lines=n_lines,
        n_chars=n_chars,
        n_tokens=n_tokens,
        avg_line_len=avg_line_len,
        blank_ratio=blank_ratio,
        comment_ratio=comment_ratio,
        has_docstring=has_docstring,
        docstring_len=docstr_len,
        indent_mean=indent_mean,
        indent_std=indent_std,
        indent_max=indent_max,
        ast_nodes=ast_nodes,
        calls=calls,
        literals=literals,
        operators_ast=operators_ast,
        branches=branches,
        complexity=complexity,
        typehint_ratio=typehint_ratio,
        avg_identifier_len=avg_id_len,
        unique_token_ratio=unique_token_ratio,
        token_entropy_norm=token_entropy_norm,
        templatedness=templatedness,
        string_token_ratio=string_token_ratio,
        op_token_ratio=op_token_ratio,
        comment_token_ratio=(n_comment_tokens / (n_tokens + 1e-9)),
    )
    return raw

def _z(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)

def build_features_df(df: pd.DataFrame) -> pd.DataFrame:
    feats = df["modified_blocks"].apply(extract_python_function_features).apply(pd.Series)
    # needed fields are present from extract_*; no composite verbosity_index here
    return feats

def add_core_verbosity_indices(features_df: pd.DataFrame) -> pd.DataFrame:
    req = ["n_tokens","avg_line_len","blank_ratio","comment_ratio","docstring_len","complexity","templatedness"]
    missing = [c for c in req if c not in features_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    df = features_df.copy()
    vs_components = ["avg_line_len","blank_ratio","comment_ratio","docstring_len"]
    df["verbosity_style_index"] = _z(df[vs_components]).mean(axis=1)
    vc_components = ["n_tokens"] + vs_components
    df["verbosity_core_index"] = _z(df[vc_components]).mean(axis=1)
    return df

# ----------------------------
# Analyses (only the 3 requested)
# ----------------------------

def _logit_fit_irls(X, y, max_iter=100, tol=1e-6):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        z = X @ beta
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        W = p * (1 - p) + 1e-12
        XTW = X.T * W
        H = XTW @ X
        g = X.T @ (y - p)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ g
        beta_new = beta + step
        if np.max(np.abs(step)) < tol:
            beta = beta_new
            break
        beta = beta_new
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    z = X @ beta
    p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    eps = 1e-12
    ll = float(np.sum(y*np.log(p+eps) + (1-y)*np.log(1-p+eps)))
    return {"coef": beta, "se": se, "ll": ll, "pred_proba": p}

def _auc(y_true, y_score):
    order = np.argsort(y_score)
    y = y_true[order]
    n1 = np.sum(y == 1); n0 = np.sum(y == 0)
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = np.arange(1, len(y)+1)
    R1 = np.sum(ranks[y == 1])
    return float((R1 - n1*(n1+1)/2) / (n1*n0))

def analyze_verbosity_core(
    df: pd.DataFrame,
    ai_threshold: float = 0.5,
    prediction_is_ai_prob: bool = True,
    individual_features: list = None
):
    """
    Returns a dict with:
      - features: engineered per-row features with VS/VC, labels, FP flag (humans)
      - corr_table: Spearman correlations of FP with individual features + VS/VC/templatedness (humans)
      - deciles_VC: FP rate by VC decile (humans)
      - deciles_VS: FP rate by VS decile (humans)
      - models: dict with 'baseline', 'plus_VS', 'full' (coeffs, AIC, AUC)
    """
    if individual_features is None:
        individual_features = ['avg_line_len','blank_ratio','comment_ratio','docstring_len','n_tokens']

    # Build features
    feats = build_features_df(df)
    feats["user_experience"] = df["user_experience"].astype(float)
    feats["true_label"] = df["true_label"].str.lower()
    ai_score = df["prediction"].astype(float) if prediction_is_ai_prob else (1.0 - df["prediction"].astype(float))
    feats["prediction_score_ai"] = ai_score
    feats["predicted_label"] = np.where(ai_score >= ai_threshold, "ai", "human")
    feats = add_core_verbosity_indices(feats)

    # Humans + FP
    humans = feats[feats["true_label"] == "human"].copy()
    humans["FP"] = (humans["predicted_label"] == "ai").astype(int)

    # 1) Correlations with FP
    rows = []
    for f in individual_features:
        if f in humans.columns:
            rho = humans["FP"].corr(humans[f], method="spearman")
            rows.append({"variable": f, "spearman_rho": float(rho), "n": int(humans[f].notna().sum())})
    # composites + templatedness
    for f in ["verbosity_style_index","verbosity_core_index","templatedness"]:
        rho = humans["FP"].corr(humans[f], method="spearman")
        rows.append({"variable": f, "spearman_rho": float(rho), "n": int(humans[f].notna().sum())})
    corr_table = pd.DataFrame(rows).reset_index(drop=True)

    # 2) FP by decile for VC/VS
    def _deciles(series):
        q = pd.qcut(series, q=10, labels=False, duplicates="drop")
        out = humans.groupby(q)["FP"].agg(["mean","count"]).rename(columns={"mean":"fp_rate"}).reset_index(names="decile")
        return out.sort_values("decile").reset_index(drop=True)
    deciles_VC = _deciles(humans["verbosity_core_index"])
    deciles_VS = _deciles(humans["verbosity_style_index"])

    # 3) Logistic regressions
    def _pack(fit, names):
        beta = fit["coef"]; se = fit["se"]; OR = np.exp(beta)
        return pd.DataFrame([{"term": n, "coef": float(b), "se": float(s), "odds_ratio": float(o)}
                             for n, b, s, o in zip(names, beta, se, OR)])

    # Prepare standardized covariates
    Z = lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-12)
    humans["z_tokens"] = Z(humans["n_tokens"])
    humans["z_complexity"] = Z(humans["complexity"])
    humans["z_experience"] = Z(humans["user_experience"])
    humans["z_VS"] = Z(humans["verbosity_style_index"])
    humans["z_templated"] = Z(humans["templatedness"])
    inter = (humans["z_VS"] * humans["z_templated"]).values

    Y = humans["FP"].values.astype(float)

    X_base = np.column_stack([np.ones(len(humans)), humans["z_tokens"].values, humans["z_complexity"].values, humans["z_experience"].values])
    fit_base = _logit_fit_irls(X_base, Y)
    AIC_base = 2*X_base.shape[1] - 2*fit_base["ll"]
    AUC_base = _auc(Y, fit_base["pred_proba"])
    names_base = ["Intercept","z_tokens","z_complexity","z_experience"]
    coeffs_base = _pack(fit_base, names_base)

    X_vs = np.column_stack([X_base, humans["z_VS"].values])
    fit_vs = _logit_fit_irls(X_vs, Y)
    AIC_vs = 2*X_vs.shape[1] - 2*fit_vs["ll"]
    AUC_vs = _auc(Y, fit_vs["pred_proba"])
    names_vs = names_base + ["z_VS"]
    coeffs_vs = _pack(fit_vs, names_vs)

    X_full = np.column_stack([X_vs, humans["z_templated"].values, inter])
    fit_full = _logit_fit_irls(X_full, Y)
    AIC_full = 2*X_full.shape[1] - 2*fit_full["ll"]
    AUC_full = _auc(Y, fit_full["pred_proba"])
    names_full = names_vs + ["z_templated","z_VS:z_templated"]
    coeffs_full = _pack(fit_full, names_full)

    models = {
        "baseline": {"coeffs": coeffs_base, "AIC": AIC_base, "AUC": AUC_base},
        "plus_VS": {"coeffs": coeffs_vs, "AIC": AIC_vs, "AUC": AUC_vs},
        "full": {"coeffs": coeffs_full, "AIC": AIC_full, "AUC": AUC_full},
    }

    return {
        "features": feats,
        "corr_table": corr_table,
        "deciles_VC": deciles_VC,
        "deciles_VS": deciles_VS,
        "models": models,
    }

__all__ = [
    "extract_python_function_features",
    "build_features_df",
    "add_core_verbosity_indices",
    "analyze_verbosity_core",
]
