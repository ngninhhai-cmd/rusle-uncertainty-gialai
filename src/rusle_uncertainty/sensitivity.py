"""Correlation and variance-based sensitivity analysis."""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def correlation_matrix(factors_df):
    """Pearson correlation matrix."""
    return factors_df.corr(method="pearson")


def correlation_with_target(factors_df, target_col="SER"):
    """Correlation of each factor with SER."""
    results = []
    for col in factors_df.columns:
        if col == target_col:
            continue
        r, p = pearsonr(factors_df[col], factors_df[target_col])
        results.append({
            "Factor": col, "r": r, "R2": r ** 2, "p_value": p,
            "Significant": "Yes" if p < 0.05 else "No",
        })
    return pd.DataFrame(results).sort_values("r", ascending=False)


def sobol_indices(factors_df, n_samples=4096, seed=42):
    """Sobol indices via SALib (independence assumption)."""
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_analyze

    names = list(factors_df.columns)
    bounds = [[float(factors_df[n].min()), float(factors_df[n].max())]
              for n in names]
    problem = {"num_vars": len(names), "names": names, "bounds": bounds}

    param_values = sobol_sample.sample(problem, n_samples, seed=seed)
    Y = param_values.prod(axis=1)
    Si = sobol_analyze.analyze(problem, Y, print_to_console=False, seed=seed)

    return pd.DataFrame({
        "Factor": names, "S1": Si["S1"], "S1_conf": Si["S1_conf"],
        "ST": Si["ST"], "ST_conf": Si["ST_conf"],
    })


def correlated_monte_carlo(factors_df, n_samples=10000, seed=42):
    """Correlated MC preserving covariance (Cholesky)."""
    names = list(factors_df.columns)
    sub = factors_df[names].dropna()
    cov = sub.cov().values
    mean = sub.mean().values

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov += np.eye(len(names)) * 1e-10
        L = np.linalg.cholesky(cov)

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_samples, len(names)))
    samples = Z @ L.T + mean

    var_total = np.var(samples.prod(axis=1))
    contributions = []
    for i in range(len(names)):
        samples_pert = samples.copy()
        samples_pert[:, i] = mean[i]
        V_i = max(var_total - np.var(samples_pert.prod(axis=1)), 0)
        contributions.append(V_i)

    contributions = np.array(contributions)
    contributions_pct = 100 * contributions / contributions.sum()

    df = pd.DataFrame({
        "Factor": names,
        "Contribution_pct": contributions_pct,
    }).sort_values("Contribution_pct", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    return df