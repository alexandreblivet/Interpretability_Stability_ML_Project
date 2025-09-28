#!/usr/bin/env python3
"""
Step 9 - Fairness analysis

This script:
- Loads the project dataset
- Identifies the protected attribute (default: `ethnicity`, with robust auto-detection)
- Chooses analysis unit (continuous score vs. binary decision with threshold)
- Computes common fairness metrics:
  * Statistical Parity Difference (SPD)
  * Disparate Impact (DI)
  * Equal Opportunity (TPR difference)
  * Equalized Odds (TPR and FPR differences)
  * Calibration within groups (reliability by score bins)
- Produces visualizations:
  * Histograms and boxplots of scores by group
  * Per-group confusion matrices
  * Per-group ROC curves

Notes:
- By default, uses the dataset column `Predicted probabilities` as scores `s`.
- If missing, the script can optionally try to compute scores from a provided model,
  but this is disabled by default to avoid feature mismatch. Use `--force-model` to try.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from statsmodels.stats.weightstats import ttest_ind  # For equivalence test
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_PLOTS_DIR = Path(__file__).resolve().parent / "graphs"


def find_dataset_path(cli_path: Optional[str]) -> Path:
    if cli_path:
        return Path(cli_path)
    # Prefer common filenames in the repo
    candidates = [
        DEFAULT_DATA_DIR / "dataproject2025.csv",
        DEFAULT_DATA_DIR / "dataproject2025 (1).csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fallback: first CSV in data/
    for p in DEFAULT_DATA_DIR.glob("*.csv"):
        return p
    raise FileNotFoundError("No CSV dataset found in data/. Provide --data path.")


def autodetect_protected_attribute(df: pd.DataFrame, cli_attr: Optional[str]) -> str:
    if cli_attr and cli_attr in df.columns:
        return cli_attr
    # Heuristic search for ethnicity/race
    lower_cols = {c.lower(): c for c in df.columns}
    candidates = [
        "ethnicity", "race", "ethnic", "ethnic_group", "ethnicgroup",
        "race_desc", "ethnicity_desc",
    ]
    for key in candidates:
        if key in lower_cols:
            return lower_cols[key]
    # If nothing found, try any column that looks categorical with few unique values
    for c in df.columns:
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= 10:
            # avoid target and score columns
            if c not in {"target", "Predictions", "Predicted probabilities"}:
                return c
    raise ValueError("Could not detect a protected attribute. Use --protected to specify one.")


def define_groups(series: pd.Series, positive_group_values: Optional[List[str]] = None) -> pd.Series:
    s = series.astype(str).str.strip()
    # If binary-like, map to majority/minority if possible
    unique_vals = s.dropna().unique().tolist()
    if positive_group_values:
        minority_mask = s.isin(positive_group_values)
        return pd.Series(np.where(minority_mask, "minority", "majority"), index=series.index)
    if len(unique_vals) == 2:
        # Assign alphabetically the first as minority for determinism
        first = sorted(unique_vals)[0]
        return pd.Series(np.where(s == first, "minority", "majority"), index=series.index)
    # Multi-class: keep as-is (for plots) and also provide a grouped view in metrics by comparing minority vs majority as top-2 frequencies
    return s


def ensure_plots_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_output_dirs(*dirs: Path):
    """Ensures output directories are clean by deleting existing files."""
    for d in dirs:
        if d.is_file():
            print(f"Error: Output path {d} is a file. Please specify a directory.")
            continue

        if d.is_dir():  # Directory exists, clear files
            for item in d.iterdir():
                if item.is_file():
                    item.unlink()
        else:  # Directory doesn't exist, create it
            d.mkdir(parents=True, exist_ok=True)


def summarize_group_distribution(groups: pd.Series) -> pd.DataFrame:
    dist = groups.value_counts(dropna=False).rename("count").to_frame()
    dist["fraction"] = dist["count"] / len(groups)
    return dist


def get_scores_and_labels(df: pd.DataFrame, score_col: Optional[str], target_col: str, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Scores
    if score_col is None:
        if "Predicted probabilities" in df.columns:
            score_col = "Predicted probabilities"
        else:
            raise ValueError("Score column not found. Provide --score-col or ensure 'Predicted probabilities' exists.")
    scores = df[score_col].astype(float).to_numpy()
    # Labels
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in dataset.")
    y_true = df[target_col].astype(int).to_numpy()
    # Binary decisions from threshold
    y_hat = (scores >= threshold).astype(int)
    return scores, y_true, y_hat


def compute_group_metrics(y_true: np.ndarray, y_hat: np.ndarray, scores: np.ndarray, groups: pd.Series) -> Dict[str, Union[float, Dict[str, float]]]:
    # Identify majority/minority from frequency among two most common groups
    counts = groups.value_counts()
    top2 = counts.index.tolist()[:2]
    if len(top2) < 2:
        raise ValueError("Need at least 2 groups to compute fairness metrics.")
    g_min, g_maj = top2[-1], top2[0]  # first is majority (most frequent)

    mask_min = (groups == g_min).to_numpy()
    mask_maj = (groups == g_maj).to_numpy()

    # Selection rates
    p1_min = y_hat[mask_min].mean() if mask_min.any() else np.nan
    p1_maj = y_hat[mask_maj].mean() if mask_maj.any() else np.nan

    spd = p1_min - p1_maj
    di = (p1_min / p1_maj) if p1_maj and not np.isnan(p1_maj) else np.nan

    # True Positive Rate (TPR) and False Positive Rate (FPR)
    def tpr(y_t, y_p):
        denom = (y_t == 1).sum()
        return ((y_t == 1) & (y_p == 1)).sum() / denom if denom > 0 else np.nan

    def fpr(y_t, y_p):
        denom = (y_t == 0).sum()
        return ((y_t == 0) & (y_p == 1)).sum() / denom if denom > 0 else np.nan

    tpr_min = tpr(y_true[mask_min], y_hat[mask_min])
    tpr_maj = tpr(y_true[mask_maj], y_hat[mask_maj])
    fpr_min = fpr(y_true[mask_min], y_hat[mask_min])
    fpr_maj = fpr(y_true[mask_maj], y_hat[mask_maj])

    eo = tpr_min - tpr_maj  # Equal Opportunity (TPR diff)
    eodds_tpr = tpr_min - tpr_maj
    eodds_fpr = fpr_min - fpr_maj

    # Calibration within groups (Brier-like by bins)
    calib = {}
    bins = np.linspace(0, 1, 11)
    for g, m in [(g_min, mask_min), (g_maj, mask_maj)]:
        s_g = scores[m]
        y_g = y_true[m]
        bin_ids = np.digitize(s_g, bins, right=True)
        diffs = []
        for b in range(1, len(bins) + 1):
            idx = bin_ids == b
            if idx.sum() >= 20:  # Minimum support per bin for stability
                avg_score = s_g[idx].mean()
                emp_rate = y_g[idx].mean()
                diffs.append(abs(emp_rate - avg_score))
        calib[g] = float(np.mean(diffs)) if diffs else np.nan

    return {
        "group_minority": str(g_min),
        "group_majority": str(g_maj),
        "selection_rate_min": float(p1_min) if p1_min == p1_min else np.nan,
        "selection_rate_maj": float(p1_maj) if p1_maj == p1_maj else np.nan,
        "SPD": float(spd) if spd == spd else np.nan,
        "DI": float(di) if di == di else np.nan,
        "EO_TPR_diff": float(eo) if eo == eo else np.nan,
        "EOdds_TPR_diff": float(eodds_tpr) if eodds_tpr == eodds_tpr else np.nan,
        "EOdds_FPR_diff": float(eodds_fpr) if eodds_fpr == eodds_fpr else np.nan,
        "Calibration_abs_error_by_group": calib,
    }


def perform_equivalence_test(
    scores: np.ndarray,
    groups: pd.Series,
    g_min: str,
    g_maj: str,
    delta: float
) -> Dict[str, Union[str, float]]:
    """Performs a TOST equivalence test for the difference in mean scores."""
    mask_min = (groups == g_min)
    mask_maj = (groups == g_maj)
    
    scores_min = scores[mask_min]
    scores_maj = scores[mask_maj]

    if len(scores_min) < 2 or len(scores_maj) < 2:
        return {
            "result": "Not enough data for test.",
            "p_value": np.nan,
            "mean_diff": np.nan,
            "delta": delta,
        }

    # TOST: Two one-sided t-tests
    # H0_upper: mean_diff >= delta  (we want to reject this) -> alternative='smaller'
    # H0_lower: mean_diff <= -delta (we want to reject this) -> alternative='larger'
    mean_diff = scores_min.mean() - scores_maj.mean()

    # Test 1: Is the difference significantly greater than -delta?
    _, p_value_lower, _ = ttest_ind(scores_min, scores_maj, alternative='larger', value=-delta)
    
    # Test 2: Is the difference significantly less than delta?
    _, p_value_upper, _ = ttest_ind(scores_min, scores_maj, alternative='smaller', value=delta)

    # Equivalence is claimed if both null hypotheses are rejected
    p_value = max(p_value_lower, p_value_upper)
    alpha = 0.05
    result = "Equivalence Accepted" if p_value < alpha else "Equivalence Rejected"

    return {
        "result": result,
        "p_value": float(p_value),
        "mean_diff": float(mean_diff),
        "delta": float(delta),
    }


def encode_image_base64(img_path: Path) -> str:
    """Reads an image and returns a base64 encoded string for HTML embedding."""
    if not img_path.exists():
        return ""
    try:
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded_string}"
    except Exception as e:
        print(f"Error encoding image {img_path}: {e}")
        return ""


def generate_html_report(report_data: Dict, output_dir: Path):
    """Generates a self-contained HTML report from the analysis results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fairness_report.html"

    # Extract data from report
    data_path = report_data.get('data_path', 'N/A')
    protected_col = report_data.get('protected_col', 'N/A')
    group_dist_df = report_data.get('group_distribution')
    metrics = report_data.get('metrics', {})
    plots_dir = report_data.get('plots_dir')
    candidate_results = report_data.get('candidate_results', {})
    tost_results = report_data.get('tost_results', {})
    
    # Prepare data for template
    group_dist_html = group_dist_df.to_html(classes='table table-striped', justify='center') if group_dist_df is not None else "<p>Not available.</p>"
    
    flat_metrics = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    calib_metrics = metrics.get("Calibration_abs_error_by_group", {})
    if isinstance(calib_metrics, dict):
        for g, val in calib_metrics.items():
            flat_metrics[f"Calibration_abs_error__{g}"] = val

    # Encode images
    hist_img = encode_image_base64(plots_dir / "scores_hist_by_group.png") if plots_dir else ""
    box_img = encode_image_base64(plots_dir / "scores_box_by_group.png") if plots_dir else ""
    roc_img = encode_image_base64(plots_dir / "roc_by_group.png") if plots_dir else ""
    fpdp_img = encode_image_base64(plots_dir / "fpdp.png") if plots_dir else ""
    dist_img = encode_image_base64(plots_dir / "group_distribution.png") if plots_dir else ""
    cand_img = encode_image_base64(plots_dir / "candidate_analysis.png") if plots_dir else ""
    
    confusion_imgs = {}
    if group_dist_df is not None and plots_dir:
        for g in group_dist_df.index:
            if pd.isna(g): continue
            g_str = str(g).replace(" ", "_") # Sanitize for filename
            img_path = plots_dir / f"confusion_{g_str}.png"
            if img_path.exists():
                confusion_imgs[g] = encode_image_base64(img_path)

    # Candidate analysis HTML
    cand_html = "<h4>Overall Test</h4>"
    if candidate_results:
        cand_html += f"<p>Chi-squared = {candidate_results.get('overall_chi2', 0):.2f}, p-value = {candidate_results.get('overall_p', 0):.4f}</p>"
        cand_html += f"<p><em>{candidate_results.get('message', '')}</em></p>"
        if candidate_results.get('candidate_variables'):
            cand_html += "<h4>Candidate Variables Found:</h4><ul>"
            for feature, values in candidate_results['candidate_variables'].items():
                cand_html += f"<li><b>{feature}</b>: Fairness restored for values: {', '.join(values)}</li>"
            cand_html += "</ul>"
        if candidate_results.get('candidate_variables_message'):
            cand_html += f"<p>{candidate_results['candidate_variables_message']}</p>"
    else:
        cand_html = "<p>Analysis not performed or failed.</p>"

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fairness Analysis Report</title>
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ font-family: sans-serif; padding: 2em; }}
            h1, h2, h3 {{ color: #333; }}
            .container {{ max-width: 1200px; margin: auto; }}
            .card {{ margin-bottom: 2em; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center mb-4">Fairness Analysis Report</h1>
            
            <div class="card">
                <div class="card-header"><h3>Setup Information</h3></div>
                <div class="card-body">
                    <p><b>Dataset Path:</b> {data_path}</p>
                    <p><b>Protected Attribute:</b> {protected_col}</p>
                </div>
            </div>

            <div class="card">
                <div class="card-header"><h3>Group Distribution</h3></div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-7">{group_dist_html}</div>
                        <div class="col-md-5"><img src="{dist_img}" alt="Group Distribution Plot"></div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header"><h3>Fairness Metrics</h3></div>
                <div class="card-body">
                    <table class="table table-sm">
                        {''.join(f"<tr><th>{k}</th><td>{v:.4f}</td></tr>" if isinstance(v, float) else f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in flat_metrics.items())}
                    </table>
                </div>
            </div>

            <div class="card">
                <div class="card-header"><h3>Fairness Equivalence Test (TOST)</h3></div>
                <div class="card-body">
                    <p>This test checks if the difference in mean scores between the minority and majority groups is practically equivalent (i.e., within a tolerance of &delta;={tost_results.get('delta', 'N/A')}).</p>
                    <table class="table table-sm">
                        <tr><th>Mean Score Difference (minority - majority)</th><td>{tost_results.get('mean_diff', 0):.4f}</td></tr>
                        <tr><th>P-value for Equivalence</th><td>{tost_results.get('p_value', 0):.4f}</td></tr>
                        <tr><th>Result (at &alpha;=0.05)</th><td><b>{tost_results.get('result', 'N/A')}</b></td></tr>
                    </table>
                </div>
            </div>

            <div class="card">
                <div class="card-header"><h3>Fairness Partial Dependence Plot (FPDP)</h3></div>
                <div class="card-body text-center">
                    <p>This plot shows the average predicted score for each protected group.</p>
                    <img src="{fpdp_img}" alt="Fairness Partial Dependence Plot" style="max-width: 700px;">
                </div>
            </div>

            <div class="card">
                <div class="card-header"><h3>Score Distributions</h3></div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6"><img src="{hist_img}" alt="Score Histogram"></div>
                        <div class="col-md-6"><img src="{box_img}" alt="Score Boxplot"></div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header"><h3>ROC Curves by Group</h3></div>
                <div class="card-body text-center">
                    <img src="{roc_img}" alt="ROC by group" style="max-width: 600px;">
                </div>
            </div>
            
            <div class="card">
                <div class="card-header"><h3>Confusion Matrices by Group</h3></div>
                <div class="card-body">
                    <div class="row">
                        {''.join(f'<div class="col-md-4 text-center"><p><b>Group: {g}</b></p><img src="{img_b64}" alt="Confusion matrix for {g}"></div>' for g, img_b64 in confusion_imgs.items())}
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header"><h3>Candidate Variable Analysis</h3></div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">{cand_html}</div>
                        <div class="col-md-6">
                            <p>The chart below shows the maximum fairness p-value achieved when controlling for each variable. Variables crossing the red line (&alpha;=0.05) are considered candidates for explaining bias.</p>
                            <img src="{cand_img}" alt="Candidate Variable Analysis Plot">
                        </div>
                    </div>
                </div>
            </div>

        </div>
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"\nSaved HTML report to: {output_path}")


def plot_group_distribution(groups: pd.Series, outdir: Path) -> None:
    """Generates and saves a bar chart of the group distribution."""
    ensure_plots_dir(outdir)
    plt.figure(figsize=(8, 5))
    order = groups.value_counts().index
    sns.countplot(y=groups, order=order)
    plt.title('Group Distribution', fontweight='bold', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Group', fontsize=12)
    plt.tight_layout()
    plt.savefig(outdir / "group_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_candidate_analysis(results: Dict, outdir: Path) -> None:
    """Generates and saves a bar chart of the candidate variable analysis."""
    p_values_df = pd.DataFrame(
        list(results.get('all_p_values', {}).items()),
        columns=['feature', 'max_p_value']
    ).sort_values('max_p_value', ascending=False)

    if p_values_df.empty:
        return

    plt.figure(figsize=(10, 8))
    sns.barplot(data=p_values_df, x='max_p_value', y='feature', palette='viridis_r')
    plt.axvline(x=results.get('alpha', 0.05), color='r', linestyle='--', label=f"Alpha = {results.get('alpha', 0.05)}")
    plt.title('Candidate Variable Analysis', fontweight='bold', fontsize=16)
    plt.xlabel('Maximum Conditional p-value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "candidate_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_fairness_partial_dependence(df: pd.DataFrame, protected_col: str, score_col: str, outdir: Path) -> None:
    """Generates and saves a Fairness Partial Dependence Plot."""
    ensure_plots_dir(outdir)
    pdp_data = df.groupby(protected_col)[score_col].mean().reset_index()

    plt.figure(figsize=(8, 6))
    sns.barplot(data=pdp_data, x=protected_col, y=score_col, palette="viridis", order=sorted(pdp_data[protected_col].unique()))
    plt.axhline(y=df[score_col].mean(), color='r', linestyle='--', label='Overall Mean Score')
    plt.title('Fairness Partial Dependence Plot', fontweight='bold', fontsize=16)
    plt.xlabel(f'Protected Attribute: {protected_col}', fontsize=12)
    plt.ylabel(f'Average Score ({score_col})', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fpdp.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_distributions(scores: np.ndarray, groups: pd.Series, outdir: Path, title_suffix: str = "") -> None:
    ensure_plots_dir(outdir)
    df_plot = pd.DataFrame({"score": scores, "group": groups.astype(str).values})

    plt.figure(figsize=(12, 6))
    sns.histplot(data=df_plot, x="score", hue="group", bins=30, kde=True, stat="density", common_norm=False, multiple="layer")
    plt.title(f"Score Distributions by Group{title_suffix}", fontweight='bold', fontsize=16)
    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(title="Group")
    plt.tight_layout()
    plt.savefig(outdir / "scores_hist_by_group.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_plot, x="group", y="score", order=sorted(df_plot['group'].unique()))
    plt.title(f"Score Boxplots by Group{title_suffix}", fontweight='bold', fontsize=16)
    plt.xlabel("Group", fontsize=12)
    plt.ylabel("Predicted Probability", fontsize=12)
    plt.tight_layout()
    plt.savefig(outdir / "scores_box_by_group.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_by_group(y_true: np.ndarray, y_hat: np.ndarray, groups: pd.Series, outdir: Path) -> None:
    ensure_plots_dir(outdir)
    df_tmp = pd.DataFrame({"y": y_true, "yhat": y_hat, "g": groups.astype(str).values})
    for g, sub in df_tmp.groupby("g"):
        cm = confusion_matrix(sub["y"], sub["yhat"], labels=[0, 1])
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", cbar=False, 
                    xticklabels=["Predicted 0", "Predicted 1"], yticklabels=["Actual 0", "Actual 1"],
                    annot_kws={"size": 14})
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.title(f"Confusion Matrix - Group: {g}", fontweight='bold', fontsize=16)
        plt.tight_layout()
        sanitized_g = str(g).replace(" ", "_")
        plt.savefig(outdir / f"confusion_{sanitized_g}.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_roc_by_group(y_true: np.ndarray, scores: np.ndarray, groups: pd.Series, outdir: Path) -> None:
    ensure_plots_dir(outdir)
    plt.figure(figsize=(8, 7))
    
    # Use a color cycle from the chosen palette
    palette = sns.color_palette("viridis", n_colors=groups.nunique())
    
    for i, g in enumerate(sorted(groups.astype(str).unique())):
        mask = (groups.astype(str) == g).to_numpy()
        y_g = y_true[mask]
        s_g = scores[mask]
        if len(np.unique(y_g)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_g, s_g)
        auc = roc_auc_score(y_g, s_g)
        plt.plot(fpr, tpr, label=f"{g} (AUC={auc:.3f})", color=palette[i], linewidth=2)
        
    plt.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Chance")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve by Group", fontweight='bold', fontsize=16)
    plt.legend(title="Group")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(outdir / "roc_by_group.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Fairness analysis (Step 9)")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset CSV")
    parser.add_argument("--protected", type=str, default="Pct_afro_american", help="Protected attribute column name (e.g., ethnicity)")
    parser.add_argument("--score-col", type=str, default=None, help="Score column (default: 'Predicted probabilities')")
    parser.add_argument("--target-col", type=str, default="target", help="Target column (default: target)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold on score (default: 0.5)")
    parser.add_argument("--delta", type=float, default=0.1, help="Equivalence margin (delta) for TOST test on scores.")
    parser.add_argument("--plots-dir", type=str, default=str(DEFAULT_PLOTS_DIR), help="Directory to save plots")
    parser.add_argument("--html-dir", type=str, default=str(Path(__file__).resolve().parent / "result_html"), help="Directory to save HTML report")
    parser.add_argument("--force-model", action="store_true", help="Try to compute scores from a model if not present (off by default)")
    args = parser.parse_args()

    # --- Setup plotting style ---
    sns.set_theme(style="whitegrid", palette="viridis")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150

    # Define and clear output directories before starting
    plots_dir = Path(args.plots_dir)
    html_dir = Path(args.html_dir)
    print(f"Clearing previous files in {plots_dir} and {html_dir}...")
    clear_output_dirs(plots_dir, html_dir)

    report_data = {}

    data_path = find_dataset_path(args.data)
    report_data['data_path'] = str(data_path)
    print(f"Loading dataset: {data_path}")
    df = pd.read_csv(data_path)

    # Identify protected attribute and define groups
    protected_col = autodetect_protected_attribute(df, args.protected)
    report_data['protected_col'] = protected_col
    print(f"Protected attribute: {protected_col}")
    raw_groups = df[protected_col]
    
    # Bin continuous protected attributes into quartiles for analysis
    if pd.api.types.is_numeric_dtype(raw_groups) and raw_groups.nunique() > 10:
        print(f"Attribute '{protected_col}' is continuous. Binning into quartiles.")
        groups = pd.qcut(raw_groups, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates='drop')
        groups.name = f"{protected_col}_quartile"
    else:
        groups = define_groups(raw_groups)
        groups.name = protected_col

    # Add the new groups column to the dataframe for the candidate variable analysis
    df[groups.name] = groups

    # Show distribution
    dist = summarize_group_distribution(groups)
    report_data['group_distribution'] = dist
    print("\nGroup distribution:")
    print(dist)

    # Scores, labels, and decisions
    scores, y_true, y_hat = get_scores_and_labels(df, args.score_col, args.target_col, args.threshold)

    # Metrics
    metrics = compute_group_metrics(y_true, y_hat, scores, groups)
    report_data['metrics'] = metrics
    print("\nFairness metrics (majority vs minority among most frequent groups):")
    for k, v in metrics.items():
        print(f"- {k}: {v}")

    # Equivalence Test
    g_min = metrics.get("group_minority")
    g_maj = metrics.get("group_majority")
    if g_min and g_maj and args.delta is not None:
        tost_results = perform_equivalence_test(scores, groups, g_min, g_maj, args.delta)
        report_data['tost_results'] = tost_results
        print("\nFairness Equivalence Test (TOST):")
        print(f"- Delta: {tost_results['delta']}")
        print(f"- Mean Score Difference: {tost_results['mean_diff']:.4f}")
        print(f"- P-value: {tost_results['p_value']:.4f}")
        print(f"- Result: {tost_results['result']}")

    # Visualizations
    report_data['plots_dir'] = plots_dir
    plot_group_distribution(groups, plots_dir)
    plot_score_distributions(scores, groups, plots_dir)
    plot_confusion_by_group(y_true, y_hat, groups, plots_dir)
    plot_roc_by_group(y_true, scores, groups, plots_dir)

    # FPDP plot
    score_col_name = args.score_col if args.score_col else "Predicted probabilities"
    plot_fairness_partial_dependence(df, groups.name, score_col_name, plots_dir)

    # Save metrics to CSV/JSON for record
    out_csv = plots_dir / "fairness_metrics.csv"
    flat = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    # Expand calibration dict
    calib = metrics.get("Calibration_abs_error_by_group", {})
    if isinstance(calib, dict):
        for g, val in (calib.items() if isinstance(calib, dict) else []):
            flat[f"Calibration_abs_error__{g}"] = val
    pd.DataFrame([flat]).to_csv(out_csv, index=False)
    print(f"\nSaved metrics to: {out_csv}")

    print(f"Saved plots to: {plots_dir}")

    # Candidate variable analysis
    df['y_hat'] = y_hat
    features_to_check = [col for col in df.columns if col not in [args.target_col, 'Predicted probabilities', 'Predictions', 'y_hat', protected_col]]
    candidate_results = find_candidate_variables(df, protected_col=groups.name, prediction_col='y_hat', features_to_test=features_to_check)
    report_data['candidate_results'] = candidate_results
    plot_candidate_analysis(candidate_results, plots_dir)

    # Print candidate results to console for consistency
    print("\n" + "="*60)
    print("CANDIDATE VARIABLE ANALYSIS")
    print("="*60)
    if candidate_results:
        print(f"Overall fairness test: Chi-squared = {candidate_results.get('overall_chi2', 0):.2f}, p-value = {candidate_results.get('overall_p', 0):.4f}")
        print(candidate_results.get('message', ''))
        if candidate_results.get('candidate_variables'):
            print("Candidate Variables Found:")
            for feature, values in candidate_results['candidate_variables'].items():
                print(f"  - '{feature}': Fairness restored for values: {values}")
        if candidate_results.get('candidate_variables_message'):
            print(candidate_results['candidate_variables_message'])

    # Generate HTML report
    generate_html_report(report_data, html_dir)


def perform_chi2_test(df: pd.DataFrame, protected_col: str, prediction_col: str) -> Tuple[float, float, int, np.ndarray]:
    """Performs a Chi-Squared test for independence between a protected group and model predictions."""
    contingency_table = pd.crosstab(df[protected_col], df[prediction_col])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p, dof, expected


def find_candidate_variables(
    df: pd.DataFrame,
    protected_col: str,
    prediction_col: str,
    features_to_test: List[str],
    alpha: float = 0.05
) -> Dict:
    """Identifies candidate variables that might be the source of bias."""
    results = {}
    chi2_overall, p_overall, _, _ = perform_chi2_test(df, protected_col, prediction_col)
    results['overall_chi2'] = chi2_overall
    results['overall_p'] = p_overall
    results['alpha'] = alpha

    if p_overall >= alpha:
        results['message'] = "The null hypothesis of fairness is NOT rejected overall."
        results['candidate_variables'] = {}
        return results

    results['message'] = f"The null hypothesis of fairness IS rejected (p < {alpha}). Searching for candidate variables..."
    candidate_variables = {}
    all_p_values = {}

    for feature in features_to_test:
        if feature == protected_col or feature in ['target', 'Predicted probabilities', 'Predictions']:
            continue

        is_candidate = False
        candidate_values = []
        max_p_subset = 0.0

        # For continuous variables, bin them first
        if df[feature].dtype in ['int64', 'float64'] and df[feature].nunique() > 10:
            try:
                # Use qcut for binning, handle potential errors with duplicate edges
                binned_feature = pd.qcut(df[feature], q=4, duplicates='drop')
                unique_values = binned_feature.unique()
            except ValueError as e:
                print(f"Could not bin continuous feature '{feature}': {e}. Skipping.")
                continue
        else: # Categorical variable
            unique_values = df[feature].unique()
            binned_feature = df[feature]

        for value in unique_values:
            subset_df = df[binned_feature == value]
            if subset_df[protected_col].nunique() < 2:
                continue

            _, p_subset, _, _ = perform_chi2_test(subset_df, protected_col, prediction_col)
            max_p_subset = max(max_p_subset, p_subset)

            # A feature is a candidate if for one of its values, fairness is restored (p > alpha)
            if p_subset >= alpha:
                is_candidate = True
                candidate_values.append(str(value))

        if is_candidate:
            candidate_variables[feature] = candidate_values
        
        all_p_values[feature] = max_p_subset

    results['candidate_variables'] = candidate_variables
    results['all_p_values'] = all_p_values
    if not candidate_variables:
        results['candidate_variables_message'] = "No candidate variables found that restore fairness by conditioning."

    return results


if __name__ == "__main__":
    main()


