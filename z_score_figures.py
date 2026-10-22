#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:51:14 2026

@author: Zuza Kozik
"""

"""
Sylvian dataset analysis:

- Per-protein two-sample Student's t-tests (equal variances) for three comparisons:
    1) Damage / No damage:
       DPN1 +S +4OHT vs No DPN1 +S +4OHT
    2) Damage / Damage + ATMi:
       DPN1 +S +4OHT vs DPN1 +S +4OHT +ATM
    3) No damage / No damage + ATMi:
       No DPN1 +S +4OHT vs No DPN1 +S +4OHT +ATM

- Log2 fold change (Difference) calculation for each comparison
  (input abundances are already log2, so Difference = mean(log2A) - mean(log2B))
- Enrichment analysis using gseapy.enrichr on up/down gene lists
- Scatter plots (rank vs log2FC) for:
    - Nuclear pore proteins (Damage / No damage)
    - Chromatin organization proteins (Damage / Damage + ATMi)
    - Chromatin organization proteins (No damage / No damage + ATMi)

  All proteins are plotted; only significant category proteins
  (p < 0.05 and |log2FC| > 0.25 and annotation keyword hit)
  are highlighted and annotated.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from adjustText import adjust_text
from scipy import stats
import gseapy as gp
from tqdm import tqdm

# -------------------------------------------------------------------------
# Global settings / thresholds
# -------------------------------------------------------------------------

# File paths
INPUT_CSV = "sylvian.csv"
GSEA_RESULTS_DIR = "gsea_results"
FIG_DIR = "figures"

# Ensure output dirs exist
os.makedirs(GSEA_RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Enrichment thresholds (for up/down gene lists)
GSEA_PVAL_THRESHOLD = 0.05
GSEA_LFC_THRESHOLD = 0.25  # since input is already log2, this is mean difference threshold

# Scatter significance thresholds
SCATTER_PVAL_THRESHOLD = 0.05
SCATTER_LFC_THRESHOLD = 0.25

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


# -------------------------------------------------------------------------
# Utility: compute Student's t-test and log2FC for a pair of conditions
# -------------------------------------------------------------------------
def compute_ttest_and_log2fc(df, groupA_cols, groupB_cols, labelA, labelB):
    """
    For each row, compute:
      - Student's two-sample t-test p-value between groupA and groupB
        (equal variances assumed)
      - log2 fold change (Difference) as:
            mean(log2A) - mean(log2B)
        because input abundances are already in log2 scale.

    Results are added as new columns in df:
      "Students T-test p-value {labelA}_{labelB}"
      "Students T-test Difference {labelA}_{labelB}"

    Returns:
      (pval_col_name, diff_col_name)
    """
    # Extract replicate intensities (already log2)
    a = df[groupA_cols].astype(float)
    b = df[groupB_cols].astype(float)

    # Means in log2 space
    meanA = a.mean(axis=1, skipna=True)
    meanB = b.mean(axis=1, skipna=True)

    # Student's two-sample t-test (equal variances)
    t_stat, p_val = stats.ttest_ind(
        a.values,
        b.values,
        axis=1,
        nan_policy="omit",
        equal_var=True,  # Student's t-test (not Welch)
    )

    # Log2 fold-change: difference of means in log2 space
    log2fc = meanA - meanB

    # Column names following your convention
    pval_col = f"Students T-test p-value {labelA}_{labelB}"
    diff_col = f"Students T-test Difference {labelA}_{labelB}"

    df[pval_col] = p_val
    df[diff_col] = log2fc

    return pval_col, diff_col


# -------------------------------------------------------------------------
# Enrichment analysis wrapper (Enrichr via gseapy)
# -------------------------------------------------------------------------
def run_enrichment_analysis(data, comparison_groups):
    """
    comparison_groups: dict
        key: comparison name (used for subdirectories and filenames)
        value: (pval_col_name, diff_col_name)

    Uses:
      - GSEA_PVAL_THRESHOLD
      - GSEA_LFC_THRESHOLD

    Writes CSV results into:
      GSEA_RESULTS_DIR / <comparison_name> / <comparison>_UP_<DB>.csv
      GSEA_RESULTS_DIR / <comparison_name> / <comparison>_DOWN_<DB>.csv
    """
    databases = [
        "GO_Biological_Process_2023",
        "GO_Cellular_Component_2023",
        "GO_Molecular_Function_2023",
        "KEGG_2021_Human",
        "Reactome_Pathways_2024",
        "MSigDB_Hallmark_2020",
        "WikiPathways_2024_Human",
        "CORUM",
    ]

    # Verify available databases
    gseapy_dbs = gp.get_library_name()
    available_databases = [db for db in databases if db in gseapy_dbs]

    if not available_databases:
        raise ValueError("None of the requested databases are available in GSEAPY.")

    for condition, (pval_col, lfc_col) in tqdm(
        comparison_groups.items(), desc="Running enrichment analysis"
    ):
        if pval_col not in data.columns or lfc_col not in data.columns:
            print(f"Skipping {condition}: required columns not found.")
            continue

        cond_dir = os.path.join(GSEA_RESULTS_DIR, condition.replace(" ", "_"))
        os.makedirs(cond_dir, exist_ok=True)

        # Upregulated genes
        up_mask = (data[pval_col] < GSEA_PVAL_THRESHOLD) & (
            data[lfc_col] >= GSEA_LFC_THRESHOLD
        )
        up_genes = (
            data.loc[up_mask, "Gene name"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        # Downregulated genes
        down_mask = (data[pval_col] < GSEA_PVAL_THRESHOLD) & (
            data[lfc_col] <= -GSEA_LFC_THRESHOLD
        )
        down_genes = (
            data.loc[down_mask, "Gene name"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        for db in available_databases:
            try:
                if up_genes:
                    res_up = gp.enrichr(
                        gene_list=up_genes,
                        gene_sets=db,
                        outdir=None,
                        no_plot=True,
                    )
                    res_up.results.to_csv(
                        os.path.join(cond_dir, f"{condition}_UP_{db}.csv"),
                        index=False,
                    )

                if down_genes:
                    res_down = gp.enrichr(
                        gene_list=down_genes,
                        gene_sets=db,
                        outdir=None,
                        no_plot=True,
                    )
                    res_down.results.to_csv(
                        os.path.join(cond_dir, f"{condition}_DOWN_{db}.csv"),
                        index=False,
                    )

            except Exception as e:
                print(f"⚠️ Error in enrichment for {condition} - {db}: {e}")

    print("✅ Enrichment analysis completed.")


# -------------------------------------------------------------------------
# Plotting: scatter for annotation-specific significant proteins
# -------------------------------------------------------------------------
def plot_annotation_scatter(
    data,
    diff_col,
    pval_col,
    annotation_keyword,
    pval_thresh,
    lfc_thresh,
    title,
    output_basename,
):
    """
    Create a scatter plot for a given comparison:

      - All proteins with non-NA diff are plotted:
          x = rank (sorted by diff descending)
          y = diff (log2FC)
          color = diff (rainbow colormap)

      - Only the subset of proteins that are:
          * nominally significant (p < pval_thresh)
          * |diff| > lfc_thresh
          * contain `annotation_keyword` (case-insensitive) in "Combined annotations"
        are highlighted with a gold edge and annotated.

    Aesthetics follow your example:
      - rainbow colormap, black edges on base points
      - grid, dashed 0 line, colorbar labelled "Log2 Fold Change"
      - adjustText for label spreading of highlighted points
    """
    if diff_col not in data.columns or pval_col not in data.columns:
        print(f"Skipping plot {title}: required columns not found.")
        return

    if "Combined annotations" not in data.columns:
        print("Combined annotations column not found; cannot filter by keyword.")
        return

    # Base DataFrame: all points that have a diff value
    df_plot = data[[diff_col, pval_col, "Gene name", "Combined annotations"]].copy()
    df_plot = df_plot.dropna(subset=[diff_col])

    if df_plot.empty:
        print(f"No data to plot for {title}.")
        return

    # Sort by diff for ranking
    df_plot = df_plot.sort_values(by=diff_col, ascending=False).reset_index(drop=True)

    # Normalization for colormap
    norm = Normalize(vmin=df_plot[diff_col].min(), vmax=df_plot[diff_col].max())
    cmap = cm.get_cmap("rainbow")

    # Boolean mask: significant + annotation keyword
    annot_mask = df_plot["Combined annotations"].astype(str).str.contains(
        annotation_keyword, case=False, na=False
    )
    sig_mask = (df_plot[pval_col] < pval_thresh) & (
        df_plot[diff_col].abs() > lfc_thresh
    )
    highlight_mask = annot_mask & sig_mask

    # Figure
    fig, ax = plt.subplots(figsize=(5, 3))

    # Base scatter: all proteins
    sc = ax.scatter(
        x=df_plot.index,
        y=df_plot[diff_col],
        c=df_plot[diff_col],
        cmap=cmap,
        norm=norm,
        alpha=0.9,
        linewidth=0.3,
        edgecolor="black",
        s=40,
    )

    # Highlight significant annotation-matching proteins
    if highlight_mask.any():
        ax.scatter(
            x=df_plot.index[highlight_mask],
            y=df_plot.loc[highlight_mask, diff_col],
            c=df_plot.loc[highlight_mask, diff_col],
            cmap=cmap,
            norm=norm,
            edgecolor="gold",
            linewidth=1.5,
            s=40,
            alpha=1.0,
        )

        # Annotate only the highlighted proteins
        texts = []
        for i, row in df_plot[highlight_mask].iterrows():
            gene_name = str(row["Gene name"])
            texts.append(
                ax.text(
                    i,
                    row[diff_col],
                    gene_name,
                    fontsize=7,
                    color="black",
                    weight="bold",
                )
            )

        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            expand_text=(1.2, 1.4),
            expand_points=(1.2, 1.4),
            only_move={"points": "y", "text": "xy"},
        )
    else:
        print(f"No significant '{annotation_keyword}' proteins to highlight for {title}.")

    # Formatting
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.75)
    ax.set_xlabel("Protein Rank", fontsize=9)
    ax.set_ylabel("Difference (Log2-FC)", fontsize=9)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, color="lightgray")

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7)
    cbar.set_label("Log2 Fold Change", fontsize=9)

    plt.tight_layout()
    out_svg = os.path.join(FIG_DIR, f"{output_basename}.svg")
    out_png = os.path.join(FIG_DIR, f"{output_basename}.png")
    plt.savefig(out_svg, format="svg", dpi=300, bbox_inches="tight")
    plt.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure: {out_svg} and {out_png}")


# -------------------------------------------------------------------------
# Main workflow
# -------------------------------------------------------------------------
def main():
    # ---------------------------
    # 1. Load data
    # ---------------------------
    data = pd.read_csv(INPUT_CSV)

    # ---------------------------
    # 2. Define replicate columns
    # ---------------------------

    # DPN1 -S (not directly used in the three comparisons, but kept here)
    cols_dpn1_damage = [
        "DPN1 -S_R1",
        "DPN1 -S_R2",
        "DPN1 -S_R3",
    ]

    # DPN1 +S +4OHT (Damage, DPN1 present)
    cols_dpn1_plusS_4oht = [
        "DPN1 +S +4OHT_R1",
        "DPN1 +S +4OHT_R2",
        "DPN1 +S +4OHT_R3",
    ]

    # DPN1 +S +4OHT +ATM (Damage + ATMi, DPN1 present)
    cols_dpn1_plusS_4oht_atm = [
        "DPN1 +S +4OHT +ATM_R1",
        "DPN1 +S +4OHT +ATM_R2",
        "DPN1 +S +4OHT +ATM_R3",
    ]

    # No DPN1 -S (not directly used in three requested comparisons)
    cols_no_dpn1_damage = [
        "No DPN1 -S_R1",
        "No DPN1 -S_R2",
        "No DPN1 -S_R3",
    ]

    # No DPN1 +S +4OHT (Damage, no DPN1)
    cols_no_dpn1_plusS_4oht = [
        "No DPN1 +S +4OHT_R1",
        "No DPN1 +S +4OHT_R2",
        "No DPN1 +S +4OHT_R3",
    ]

    # No DPN1 +S +4OHT +ATM (Damage + ATMi, no DPN1)
    cols_no_dpn1_plusS_4oht_atm = [
        "No DPN1 +S +4OHT +ATM_R1",
        "No DPN1 +S +4OHT +ATM_R2",
        "No DPN1 +S +4OHT +ATM_R3",
    ]

    # ---------------------------
    # 3. Compute the requested comparisons
    # ---------------------------

    # 3.1 Damage / No damage:
    #     DPN1 +S +4OHT vs No DPN1 +S +4OHT
    labelA_1 = "DPN1 +S +4OHT"
    labelB_1 = "No DPN1 +S +4OHT"
    pval_col_1, diff_col_1 = compute_ttest_and_log2fc(
        data,
        groupA_cols=cols_dpn1_plusS_4oht,
        groupB_cols=cols_no_dpn1_plusS_4oht,
        labelA=labelA_1,
        labelB=labelB_1,
    )

    # 3.2 Damage / Damage + ATMi:
    #     DPN1 +S +4OHT vs DPN1 +S +4OHT +ATM
    labelA_2 = "DPN1 +S +4OHT"
    labelB_2 = "DPN1 +S +4OHT +ATM"
    pval_col_2, diff_col_2 = compute_ttest_and_log2fc(
        data,
        groupA_cols=cols_dpn1_plusS_4oht,
        groupB_cols=cols_dpn1_plusS_4oht_atm,
        labelA=labelA_2,
        labelB=labelB_2,
    )

    # 3.3 No damage / No damage + ATMi:
    #     No DPN1 +S +4OHT vs No DPN1 +S +4OHT +ATM
    labelA_3 = "No DPN1 +S +4OHT"
    labelB_3 = "No DPN1 +S +4OHT +ATM"
    pval_col_3, diff_col_3 = compute_ttest_and_log2fc(
        data,
        groupA_cols=cols_no_dpn1_plusS_4oht,
        groupB_cols=cols_no_dpn1_plusS_4oht_atm,
        labelA=labelA_3,
        labelB=labelB_3,
    )

    # ---------------------------
    # 4. Run enrichment analysis
    # ---------------------------
    comparison_groups = {
        "DPN1+S+4OHT_vs_NoDPN1+S+4OHT": (pval_col_1, diff_col_1),
        "DPN1+S+4OHT_vs_DPN1+S+4OHT+ATM": (pval_col_2, diff_col_2),
        "NoDPN1+S+4OHT_vs_NoDPN1+S+4OHT+ATM": (pval_col_3, diff_col_3),
    }

    run_enrichment_analysis(data, comparison_groups)

    # ---------------------------
    # 5. Scatter plots with your aesthetics
    # ---------------------------

    # 5.1 Nuclear pore proteins: Damage / No damage
    plot_annotation_scatter(
        data=data,
        diff_col=diff_col_1,
        pval_col=pval_col_1,
        annotation_keyword="nuclear pore",
        pval_thresh=SCATTER_PVAL_THRESHOLD,
        lfc_thresh=SCATTER_LFC_THRESHOLD,
        title="Nuclear pore proteins: DPN1 +S +4OHT vs No DPN1 +S +4OHT",
        output_basename="scatter_nuclear_pore_Damage_vs_NoDamage",
    )

    # 5.2 Chromatin organization: Damage / Damage + ATMi
    plot_annotation_scatter(
        data=data,
        diff_col=diff_col_2,
        pval_col=pval_col_2,
        annotation_keyword="chromatin organization",
        pval_thresh=SCATTER_PVAL_THRESHOLD,
        lfc_thresh=SCATTER_LFC_THRESHOLD,
        title="Chromatin organization: DPN1 +S +4OHT vs DPN1 +S +4OHT +ATM",
        output_basename="scatter_chromatin_org_Damage_vs_Damage_ATMi",
    )

    # 5.3 Chromatin organization: No damage / No damage + ATMi
    plot_annotation_scatter(
        data=data,
        diff_col=diff_col_3,
        pval_col=pval_col_3,
        annotation_keyword="chromatin organization",
        pval_thresh=SCATTER_PVAL_THRESHOLD,
        lfc_thresh=SCATTER_LFC_THRESHOLD,
        title="Chromatin organization: No DPN1 +S +4OHT vs No DPN1 +S +4OHT +ATM",
        output_basename="scatter_chromatin_org_NoDamage_vs_NoDamage_ATMi",
    )

    # ---------------------------
    # 6. Save updated table with stats
    # ---------------------------
    out_table = "sylvian_with_ttests_and_log2fc.csv"
    data.to_csv(out_table, index=False)
    print(f"✅ Analysis complete. Results table saved as: {out_table}")


if __name__ == "__main__":
    main()
    
#%%
"""
Sylvian dataset – t-tests + scatter plots only

Comparisons (per-protein two-sample Student's t-tests, equal variances):

1) Damage / No damage:
   DPN1 +S +4OHT vs No DPN1 +S +4OHT

2) Damage / Damage + ATMi:
   DPN1 +S +4OHT vs DPN1 +S +4OHT +ATM

3) No damage / No damage + ATMi:
   No DPN1 +S +4OHT vs No DPN1 +S +4OHT +ATM

Assumptions:
- Abundances are already log2-transformed
- Log2FC (Difference) is computed as:
      mean(log2A) - mean(log2B)

Scatter plots:
- All proteins plotted (rank vs Difference)
- Only proteins that are:
    * p < 0.05
    * |Difference| > 0.5
    * contain specified keyword in "Combined annotations"
  are highlighted and annotated.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from adjustText import adjust_text
from scipy import stats

# -------------------------------------------------------------------------
# I have adjusted this part to include swi/snf factors in pink in the plot, 
# so that there's a dual annotation of sorts
# -------------------------------------------------------------------------

SWI_SNF_GENES = {
    "SMARCA4", "SMARCA2",
    "SMARCB1", "SMARCC1", "SMARCC2",
    "SMARCD1", "SMARCD2", "SMARCE1",
    "ARID1A", "ARID1B", "ARID2",
    "PBRM1", "BRD7", "BRD9",
    "DPF2", "PHF10",
    "ACTL6A", "ACTB",
    "BCL7B", "BCL7C",
    "SS18", "SS18L1",
}

# File paths
INPUT_CSV = "sylvian.csv"
FIG_DIR = "figures"

# Ensure output dir exists
os.makedirs(FIG_DIR, exist_ok=True)

# Significance thresholds
PVALUE_THRESHOLD = 0.05
LOG2FC_THRESHOLD = 0.3  # |Difference| > 0.5

# Suppress some warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


# -------------------------------------------------------------------------
# Utility: compute Student's t-test and log2FC for a pair of conditions
# -------------------------------------------------------------------------
def compute_ttest_and_log2fc(df, groupA_cols, groupB_cols, labelA, labelB):
    """
    For each row, compute:
      - Student's two-sample t-test p-value between groupA and groupB
        (equal variances assumed)
      - log2 fold change (Difference) as:
            mean(log2A) - mean(log2B)
        because input abundances are already in log2 scale.

    Results are added as new columns in df:
      "Students T-test p-value {labelA}_{labelB}"
      "Students T-test Difference {labelA}_{labelB}"

    Returns:
      (pval_col_name, diff_col_name)
    """
    # Extract replicate intensities (already log2)
    a = df[groupA_cols].astype(float)
    b = df[groupB_cols].astype(float)

    # Means in log2 space
    meanA = a.mean(axis=1, skipna=True)
    meanB = b.mean(axis=1, skipna=True)

    # Student's two-sample t-test (equal variances)
    t_stat, p_val = stats.ttest_ind(
        a.values,
        b.values,
        axis=1,
        nan_policy="omit",
        equal_var=True,  # Student's t-test (not Welch)
    )

    # Log2 fold-change: difference of means in log2 space
    log2fc = meanA - meanB

    # Column names following your convention
    pval_col = f"Students T-test p-value {labelA}_{labelB}"
    diff_col = f"Students T-test Difference {labelA}_{labelB}"

    df[pval_col] = p_val
    df[diff_col] = log2fc

    return pval_col, diff_col


# -------------------------------------------------------------------------
# Plotting: scatter for annotation-specific significant proteins
# -------------------------------------------------------------------------
def plot_annotation_scatter(
    data,
    diff_col,
    pval_col,
    annotation_keyword,
    pval_thresh,
    lfc_thresh,
    title,
    output_basename,
):
    """
    Create a scatter plot for a given comparison:

      - All proteins with non-NA diff are plotted:
          x = rank (sorted by diff descending)
          y = diff (log2FC)
          color = diff (rainbow colormap)

      - Only the subset of proteins that are:
          * nominally significant (p < pval_thresh)
          * |diff| > lfc_thresh
          * contain `annotation_keyword` (case-insensitive) in "Combined annotations"
        are highlighted with a gold edge and annotated.

    Aesthetics:
      - rainbow colormap, black edges on base points
      - grid, dashed 0 line, colorbar labelled "Log2 Fold Change"
      - adjustText for label spreading of highlighted points
    """
    if diff_col not in data.columns or pval_col not in data.columns:
        print(f"Skipping plot {title}: required columns not found.")
        return

    if "Combined annotations" not in data.columns:
        print("Combined annotations column not found; cannot filter by keyword.")
        return

    # Base DataFrame: all points that have a diff value
    df_plot = data[[diff_col, pval_col, "Gene name", "Combined annotations"]].copy()
    df_plot = df_plot.dropna(subset=[diff_col])

    if df_plot.empty:
        print(f"No data to plot for {title}.")
        return

    # Sort by diff for ranking
    df_plot = df_plot.sort_values(by=diff_col, ascending=False).reset_index(drop=True)

    # Normalization for colormap
    norm = Normalize(vmin=df_plot[diff_col].min(), vmax=df_plot[diff_col].max())
    cmap = cm.get_cmap("rainbow")

    # Boolean mask: significant + annotation keyword
    # Chromatin remodelling annotations
    chromatin_mask = (
        df_plot["Combined annotations"]
        .fillna("")
        .str.contains(annotation_keyword, case=False, regex=False)
    )
    
    # SWI/SNF complex membership
    swi_snf_mask = (
        df_plot["Gene name"]
        .fillna("")
        .str.strip()
        .str.upper()
        .isin(SWI_SNF_GENES)
    )
    
    # Statistical significance
    sig_mask = (
        (df_plot[pval_col] < pval_thresh) &
        (df_plot[diff_col].abs() > lfc_thresh)
    )

# Highlight either category
    highlight_mask = (chromatin_mask | swi_snf_mask) & sig_mask

    # Figure
    fig, ax = plt.subplots(figsize=(5, 3))

    # Base scatter: all proteins
    sc = ax.scatter(
        x=df_plot.index,
        y=df_plot[diff_col],
        c=df_plot[diff_col],
        cmap=cmap,
        norm=norm,
        alpha=0.9,
        linewidth=0.3,
        edgecolor="black",
        s=40,
    )

    # Highlight significant annotation-matching proteins
    if highlight_mask.any():
        ax.scatter(
            x=df_plot.index[highlight_mask],
            y=df_plot.loc[highlight_mask, diff_col],
            c=df_plot.loc[highlight_mask, diff_col],
            cmap=cmap,
            norm=norm,
            edgecolor="gold",
            linewidth=1.5,
            s=40,
            alpha=1.0,
        )

        # Annotate only the highlighted proteins
        texts = []
        for i, row in df_plot[highlight_mask].iterrows():

            gene_name = str(row["Gene name"])
        
            # SWI/SNF takes priority if a protein belongs to both groups
            if swi_snf_mask.loc[i]:
                text_color = "#DC6B83"
            else:
                text_color = "black"
        
            texts.append(
                ax.text(
                    i,
                    row[diff_col],
                    gene_name,
                    fontsize=7,
                    color=text_color,
                    weight="bold",
                )
            )

        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            expand_text=(1.2, 1.4),
            expand_points=(1.2, 1.4),
            only_move={"points": "y", "text": "xy"},
        )
    else:
        print(f"No significant '{annotation_keyword}' proteins to highlight for {title}.")

    # Formatting
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.75)
    ax.set_xlabel("Protein Rank", fontsize=9)
    ax.set_ylabel("Difference (Log2-FC)", fontsize=9)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, color="lightgray")

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7)
    cbar.set_label("Log2 Fold Change", fontsize=9)

    plt.tight_layout()
    out_svg = os.path.join(FIG_DIR, f"{output_basename}.svg")
    out_png = os.path.join(FIG_DIR, f"{output_basename}.png")
    plt.savefig(out_svg, format="svg", dpi=300, bbox_inches="tight")
    plt.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure: {out_svg} and {out_png}")



# -------------------------------------------------------------------------
# Main workflow
# -------------------------------------------------------------------------
def main():
    # ---------------------------
    # 1. Load data
    # ---------------------------
    data = pd.read_csv(INPUT_CSV)

    # ---------------------------
    # 2. Define replicate columns
    # ---------------------------

    # DPN1 +S +4OHT (Damage, DPN1 present)
    cols_dpn1_plusS_4oht = [
        "DPN1 +S +4OHT_R1",
        "DPN1 +S +4OHT_R2",
        "DPN1 +S +4OHT_R3",
    ]

    # DPN1 +S +4OHT +ATM (Damage + ATMi, DPN1 present)
    cols_dpn1_plusS_4oht_atm = [
        "DPN1 +S +4OHT +ATM_R1",
        "DPN1 +S +4OHT +ATM_R2",
        "DPN1 +S +4OHT +ATM_R3",
    ]

    # No DPN1 +S +4OHT (Damage, no DPN1)
    cols_no_dpn1_plusS_4oht = [
        "No DPN1 +S +4OHT_R1",
        "No DPN1 +S +4OHT_R2",
        "No DPN1 +S +4OHT_R3",
    ]

    # No DPN1 +S +4OHT +ATM (Damage + ATMi, no DPN1)
    cols_no_dpn1_plusS_4oht_atm = [
        "No DPN1 +S +4OHT +ATM_R1",
        "No DPN1 +S +4OHT +ATM_R2",
        "No DPN1 +S +4OHT +ATM_R3",
    ]

    # ---------------------------
    # 3. Compute the requested comparisons
    # ---------------------------

    # 3.1 Damage / No damage:
    #     DPN1 +S +4OHT vs No DPN1 +S +4OHT
    labelA_1 = "DPN1 +S +4OHT"
    labelB_1 = "No DPN1 +S +4OHT"
    pval_col_1, diff_col_1 = compute_ttest_and_log2fc(
        data,
        groupA_cols=cols_dpn1_plusS_4oht,
        groupB_cols=cols_no_dpn1_plusS_4oht,
        labelA=labelA_1,
        labelB=labelB_1,
    )

    # 3.2 Damage / Damage + ATMi:
    #     DPN1 +S +4OHT vs DPN1 +S +4OHT +ATM
    labelA_2 = "DPN1 +S +4OHT"
    labelB_2 = "DPN1 +S +4OHT +ATM"
    pval_col_2, diff_col_2 = compute_ttest_and_log2fc(
        data,
        groupA_cols=cols_dpn1_plusS_4oht,
        groupB_cols=cols_dpn1_plusS_4oht_atm,
        labelA=labelA_2,
        labelB=labelB_2,
    )

    # 3.3 No damage / No damage + ATMi:
    #     No DPN1 +S +4OHT vs No DPN1 +S +4OHT +ATM
    labelA_3 = "No DPN1 +S +4OHT"
    labelB_3 = "No DPN1 +S +4OHT +ATM"
    pval_col_3, diff_col_3 = compute_ttest_and_log2fc(
        data,
        groupA_cols=cols_no_dpn1_plusS_4oht,
        groupB_cols=cols_no_dpn1_plusS_4oht_atm,
        labelA=labelA_3,
        labelB=labelB_3,
    )

    # ---------------------------
    # 4. Scatter plots with your aesthetics
    # ---------------------------

    # 4.1 Nuclear pore proteins: Damage / No damage
    plot_annotation_scatter(
        data=data,
        diff_col=diff_col_1,
        pval_col=pval_col_1,
        annotation_keyword="nuclear pore",
        pval_thresh=PVALUE_THRESHOLD,
        lfc_thresh=LOG2FC_THRESHOLD,
        title="Nuclear pore proteins: DPN1 +S +4OHT vs No DPN1 +S +4OHT",
        output_basename="scatter_nuclear_pore_Damage_vs_NoDamage",
    )

    # 4.2 Chromatin organization: Damage / Damage + ATMi
    plot_annotation_scatter(
        data=data,
        diff_col=diff_col_2,
        pval_col=pval_col_2,
        annotation_keyword="chromatin remodeling",
        pval_thresh=PVALUE_THRESHOLD,
        lfc_thresh=LOG2FC_THRESHOLD,
        title="Chromatin remodel: DPN1 +S +4OHT vs DPN1 +S +4OHT +ATM",
        output_basename="scatter_chromatin_org_Damage_vs_Damage_ATMi",
    )

    # 4.3 Chromatin organization: No damage / No damage + ATMi
    plot_annotation_scatter(
        data=data,
        diff_col=diff_col_3,
        pval_col=pval_col_3,
        annotation_keyword="chromatin remodeling",
        pval_thresh=PVALUE_THRESHOLD,
        lfc_thresh=LOG2FC_THRESHOLD,
        title="Chromatin remodel: No DPN1 +S +4OHT vs No DPN1 +S +4OHT +ATM",
        output_basename="scatter_chromatin_org_NoDamage_vs_NoDamage_ATMi",
    )

    # ---------------------------
    # 5. Save updated table with stats
    # ---------------------------
    out_table = "sylvian_with_ttests_and_log2fc_only.csv"
    data.to_csv(out_table, index=False)
    print(f"✅ T-test + scatter analysis complete. Results table saved as: {out_table}")


if __name__ == "__main__":
    main()
    
    
#%%


data = pd.read_csv("sylvian.csv")

keywords = [
    "chromatin remodel",
    "chromatin remodell",
    "nucleosome remodel",
    "nucleosome remodell",
    "ATP-dependent chromatin",
    "SWI/SNF",
    "chromatin remodeling",
    "chromatin remodelling",
]

for keyword in keywords:
    matches = data[
        data["Combined annotations"].fillna("").str.contains(
            keyword, case=False, regex=False
        )
    ]

    print(f"\n{keyword}: {len(matches)} proteins")

    if not matches.empty:
        print(matches["Gene name"].dropna().unique())
