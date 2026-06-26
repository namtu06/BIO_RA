import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt


def differential_expression(case_expr, control_expr):
    """
    case_expr and control_expr:
    DataFrames with genes as index and samples as columns
    """

    # log2 fold change
    log2fc = (
        case_expr.mean(axis=1)
        - control_expr.mean(axis=1)
    )

    # p-values
    pvals = []

    for gene in case_expr.index:
        _, p = ttest_ind(
            case_expr.loc[gene],
            control_expr.loc[gene],
            equal_var=False,
            nan_policy="omit"
        )
        pvals.append(p)

    pvals = np.nan_to_num(
    pvals,
    nan=1.0
    )

    # FDR correction
    fdr = multipletests(
        pvals,
        method="fdr_bh"
    )[1]

    results = pd.DataFrame({
        "log2FC": log2fc,
        "pvalue": pvals,
        "adjustedpvalue": fdr
    }, index=case_expr.index)

    results["-log(adjustedp)"] = -np.log10(
        results["adjustedpvalue"]
    )

    # categories
    results["category"] = "Not Significant"

    results.loc[
        (results["adjustedpvalue"] < 0.05)
        & (results["log2FC"] > 1),
        "category"
    ] = "Up"

    results.loc[
        (results["adjustedpvalue"] < 0.05)
        & (results["log2FC"] < -1),
        "category"
    ] = "Down"

    return results


def volcano_plot(results, title):
    colors = {
        "Up": "red",
        "Down": "blue",
        "Not Significant": "lightgray"
    }

    plt.figure(figsize=(10, 8))

    for cat in ["Not Significant", "Up", "Down"]:

        subset = results[
            results["category_rna"] == cat
        ]

        plt.scatter(
            subset["log2FC_rna"],
            subset["-log(adjustedp)_rna"],
            c=colors[cat],
            label=cat,
            s=10,
            alpha=0.7
        )

    plt.axvline(
        1,
        linestyle="--",
        color="black"
    )

    plt.axvline(
        -1,
        linestyle="--",
        color="black"
    )

    plt.axhline(
        -np.log10(0.05),
        linestyle="--",
        color="black"
    )

    plt.xlabel("log2 Fold Change")
    plt.ylabel("-log10(adjusted p-value)")
    plt.title(title)
    plt.legend()

    plt.show()
    
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt


def pca_plot(
    case_expr,
    control_expr,
    case_label="Case",
    control_label="Control",
    title="PCA"
):
    """
    Expression matrices:
    rows = genes
    columns = samples
    """

    # combine samples
    X = pd.concat(
        [case_expr, control_expr],
        axis=1
    )

    # transpose: samples x genes
    X = X.T

    # standardize
    X_scaled = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    # plotting dataframe
    pca_df = pd.DataFrame(
        pcs,
        columns=["PC1", "PC2"]
    )

    pca_df["Group"] = (
        [case_label] * case_expr.shape[1]
        + [control_label] * control_expr.shape[1]
    )

    # plot
    plt.figure(figsize=(8, 6))

    for group in pca_df["Group"].unique():

        subset = pca_df[
            pca_df["Group"] == group
        ]

        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            label=group,
            alpha=0.8
        )

    plt.xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
    )

    plt.ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
    )

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return pca_df, pca