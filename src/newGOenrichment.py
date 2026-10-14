import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gseapy as gp
import textwrap


# ============================================================
# USER SETTINGS
# ============================================================

PVAL_COLUMN = "pvalue"
LOGFC_COLUMN = "log2FC"
GENE_COLUMN = "gene_name"

PVAL_THRESHOLD = 0.05
LOGFC_THRESHOLD = 1

# Number of genes sent to Enrichr
TOP_GENES = 4000

# Number of GO terms shown in plots
TOP_N = 20


# ============================================================
# GO LIBRARY
# ============================================================

GO_LIBRARY = "GO_Biological_Process_2023"


# ============================================================
# RUN GO ENRICHMENT
# ============================================================

def run_go(gene_list):

    if len(gene_list) == 0:
        return pd.DataFrame()

    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=GO_LIBRARY,
        organism="human",
        outdir=None
    )

    df = enr.results.copy()

    df = (
        df.sort_values("Adjusted P-value")
          .reset_index(drop=True)
    )

    return df


# ============================================================
# PLOT GO ENRICHMENT
# ============================================================

def plot_go(
    up_results,
    down_results=None,
    title=""
):

    # ========================================================
    # COMBINED MODE
    # ========================================================

    if down_results is None:

        df = up_results.copy()

        if df.empty:
            print(
                f"No significant BP terms found for {title}."
            )
            return

        df = (
            df.sort_values("Adjusted P-value")
              .head(TOP_N)
              .copy()
        )

        df["Term"] = (
            df["Term"]
            .str.replace(
                r"\s*\(GO:\d+\)",
                "",
                regex=True
            )
        )

        score = -np.log10(
            df["Adjusted P-value"]
        )

        y = np.arange(len(df))

        fig, ax = plt.subplots(
            figsize=(10, 10)
        )

        ax.barh(
            y,
            score,
            height=0.72
        )

        ax.set_yticks(y)

        ax.set_yticklabels(
            df["Term"],
            fontsize=10
        )

        ax.invert_yaxis()

        ax.set_xlabel(
            r"$-\log_{10}(\mathrm{Adjusted}\ p)$",
            fontsize=10
        )

        ax.set_title(
            title,
            fontsize=20,
            fontweight="bold"
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.show()

        return


    # ========================================================
    # SPLIT MODE
    # ========================================================

    up = up_results.copy()
    down = down_results.copy()


    # --------------------------------------------------------
    # Prepare UP
    # --------------------------------------------------------

    if not up.empty:

        up = (
            up.sort_values("Adjusted P-value")
              .head(TOP_N)
              .copy()
        )

        up["Term"] = (
            up["Term"]
            .str.replace(
                r"\s*\(GO:\d+\)",
                "",
                regex=True
            )
        )

        up["score"] = -np.log10(
            up["Adjusted P-value"]
        )


    # --------------------------------------------------------
    # Prepare DOWN
    # --------------------------------------------------------

    if not down.empty:

        down = (
            down.sort_values("Adjusted P-value")
              .head(TOP_N)
              .copy()
        )

        down["Term"] = (
            down["Term"]
            .str.replace(
                r"\s*\(GO:\d+\)",
                "",
                regex=True
            )
        )

        down["score"] = -np.log10(
            down["Adjusted P-value"]
        )


    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 10),
        gridspec_kw={
            "width_ratios": [1, 1]
        }
    )


    # --------------------------------------------------------
    # UPREGULATED
    # --------------------------------------------------------

    if up.empty:

        axes[0].axis("off")

    else:

        y_up = np.arange(len(up))

        axes[0].barh(
            y_up,
            up["score"],
            height=0.72
        )

        axes[0].set_yticks(y_up)

        axes[0].set_yticklabels(
            up["Term"],
            fontsize=10
        )

        axes[0].invert_yaxis()

        axes[0].set_xlabel(
            r"$-\log_{10}(\mathrm{Adjusted}\ p)$"
        )

        axes[0].set_title(
            "Upregulated",
            fontsize=18,
            fontweight="bold"
        )

        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)


    # --------------------------------------------------------
    # DOWNREGULATED
    # --------------------------------------------------------

    if down.empty:

        axes[1].axis("off")

    else:

        y_down = np.arange(len(down))

        axes[1].barh(
            y_down,
            down["score"],
            height=0.72
        )

        axes[1].set_yticks(y_down)

        axes[1].set_yticklabels(
            down["Term"],
            fontsize=10
        )

        axes[1].invert_yaxis()

        axes[1].set_xlabel(
            r"$-\log_{10}(\mathrm{Adjusted}\ p)$"
        )

        axes[1].set_title(
            "Downregulated",
            fontsize=18,
            fontweight="bold"
        )

        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)


    fig.suptitle(
        title,
        fontsize=24,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# ANALYZE SUBTYPE
# ============================================================

def analyze_subtype(
    csv_file,
    subtype_name,
    direction="combined",
    gene_filter=None,
    filter_before_significance=False
):

    df = pd.read_csv(
        csv_file
    )


    # ========================================================
    # OPTIONAL GENE FILTER
    # ========================================================

    if (
        gene_filter is not None
        and filter_before_significance
    ):

        df = df[
            df[GENE_COLUMN]
            .fillna("")
            .str.upper()
            .isin(gene_filter)
        ]


    # ========================================================
    # SIGNIFICANT GENES
    # ========================================================

    sig = df[
        (df[PVAL_COLUMN] < PVAL_THRESHOLD)
        &
        (abs(df[LOGFC_COLUMN]) > LOGFC_THRESHOLD)
    ].copy()


    if (
        gene_filter is not None
        and not filter_before_significance
    ):

        sig = sig[
            sig[GENE_COLUMN]
            .fillna("")
            .str.upper()
            .isin(gene_filter)
        ]


    print(
        f"\n{subtype_name}"
    )

    print(
        f"Significant genes after filtering: {len(sig)}"
    )


    # ========================================================
    # COMBINED
    # ========================================================

    if direction.lower() == "combined":

        # ----------------------------------------------------
        # Rank genes by p-value
        # ----------------------------------------------------

        sig = (
            sig
            .sort_values(PVAL_COLUMN)
            .drop_duplicates(
                subset=GENE_COLUMN
            )
        )


        # ----------------------------------------------------
        # Select top genes
        # ----------------------------------------------------

        genes = (
            sig[GENE_COLUMN]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .head(TOP_GENES)
            .tolist()
        )


        print(
            f"Genes sent to Enrichr: {len(genes)}"
        )


        print(
            f"{subtype_name}: BP"
        )


        results = run_go(
            genes
        )


        plot_go(
            results,
            title=subtype_name
        )


        return results


    # ========================================================
    # SPLIT
    # ========================================================

    elif direction.lower() == "split":

        up = sig[
            sig[LOGFC_COLUMN] > 0
        ].copy()

        down = sig[
            sig[LOGFC_COLUMN] < 0
        ].copy()


        # ----------------------------------------------------
        # Rank UPREGULATED genes by p-value
        # ----------------------------------------------------

        up = (
            up
            .sort_values(PVAL_COLUMN)
            .drop_duplicates(
                subset=GENE_COLUMN
            )
        )


        up_genes = (
            up[GENE_COLUMN]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .head(TOP_GENES)
            .tolist()
        )


        # ----------------------------------------------------
        # Rank DOWNREGULATED genes by p-value
        # ----------------------------------------------------

        down = (
            down
            .sort_values(PVAL_COLUMN)
            .drop_duplicates(
                subset=GENE_COLUMN
            )
        )


        down_genes = (
            down[GENE_COLUMN]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .head(TOP_GENES)
            .tolist()
        )


        print(
            f"Upregulated genes sent to Enrichr: "
            f"{len(up_genes)}"
        )

        print(
            f"Downregulated genes sent to Enrichr: "
            f"{len(down_genes)}"
        )


        # ----------------------------------------------------
        # UPREGULATED GO
        # ----------------------------------------------------

        print(
            f"{subtype_name}: BP UP"
        )

        up_results = run_go(
            up_genes
        )


        # ----------------------------------------------------
        # DOWNREGULATED GO
        # ----------------------------------------------------

        print(
            f"{subtype_name}: BP DOWN"
        )

        down_results = run_go(
            down_genes
        )


        plot_go(
            up_results,
            down_results,
            subtype_name
        )


        return {
            "BP_up": up_results,
            "BP_down": down_results
        }


    else:

        raise ValueError(
            "direction must be either "
            "'split' or 'combined'"
        )


# ============================================================
# BUILD GO MATRIX
# ============================================================

def build_go_matrix(
    go_results,
    ontology="BP",
    n_terms=20,
    min_subtypes=3
):

    # --------------------------------------------------------
    # Clean GO terms
    # --------------------------------------------------------

    subtype_terms = {}


    for subtype, res in go_results.items():

        df = res[ontology].copy()

        if df.empty:
            subtype_terms[subtype] = df
            continue

        df["Term"] = (
            df["Term"]
            .str.replace(
                r"\s*\(GO:\d+\)",
                "",
                regex=True
            )
        )

        subtype_terms[subtype] = df


    # --------------------------------------------------------
    # Count subtype occurrence
    # --------------------------------------------------------

    term_counts = {}


    for df in subtype_terms.values():

        for term in df["Term"].unique():

            term_counts[term] = (
                term_counts.get(term, 0) + 1
            )


    candidate_terms = [
        term
        for term, count in term_counts.items()
        if count >= min_subtypes
    ]


    if len(candidate_terms) == 0:

        raise ValueError(
            f"No GO terms appear in at least "
            f"{min_subtypes} subtypes."
        )


    # --------------------------------------------------------
    # Rank by best adjusted p-value
    # --------------------------------------------------------

    scores = {}


    for term in candidate_terms:

        best_p = np.inf


        for df in subtype_terms.values():

            hit = df[
                df["Term"] == term
            ]


            if not hit.empty:

                best_p = min(
                    best_p,
                    hit["Adjusted P-value"].iloc[0]
                )


        scores[term] = best_p


    top_terms = (
        pd.Series(scores)
        .sort_values()
        .head(n_terms)
        .index
    )


    # --------------------------------------------------------
    # Build matrix
    # --------------------------------------------------------

    matrix = pd.DataFrame(
        0.0,
        index=top_terms,
        columns=subtype_terms.keys(),
        dtype=float
    )


    for subtype, df in subtype_terms.items():

        for term in top_terms:

            hit = df[
                df["Term"] == term
            ]


            if not hit.empty:

                matrix.loc[
                    term,
                    subtype
                ] = -np.log10(
                    hit[
                        "Adjusted P-value"
                    ].iloc[0]
                )


    return matrix


# ============================================================
# GO COMPARISON PLOT
# ============================================================

def plot_go_comparison(
    matrix,
    title
):

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    y = np.arange(
        len(matrix)
    )

    width = 0.18


    for i, subtype in enumerate(
        matrix.columns
    ):

        ax.barh(
            y + (i - 1.5) * width,
            matrix[subtype],
            height=width,
            label=subtype
        )


    ax.set_yticks(y)

    ax.set_yticklabels(
        matrix.index,
        fontsize=10
    )

    ax.invert_yaxis()


    ax.set_xlabel(
        r"$-\log_{10}(\mathrm{Adjusted}\ p)$"
    )


    ax.legend(
        frameon=False
    )


    ax.set_title(
        title
    )


    plt.tight_layout()

    plt.show()