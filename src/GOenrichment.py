import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gseapy as gp
import textwrap

# -----------------------------
# USER SETTINGS
# -----------------------------
PVAL_COLUMN = "pvalue"      # change if necessary
LOGFC_COLUMN = "log2FC"         # change if necessary
GENE_COLUMN = "gene_name"    # change if necessary

PVAL_THRESHOLD = 0.05
LOGFC_THRESHOLD = 1          # set to 1 if you want |logFC| > 1

TOP_N = 20

# GO libraries
GO_LIBRARIES = {
    "BP": "GO_Biological_Process_2023",
    "CC": "GO_Cellular_Component_2023",
    "MF": "GO_Molecular_Function_2023"
}

def run_go(gene_list, gene_set):

    if len(gene_list) == 0:
        return pd.DataFrame()

    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=gene_set,
        organism="human",
        outdir=None
    )

    df = enr.results.copy()

    df = (
        df.sort_values("Adjusted P-value")
          .query("`Adjusted P-value` < 0.05")
          .reset_index(drop=True)
    )

    return df



def plot_go(up_results, down_results=None, title=""):

    colors = {
        "BP": "#355C8C",
        "CC": "#A35A19",
        "MF": "#5C7732"
    }

    ontologies = ["BP", "CC", "MF"]

    ############################################################
    # COMBINED MODE
    ############################################################
    if down_results is None:

        fig = plt.figure(figsize=(12, 12))

        gs = gridspec.GridSpec(
            3,
            3,
            width_ratios=[0.75, 0.80, 0.06],
            hspace=0.45,
            wspace=0.02
        )

        for row, ontology in enumerate(ontologies):

            ax_text = fig.add_subplot(gs[row, 0])
            ax_bar = fig.add_subplot(gs[row, 1])
            ax_type = fig.add_subplot(gs[row, 2])

            ax_type.axis("off")
            ax_type.text(
                0.5,
                0.5,
                ontology,
                rotation=270,
                fontsize=18,
                fontweight="bold",
                ha="center",
                va="center"
            )

            df = up_results[ontology].copy()

            if df.empty:
                ax_text.axis("off")
                ax_bar.axis("off")
                continue

            df = df.sort_values("Adjusted P-value").head(TOP_N)

            df["Term"] = (
                df["Term"]
                .str.replace(r"\s*\(GO:\d+\)", "", regex=True)
            )

            score = -np.log10(df["Adjusted P-value"])

            y = np.arange(len(df))

            ########################################
            # TEXT
            ########################################

            ax_text.set_xlim(0, 1)
            ax_text.set_ylim(-0.5, len(df)-0.5)
            ax_text.invert_yaxis()

            for yy, term in zip(y, df["Term"]):
                ax_text.text(
                    0.99,
                    yy,
                    term,
                    ha="right",
                    va="center",
                    fontsize=10
                )

            ax_text.axis("off")

            ########################################
            # BAR
            ########################################

            ax_bar.barh(
                y,
                score,
                color=colors[ontology],
                height=0.72
            )

            ax_bar.set_yticks([])
            ax_bar.invert_yaxis()

            ax_bar.set_xlabel(
                r"$-\log_{10}(\mathrm{Adjusted}\ p)$",
                fontsize=10
            )

            if row == 0:
                ax_bar.set_title("Combined", fontsize=20)

            ax_bar.spines["top"].set_visible(False)
            ax_bar.spines["right"].set_visible(False)

            pos = ax_bar.get_position()

            ax_bar.set_position([
                pos.x0 + 0.01,
                pos.y0,
                pos.width * 0.78,
                pos.height
            ])

        fig.suptitle(title, fontsize=24, fontweight="bold")

        plt.show()
        return

    ############################################################
    # SPLIT MODE (YOUR ORIGINAL PLOT)
    ############################################################

    fig = plt.figure(figsize=(20, 12))

    gs = gridspec.GridSpec(
        3,
        5,
        width_ratios=[0.75, 0.80, 0.75, 0.80, 0.06],
        hspace=0.45,
        wspace=0.02
    )

    for row, ontology in enumerate(ontologies):

        ax_text_up = fig.add_subplot(gs[row, 0])
        ax_bar_up = fig.add_subplot(gs[row, 1])

        ax_text_down = fig.add_subplot(gs[row, 2])
        ax_bar_down = fig.add_subplot(gs[row, 3])

        ax_type = fig.add_subplot(gs[row, 4])

        ax_type.axis("off")
        ax_type.text(
            0.5,
            0.5,
            ontology,
            rotation=270,
            fontsize=18,
            fontweight="bold",
            ha="center",
            va="center"
        )

        for results, ax_text, ax_bar, heading in [

            (up_results, ax_text_up, ax_bar_up, "Upregulated"),
            (down_results, ax_text_down, ax_bar_down, "Downregulated")

        ]:

            df = results[ontology].copy()

            if df.empty:
                ax_text.axis("off")
                ax_bar.axis("off")
                continue

            df = df.sort_values("Adjusted P-value").head(TOP_N)

            df["Term"] = (
                df["Term"]
                .str.replace(r"\s*\(GO:\d+\)", "", regex=True)
            )

            score = -np.log10(df["Adjusted P-value"])

            y = np.arange(len(df))

            ax_text.set_xlim(0, 1)
            ax_text.set_ylim(-0.5, len(df)-0.5)
            ax_text.invert_yaxis()

            for yy, term in zip(y, df["Term"]):
                ax_text.text(
                    0.99,
                    yy,
                    term,
                    ha="right",
                    va="center",
                    fontsize=10
                )

            ax_text.axis("off")

            ax_bar.barh(
                y,
                score,
                color=colors[ontology],
                height=0.72
            )

            ax_bar.set_yticks([])
            ax_bar.invert_yaxis()

            ax_bar.set_xlabel(
                r"$-\log_{10}(\mathrm{Adjusted}\ p)$",
                fontsize=10
            )

            if row == 0:
                ax_bar.set_title(heading, fontsize=20)

            ax_bar.spines["top"].set_visible(False)
            ax_bar.spines["right"].set_visible(False)

            pos = ax_bar.get_position()

            ax_bar.set_position([
                pos.x0 + 0.01,
                pos.y0,
                pos.width * 0.78,
                pos.height
            ])

    fig.suptitle(title, fontsize=24, fontweight="bold")

    plt.show()

def analyze_subtype(
    csv_file,
    subtype_name,
    direction="combined",
    gene_filter=None,
    filter_before_significance=False
):

    df = pd.read_csv(csv_file)

    if gene_filter is not None and filter_before_significance:
        # Restrict to the shared/intersection gene universe before DE filtering.
        df = df[
            df[GENE_COLUMN].fillna("").str.upper().isin(gene_filter)
        ]

    sig = df[
        (df[PVAL_COLUMN] < PVAL_THRESHOLD)
        &
        (abs(df[LOGFC_COLUMN]) > LOGFC_THRESHOLD)
    ]

    if gene_filter is not None and not filter_before_significance:
        sig = sig[
        sig[GENE_COLUMN].fillna("").str.upper().isin(gene_filter)
        ]
    
    print(f"\n{subtype_name}")
    print(f"Significant genes after filtering: {len(sig)}")
    
    if direction.lower() == "combined":

        genes = sig[GENE_COLUMN].dropna().unique().tolist()

        results = {}

        for ontology, library in GO_LIBRARIES.items():

            print(f"{subtype_name}: {ontology}")

            results[ontology] = run_go(
                genes,
                library
            )

        plot_go(results, title=subtype_name)

        return results

    elif direction.lower() == "split":

        up = sig[sig[LOGFC_COLUMN] > 0]
        down = sig[sig[LOGFC_COLUMN] < 0]

        up_genes = up[GENE_COLUMN].dropna().unique().tolist()
        down_genes = down[GENE_COLUMN].dropna().unique().tolist()

        up_results = {}
        down_results = {}

        for ontology, library in GO_LIBRARIES.items():

            print(f"{subtype_name}: {ontology} UP")
            up_results[ontology] = run_go(
                up_genes,
                library
            )

            print(f"{subtype_name}: {ontology} DOWN")
            down_results[ontology] = run_go(
                down_genes,
                library
            )

        plot_go(
            up_results,
            down_results,
            subtype_name
        )

        return {
            "BP_up": up_results["BP"],
            "BP_down": down_results["BP"],
            "CC_up": up_results["CC"],
            "CC_down": down_results["CC"],
            "MF_up": up_results["MF"],
            "MF_down": down_results["MF"],
        }

    else:
        raise ValueError(
            "direction must be either 'split' or 'combined'"
        )


def build_go_matrix(
    go_results,
    ontology="BP",
    n_terms=20,
    min_subtypes=3
):

    # -----------------------------
    # Clean GO terms
    # -----------------------------
    subtype_terms = {}

    for subtype, res in go_results.items():

        df = res[ontology].copy()

        df["Term"] = (
            df["Term"]
            .str.replace(r"\s*\(GO:\d+\)", "", regex=True)
        )

        subtype_terms[subtype] = df

    # -----------------------------
    # Count how many subtypes each term appears in
    # -----------------------------
    term_counts = {}

    for df in subtype_terms.values():

        for term in df["Term"].unique():

            term_counts[term] = term_counts.get(term, 0) + 1

    # Keep only terms appearing in enough subtypes
    candidate_terms = [
        term
        for term, count in term_counts.items()
        if count >= min_subtypes
    ]

    if len(candidate_terms) == 0:
        raise ValueError(
            f"No GO terms appear in at least {min_subtypes} subtypes."
        )

    # -----------------------------
    # Rank by best adjusted p-value
    # -----------------------------
    scores = {}

    for term in candidate_terms:

        best_p = np.inf

        for df in subtype_terms.values():

            hit = df[df["Term"] == term]

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

    # -----------------------------
    # Build matrix
    # -----------------------------
    matrix = pd.DataFrame(
        0.0,
        index=top_terms,
        columns=subtype_terms.keys(),
        dtype=float
    )

    for subtype, df in subtype_terms.items():

        for term in top_terms:

            hit = df[df["Term"] == term]

            if not hit.empty:

                matrix.loc[term, subtype] = -np.log10(
                    hit["Adjusted P-value"].iloc[0]
                )

    return matrix


def plot_go_comparison(matrix, title):

    colors = [
        "#4E79A7",
        "#F28E2B",
        "#59A14F",
        "#E15759"
    ]

    fig, ax = plt.subplots(figsize=(10, 10))

    y = np.arange(len(matrix))

    width = 0.18

    for i, subtype in enumerate(matrix.columns):

        ax.barh(
            y + (i-1.5)*width,
            matrix[subtype],
            height=width,
            label=subtype,
            color=colors[i]
        )

    ax.set_yticks(y)
    ax.set_yticklabels(matrix.index, fontsize=10)

    ax.invert_yaxis()

    ax.set_xlabel(r"$-\log_{10}(\mathrm{Adjusted}\ p)$")

    ax.legend(frameon=False)

    ax.set_title(title)

    plt.tight_layout()

    plt.show()