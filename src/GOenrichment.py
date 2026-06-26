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

TOP_N = 10

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

    df = df.sort_values("Adjusted P-value")

    return df.head(TOP_N)



def plot_go(up_results, down_results, title):

    fig = plt.figure(figsize=(20, 12))

    # More space for labels, less space for bars
    gs = gridspec.GridSpec(
        3,
        5,
        width_ratios=[0.75, 0.80, 0.75, 0.80, 0.06],
        hspace=0.45,
        wspace=0.02
    )

    colors = {
        "BP": "#355C8C",
        "CC": "#A35A19",
        "MF": "#5C7732"
    }

    ontologies = ["BP", "CC", "MF"]

    for row, ontology in enumerate(ontologies):

        ##############################
        # UP
        ##############################

        ax_text_up = fig.add_subplot(gs[row, 0])
        ax_bar_up = fig.add_subplot(gs[row, 1])

        ##############################
        # DOWN
        ##############################

        ax_text_down = fig.add_subplot(gs[row, 2])
        ax_bar_down = fig.add_subplot(gs[row, 3])

        ##############################
        # BP / CC / MF LABEL
        ##############################

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

            df = df.sort_values("Adjusted P-value")

            df["Term"] = (
                df["Term"]
                .str.replace(r"\s*\(GO:\d+\)", "", regex=True)
                )

            score = -np.log10(df["Adjusted P-value"])

            y = np.arange(len(df))

            ################################################
            # LABEL COLUMN
            ################################################

            ax_text.set_xlim(0, 1)
            ax_text.set_ylim(-0.5, len(df) - 0.5)
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

            ################################################
            # BAR COLUMN
            ################################################

            ax_bar.barh(
                y,
                score,
                color=colors[ontology],
                height=0.72
            )

            ax_bar.set_yticks([])
            ax_bar.invert_yaxis()

            ax_bar.set_xlabel(r"$-\log_{10}(\mathrm{Adjusted}\ p)$", fontsize=10)

            if row == 0:
                ax_bar.set_title(heading, fontsize=20)

            ax_bar.spines["top"].set_visible(False)
            ax_bar.spines["right"].set_visible(False)

            # Shrink plotting area to leave whitespace like the paper
            pos = ax_bar.get_position()
            ax_bar.set_position([
                pos.x0 + 0.01,
                pos.y0,
                pos.width * 0.78,
                pos.height
            ])

    fig.suptitle(title, fontsize=24, fontweight="bold")

    plt.show()

def analyze_subtype(csv_file, subtype_name):

    df = pd.read_csv(csv_file)

    sig = df[
        (df[PVAL_COLUMN] < PVAL_THRESHOLD)
        &
        (abs(df[LOGFC_COLUMN]) > LOGFC_THRESHOLD)
    ]

    up = sig[sig[LOGFC_COLUMN] > 0]
    down = sig[sig[LOGFC_COLUMN] < 0]

    up_genes = up[GENE_COLUMN].dropna().unique().tolist()
    down_genes = down[GENE_COLUMN].dropna().unique().tolist()

    up_results = {}
    down_results = {}

    for ontology, library in GO_LIBRARIES.items():

        print(f"{subtype_name}: {ontology} UP")
        up_results[ontology] = run_go(up_genes, library)

        print(f"{subtype_name}: {ontology} DOWN")
        down_results[ontology] = run_go(down_genes, library)

    plot_go(
    up_results,
    down_results,
    subtype_name
    )