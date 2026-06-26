import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gseapy as gp

# -----------------------------
# USER SETTINGS
# -----------------------------
PVAL_COLUMN = "adj.P.Val"      # change if necessary
LOGFC_COLUMN = "logFC"         # change if necessary
GENE_COLUMN = "Gene.symbol"    # change if necessary

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

    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=gene_set,
        organism="Human",
        outdir=None
    )

    df = enr.results.copy()

    df = df.sort_values("Adjusted P-value")

    return df.head(TOP_N)



def plot_go(go_results, title, outfile=None):

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9, 12),
        constrained_layout=True
    )

    colors = {
        "BP": "#355C8C",
        "CC": "#8B4513",
        "MF": "#556B2F"
    }

    for ax, ontology in zip(axes, ["BP", "CC", "MF"]):

        df = go_results[ontology].copy()

        df = df.iloc[::-1]

        df["score"] = -np.log10(df["Adjusted P-value"])

        ax.barh(
            df["Term"],
            df["score"],
            color=colors[ontology]
        )

        ax.set_title(ontology)

        ax.set_xlabel("-log10(adj. p-value)")

    fig.suptitle(title, fontsize=18)

    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")

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
        f"{subtype_name} (Upregulated)",
        outfile=f"{subtype_name}_GO_UP.png"
    )

    plot_go(
        down_results,
        f"{subtype_name} (Downregulated)",
        outfile=f"{subtype_name}_GO_DOWN.png"
    )