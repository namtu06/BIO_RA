
import os
import GOenrichment as ge


import importlib
importlib.reload(ge)


files = {
    "Proneural": "../data/interim/proneural_rna_diff.csv",
    "Classical": "../data/interim/classical_rna_diff.csv",
    "Mesenchymal": "../data/interim/mesenchymal_rna_diff.csv"
}

PATHWAY_DIR = "../data/processed/pathway_diff"
OXPHOS_DIR = "../data/processed/oxphos"

PVAL = "pvalue_rna"
LOGFC = "log2FC_rna"
GENE = ge.GENE_COLUMN

pathways = [
    "Apoptosis",
    "Autophagy",
    "Ferroptosis",
    "Necroptosis",
    "Pyroptosis"
]

complexes = [
    "complex_i",
    "complex_ii",
    "complex_iii",
    "complex_iv",
    "complex_v"
]

# ==========================================================
# GO enrichment
# ==========================================================

go_combined_by_subtype = {}

for subtype, file in files.items():

    go_combined_by_subtype[subtype] = ge.analyze_subtype(
        file,
        subtype,
        direction="combined"
    )



# ==========================================================
# Build heatmap gene sets
# (GO genes ∩ pathway genes ∩ significant DEGs)
# ==========================================================

heatmap_gene_sets = {}

for subtype in ["Proneural", "Classical", "Mesenchymal"]:

    bp = go_combined_by_subtype[subtype]["BP"]

    go_genes = set()

    for _, row in bp.iterrows():

        go_genes.update(
            g.upper()
            for g in row["Genes"].split(";")
        )

    folder = subtype.lower()

    for pathway in pathways:

        pathway_file = os.path.join(
            PATHWAY_DIR,
            folder,
            f"{folder}_{pathway}_diff.csv"
        )

        df = pd.read_csv(
            pathway_file,
            index_col=0
        )

        df.index = (
            df.index
            .astype(str)
            .str.upper()
        )

        sig = df[
            (df[PVAL] < ge.PVAL_THRESHOLD) &
            (abs(df[LOGFC]) > ge.LOGFC_THRESHOLD)
        ]

        heatmap_gene_sets[(subtype, pathway)] = (
            set(sig.index) & go_genes
        )

############################################################
# Create STRING input files
############################################################

for subtype in ["Proneural", "Classical", "Mesenchymal"]:

    folder = subtype.lower()

    #########################################
    # Heatmap genes (union of all pathways)
    #########################################

    heatmap_genes = set()

    for pathway in pathways:

        heatmap_genes |= heatmap_gene_sets[
            (subtype, pathway)
        ]

    #########################################
    # Significant OXPHOS genes
    #########################################

    oxphos_genes = set()

    for complex_name in complexes:

        file = os.path.join(
            OXPHOS_DIR,
            folder,
            f"{folder}_{complex_name}.csv"
        )

        df = pd.read_csv(
            file,
            index_col=0
        )

        df.index = (
            df.index
            .astype(str)
            .str.upper()
        )

        sig = df[
            (df[PVAL] < ge.PVAL_THRESHOLD) &
            (abs(df[LOGFC]) > ge.LOGFC_THRESHOLD)
        ]

        oxphos_genes |= set(sig.index)

    #########################################
    # Union
    #########################################

    upload_genes = sorted(
        heatmap_genes | oxphos_genes
    )

    print(
        f"{subtype}: "
        f"{len(heatmap_genes)} heatmap genes, "
        f"{len(oxphos_genes)} OXPHOS genes, "
        f"{len(upload_genes)} total"
    )

    #########################################
    # Save for STRING
    #########################################

    output = f"../data/interim/STRING_input/{folder}_sig_STRING_input.txt"

    with open(output, "w") as f:

        for gene in upload_genes:

            f.write(gene + "\n")

    print(f"Saved to {output}")