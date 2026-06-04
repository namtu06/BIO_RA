import pandas as pd

def collapse_probes_to_genes(
    expr_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    probe_col: str = "ID",
    gene_col: str = "Gene Symbol"
) -> pd.DataFrame:
    """
    Convert probe-level expression matrix to gene-level expression matrix.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Expression matrix with probe IDs as index and samples as columns.

    mapping_df : pd.DataFrame
        Probe-to-gene mapping dataframe.

    probe_col : str
        Column containing probe IDs.

    gene_col : str
        Column containing gene symbols, potentially separated by '///'.

    Returns
    -------
    pd.DataFrame
        Gene-level expression matrix.
    """

    # Keep only relevant columns
    mapping = mapping_df[[probe_col, gene_col]].copy()

    # Split multi-gene mappings
    mapping[gene_col] = (
        mapping[gene_col]
        .fillna("")
        .str.split(r"\s*///\s*")
    )

    # Expand one probe -> many genes
    mapping = mapping.explode(gene_col)

    # Remove empty symbols
    mapping = mapping[
        (mapping[gene_col] != "")
        & mapping[gene_col].notna()
    ]

    # Join expression values
    expanded = mapping.merge(
        expr_df,
        left_on=probe_col,
        right_index=True,
        how="inner"
    )

    # Collapse probes to genes by maximum expression
    gene_expr = (
        expanded
        .groupby(gene_col)
        .max(numeric_only=True)
    )

    return gene_expr