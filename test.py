# Cell 1: imports and settings
import GEOparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

sns.set(style="whitegrid")
pd.set_option('display.max_columns', 40)

# Cell 2: fetch GEO
gse = GEOparse.get_GEO("GSE4271", destdir="data/", silent=True)
gsm0 = list(gse.gsms.values())[0]
print("Data processing (first GSM):", gsm0.metadata.get("data_processing", ["N/A"]))

# Cell 3: build expression matrix (probe x sample)
expr_dict = {}
for gsm_name, gsm in gse.gsms.items():
    vals = pd.to_numeric(gsm.table.set_index("ID_REF")["VALUE"], errors="coerce")
    expr_dict[gsm_name] = vals
expression_df = pd.DataFrame(expr_dict)
expression_df.index.name = "ID_REF"
print("Expression shape:", expression_df.shape)
print(expression_df.stack().quantile([0, 0.25, 0.5, 0.75, 1.0]))

# Cell 4: check/perform log2 (only if needed)
max_val = expression_df.max().max()
if max_val > 100:  # heuristic: raw intensities large
    print(f"Max {max_val:.1f} suggests non-log data -> applying log2(x+1)")
    expression_df = np.log2(expression_df + 1)
else:
    print(f"Max {max_val:.1f} suggests data already log2 (no transform)")

# Cell 5: build probe->gene mapping from GPLs
gpl96 = gse.gpls.get("GPL96")
gpl97 = gse.gpls.get("GPL97")
gpl_frames = []
if gpl96 is not None:
    gpl_frames.append(gpl96.table[["ID", "Gene Symbol"]])
if gpl97 is not None:
    gpl_frames.append(gpl97.table[["ID", "Gene Symbol"]])
gene_map = pd.concat(gpl_frames)
gene_map = gene_map.rename(columns={"ID": "ID_REF"}).set_index("ID_REF")
gene_map["Gene Symbol"] = gene_map["Gene Symbol"].astype(str).str.split(" /// ")
gene_map = gene_map.explode("Gene Symbol")
gene_map["Gene Symbol"] = gene_map["Gene Symbol"].str.strip()
gene_map = gene_map.replace({"Gene Symbol": {"nan": np.nan}}).dropna(subset=["Gene Symbol"])
print("Probe->gene mappings:", gene_map.shape)

# Cell 6: sample metadata and group assignment
meta = []
for gsm_name, gsm in gse.gsms.items():
    chars = gsm.metadata.get("characteristics_ch1", [])
    d = {"sample": gsm_name}
    for c in chars:
        if ":" in c:
            k, v = c.split(":", 1)
            d[k.strip()] = v.strip()
    d["title"] = gsm.metadata.get("title", [""])[0]
    meta.append(d)
meta_df = pd.DataFrame(meta).set_index("sample")
print("metadata columns:", meta_df.columns.tolist())
if 'specimen type' in meta_df.columns:
    primary_samples = meta_df[meta_df['specimen type'] == 'primary'].index.tolist()
    recurrent_samples = meta_df[meta_df['specimen type'] == 'recurrent'].index.tolist()
else:
    mask_primary = meta_df.apply(lambda row: row.astype(str).str.contains('primary', case=False).any(), axis=1)
    mask_recurrent = meta_df.apply(lambda row: row.astype(str).str.contains('recurrent', case=False).any(), axis=1)
    primary_samples = meta_df[mask_primary].index.tolist()
    recurrent_samples = meta_df[mask_recurrent].index.tolist()

print("primary samples:", len(primary_samples), "recurrent samples:", len(recurrent_samples))
assert len(primary_samples) >= 2 and len(recurrent_samples) >= 2, "Not enough samples per group; check metadata parsing"

# Cell 7: subset expression to used samples and map probes to genes
expr_sub = expression_df[primary_samples + recurrent_samples].copy()
expr_with_gene = expr_sub.merge(gene_map[["Gene Symbol"]], left_index=True, right_index=True, how="left")
expr_with_gene = expr_with_gene.dropna(subset=["Gene Symbol"])
print("Probes with gene symbol:", expr_with_gene.shape)

# Cell 8: collapse probes to gene-level by selecting probe with highest IQR per gene
iqr = expr_with_gene.iloc[:, :-1].apply(lambda row: np.nanpercentile(row.dropna(), 75) - np.nanpercentile(row.dropna(), 25), axis=1)
expr_with_gene["IQR"] = iqr
best_probes = expr_with_gene.reset_index().sort_values(["Gene Symbol", "IQR"], ascending=[True, False]).groupby("Gene Symbol").first()
gene_expr = best_probes.drop(columns=["Gene Symbol", "IQR"]).set_index(best_probes.index)
gene_expr.index.name = "Gene Symbol"
print("Gene-level expression shape:", gene_expr.shape)

# Cell 9: filter genes (low expression and low variance)
mean_expr = gene_expr.mean(axis=1)
var_expr = gene_expr.var(axis=1, ddof=1)
expr_mask = mean_expr > 4.0  # tune if needed
var_cut = np.percentile(var_expr.dropna(), 20)
var_mask = var_expr > var_cut
keep = expr_mask & var_mask
print(f"Genes before filter: {gene_expr.shape[0]}, after filter: {keep.sum()}")
gene_expr_f = gene_expr.loc[keep]

# Cell 10: differential testing (Welch t-test), compute log2FC and p-values
group_a = primary_samples
group_b = recurrent_samples
log2_fc = gene_expr_f[group_b].mean(axis=1) - gene_expr_f[group_a].mean(axis=1)

pvals = []
tested = []
for g in gene_expr_f.index:
    a = gene_expr_f.loc[g, group_a].values
    b = gene_expr_f.loc[g, group_b].values
    if np.isfinite(a).sum() >= 2 and np.isfinite(b).sum() >= 2:
        _, pv = stats.ttest_ind(b, a, equal_var=False, nan_policy='omit')
        if np.isnan(pv):
            pv = 1.0
        pvals.append(pv)
        tested.append(g)
    else:
        continue

res = pd.DataFrame({
    "Gene Symbol": tested,
    "log2FC": log2_fc.loc[tested].values,
    "p_value": pvals
}).set_index("Gene Symbol")

res["p_adj"] = multipletests(res["p_value"].fillna(1.0).values, method="fdr_bh")[1]
res["minus_log10_p"] = -np.log10(res["p_value"].replace(0, 1e-300))
res["minus_log10_p_adj"] = -np.log10(res["p_adj"].replace(0, 1e-300))
res = res.sort_values("p_adj")
res.to_csv("data/GSE4271_differential_results.csv")
print(res.head())

# Cell 11: volcano + MA plots
plt.figure(figsize=(9,6))
plt.scatter(res["log2FC"], res["minus_log10_p_adj"], c='k', alpha=0.5, s=12)
sig = res["p_adj"] < 0.05
plt.scatter(res.loc[sig, "log2FC"], res.loc[sig, "minus_log10_p_adj"], c='red', s=18, alpha=0.8)
plt.xlabel("log2 fold change (Recurrent - Primary)")
plt.ylabel("-log10(adj p-value)")
plt.title("Volcano: Recurrent vs Primary (GSE4271)")
plt.axhline(-np.log10(0.05), color='gray', linestyle='--', linewidth=1)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/GSE4271_volcano.png", dpi=150)
plt.show()

avg = (gene_expr_f[group_a].mean(axis=1) + gene_expr_f[group_b].mean(axis=1)) / 2
plt.figure(figsize=(8,6))
plt.scatter(avg, log2_fc.loc[gene_expr_f.index], c='k', alpha=0.5, s=8)
plt.xlabel("Average expression (log2)")
plt.ylabel("log2FC (Recurrent - Primary)")
plt.title("MA plot")
plt.axhline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.savefig("data/GSE4271_MA.png", dpi=150)
plt.show()

# Cell 12: notes / next steps
# - If many high adjusted p-values persist: use limma/voom (R) or reduce testing set.
# - Tweak probe collapse and filters as needed.