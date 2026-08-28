# 🧬 Integrative Network Analysis of Mitochondrial Complexes and Glioblastoma Multiomics Data

> **Computational analysis of glioblastoma transcriptomics, biological processes, mitochondrial OXPHOS complexes, and protein-interaction networks.**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![PyDESeq2](https://img.shields.io/badge/PyDESeq2-Differential%20Expression-green)
![NetworkX](https://img.shields.io/badge/NetworkX-Network%20Analysis-red)
![STRING](https://img.shields.io/badge/STRING-Protein%20Interactions-purple)

---

## 🧠 Project Description

Glioblastoma (GBM) is an aggressive brain tumor with substantial molecular heterogeneity. We know that its aggressiveness stems from improper cell deaths, which are mainly maintained by the mitochondria and its activities. We also know that OXPHOS plays a major role in enabling those activities and performing mitochondrial homeostasis.

This project asks a more specific question:

> **How are apoptosis-related molecular processes connected to mitochondrial oxidative phosphorylation (OXPHOS) complexes in glioblastoma?**

Rather than looking at differential expression alone, the project combines several levels of analysis:

```text
                 🧬 GBM transcriptomics
                         │
                         ▼
                Differential expression
                      PyDESeq2
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        GO enrichment         OXPHOS genes
              │               Complex I–V
              |                     |
              |                     │
              └──────────┬──────────┘
                         ▼
                🔗 STRING network
                         │
                         ▼
              NetworkX analysis
                         │
                         ▼
             🧩 Apoptosis–OXPHOS
                  interactions



```
With home-made plotting for easier customization!

DISCLAIMER: All results currently displayed for Classical Subtype only, sorry for the inconvinience
# 🧬 1. Transcriptomic Differential Expression

```
CPTAC expression data
        ↓
Sample selection
        ↓
Count matrix preparation
        ↓
Sample metadata
        ↓
PyDESeq2
        ↓
log2FC + p-values
        ↓
Multiple-testing correction
        ↓
Significantly dysregulated genes
```
The differential-expression results provide the foundation for the downstream functional and network analyses.

<img src="https://github.com/namtu06/BIO_RA/blob/main/results/figures/TRANS_VOLCANO.png" alt="Volcano plot" width="400"/> <img src="https://github.com/namtu06/BIO_RA/blob/main/results/figures/TRANS_PCA.png" alt="PCA" width="445"/>


Figure: Differential expression between GBM and reference samples (Left). PCA with raw gene count (Right).

## Key metrics

| Metric            | Result |
| ----------------- | -----: |
| Significant genes |  `32067` |
| Upregulated       |  `20328` |
| Downregulated     |  `6949` |
| FDR threshold     |  `0.05` |
| Log2FC threshold  |  `1.00`  |

# 🧩 2. Gene Ontology Enrichment


Significantly dysregulated genes are investigated using Gene Ontology Biological Process enrichment.

The purpose is to determine which biological processes are overrepresented among the dysregulated genes.

Particular attention is given to processes associated with:

Cellular Respiration
Mitochondrial function
OXPHOS

📊 GO enrichment

<img src="https://github.com/namtu06/BIO_RA/blob/main/results/figures/GO_Enrich.png" alt="GO Enrichment" width="500"/>




OXPHOS-related GO Biological Process terms are highlighted to identify processes that can subsequently be investigated in the protein-interaction network.


# 🔗 3. Protein-Interaction Network Analysis

Differentially relevant genes are mapped to protein-interaction networks using STRING.

Network analysis is performed using NetworkX.

```
 OXPHOS-related genes from DE               Apoptosis-related genes
        │                                       |
        └────────────┼──────────────────────────┘
                     │
              Protein network
```

# To be continued...











