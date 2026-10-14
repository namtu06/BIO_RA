import numpy as np
import pandas as pd

def make_ranking(subtype):
    results = pd.read_csv(f"../data/processed/{subtype}_combined.csv",index_col=0)
    
    ranking = (
    np.sign(results["log2FC_rna"])
    * results["-log(adjustedp)_rna"])

    ranking.name = "score"
    
    rnk = ranking.sort_values(
    ascending=False
    ).reset_index()

    rnk.columns = ["gene", "score"]
    
    return rnk