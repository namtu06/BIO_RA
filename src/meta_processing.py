import pandas as pd

general_df2 = pd.read_csv("../data/processed/metadata.csv")
general_df2 = general_df2.set_index("id")

labels = general_df2[general_df2["WHO grade"].isin(["III", "III with necrosis", "III without necrosis","IV", "IV with necrosis", "IV without necrosis"])]

typeIII = general_df2[general_df2["WHO grade"].isin(["III", "III with necrosis", "III without necrosis"])]
III_primary = typeIII[typeIII["Primary/Recurrent"] == "primary"]
III_recurrent = typeIII[typeIII["Primary/Recurrent"] == "recurrent"]
print(f"Number of primary type III samples: {len(III_primary.index)}")
print(f"Number of recurrent type III samples: {len(III_recurrent.index)}")




typeIV = general_df2[general_df2["WHO grade"].isin(["IV", "IV with necrosis", "IV without necrosis"])]
IV_primary = typeIV[typeIV["Primary/Recurrent"] == "primary"]
IV_recurrent = typeIV[typeIV["Primary/Recurrent"] == "recurrent"]
print(f"Number of primary type IV samples: {len(IV_primary.index)}")
print(f"Number of recurrent type IV samples: {len(IV_recurrent.index)}")

print("\n")

# Primary
print("PRIMARY III")
III_prim_PN = (III_primary[III_primary["Subtype"] == "PN"])
III_prim_Mes = (III_primary[III_primary["Subtype"] == "Mes"])
III_prim_Prolif = (III_primary[III_primary["Subtype"] == "Prolif"])
print(f"PN: {len(III_prim_PN.index)}")
print(f"Mes: {len(III_prim_Mes.index)}")
print(f"Prolif: {len(III_prim_Prolif.index)}")

print("\n")
# Recurrent
print("RECURRENT III")
III_rec_PN = (III_recurrent[III_recurrent["Subtype"] == "PN"])
III_rec_Mes = (III_recurrent[III_recurrent["Subtype"] == "Mes"])
III_rec_Prolif = (III_recurrent[III_recurrent["Subtype"] == "Prolif"])
print(f"PN: {len(III_rec_PN.index)}")
print(f"Mes: {len(III_rec_Mes.index)}")
print(f"Prolif: {len(III_rec_Prolif.index)}")

print("\n")
# Primary
print("PRIMARY IV")
IV_prim_PN = (IV_primary[IV_primary["Subtype"] == "PN"])
IV_prim_Mes = (IV_primary[IV_primary["Subtype"] == "Mes"])
IV_prim_Prolif = (IV_primary[IV_primary["Subtype"] == "Prolif"])
print(f"PN: {len(IV_prim_PN.index)}")
print(f"Mes: {len(IV_prim_Mes.index)}")
print(f"Prolif: {len(IV_prim_Prolif.index)}")

print("\n")
# Recurrent
print("RECURRENT IV")
IV_rec_PN = (IV_recurrent[IV_recurrent["Subtype"] == "PN"])
IV_rec_Mes = (IV_recurrent[IV_recurrent["Subtype"] == "Mes"])
IV_rec_Prolif = (IV_recurrent[IV_recurrent["Subtype"] == "Prolif"])
print(f"PN: {len(IV_rec_PN.index)}")
print(f"Mes: {len(IV_rec_Mes.index)}")
print(f"Prolif: {len(IV_rec_Prolif.index)}")