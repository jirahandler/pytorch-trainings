#!/usr/bin/env python3
"""
Process signal and background ntuples to create balanced discriminant ntuples (S:B = 1:1).
Reads 'sel_tree' from each file, applies model, and writes one ROOT per signal mass-point and
one balanced background ROOT with equal total entries.
"""
import os
import uproot
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
NTUPLE_BASE_PATH = "/home/sgoswami/monobcntuples/run3_btag/all"
MODEL_PATH       = "/home/sgoswami/monobcntuples/ML/best_model_stop.keras"
SUFFIX           = "_stopana"
BRANCH           = f"discriminant{SUFFIX}"
FEATURES = [
    "jet1_pt", "jet1_eta", "jet1_svmass", "jet1met_dphi", "jet2met_dphi",
    "jet3met_dphi", "jet4met_dphi", "met_pt", "met_phi", "met_sig", "mjj",
    "pTjj", "mbb", "pTbb", "dRjj", "dEtajj", "dPhijj", "dRbb", "dEtabb",
    "dPhibb", "jet2_pt", "jet2_eta", "jet2_svmass"
]

# Signal and background definitions
signal_files = {
    "sT_tN1_1000_102_100": os.path.join(NTUPLE_BASE_PATH, "singlestop", "basicSel_sT_tN1_1000_102_100.root"),
    "sT_tN1_1000_202_200": os.path.join(NTUPLE_BASE_PATH, "singlestop", "basicSel_sT_tN1_1000_202_200.root"),
    "sT_tN1_1000_3_1":     os.path.join(NTUPLE_BASE_PATH, "singlestop", "basicSel_sT_tN1_1000_3_1.root"),
}
background_files = {
    "diboson":   os.path.join(NTUPLE_BASE_PATH, "diboson",   "basicSel_diboson.root"),
    "dijet":     os.path.join(NTUPLE_BASE_PATH, "dijet",     "basicSel_dijet.root"),
    "singletop": os.path.join(NTUPLE_BASE_PATH, "singletop", "basicSel_singletop.root"),
    "ttbar":     os.path.join(NTUPLE_BASE_PATH, "ttbar",     "basicSel_ttbar.root"),
    "wlnu":      os.path.join(NTUPLE_BASE_PATH, "wlnu",      "basicSel_wlnu.root"),
    "zll":       os.path.join(NTUPLE_BASE_PATH, "zll",       "basicSel_zll.root"),
    "znunu":     os.path.join(NTUPLE_BASE_PATH, "znunu",     "basicSel_znunu.root"),
}

def load_df(path):
    """Load features from 'sel_tree', return pandas DataFrame (or None)."""
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return None
    with uproot.open(path) as f:
        if "sel_tree" not in f:
            print(f"WARNING: sel_tree missing in {path}")
            return None
        tree = f["sel_tree"]
        # check features
        branches = set(tree.keys())
        missing = [b for b in FEATURES if b not in branches]
        if missing:
            print(f"WARNING: missing {missing} in {path}")
            return None
        return tree.arrays(FEATURES, library="pd")

# --- Main processing ---
# 1) Load all DataFrames and record indices
dfs = []
indices = {}
pos = 0
for name, path in {**signal_files, **background_files}.items():
    df = load_df(path)
    if df is None or df.empty:
        continue
    n = len(df)
    dfs.append(df)
    indices[name] = (pos, pos + n)
    pos += n
if not dfs:
    raise RuntimeError("No data loaded")
combined = pd.concat(dfs, ignore_index=True)

# 2) Scale features
df_features = combined[FEATURES]
scaler = StandardScaler()
X_all = scaler.fit_transform(df_features)

# 3) Predict with model
model = tf.keras.models.load_model(MODEL_PATH)
preds = model.predict(X_all, batch_size=4096).flatten()

# 4) Write per-signal ntuples
total_signal = 0
for name in signal_files:
    if name not in indices:
        continue
    start, end = indices[name]
    arr = preds[start:end]
    total_signal += len(arr)
    out_file = f"discriminant_{name}{SUFFIX}.root"
    with uproot.recreate(out_file) as out:
        out[f"{name}{SUFFIX}"] = {BRANCH: arr}
    print(f"Wrote {out_file}: {len(arr)} events")

# 5) Build balanced background (S:B = 1:1)
bkg_preds = []
for name in background_files:
    if name in indices:
        s,e = indices[name]
        bkg_preds.append(preds[s:e])
if not bkg_preds:
    raise RuntimeError("No background predictions loaded")
combined_bkg = np.concatenate(bkg_preds)
# sample to match total_signal
rng = np.random.default_rng(42)
if len(combined_bkg) >= total_signal:
    bkg_sampled = rng.choice(combined_bkg, size=total_signal, replace=False)
else:
    bkg_sampled = rng.choice(combined_bkg, size=total_signal, replace=True)
out_bkg = f"discriminant_background{SUFFIX}.root"
with uproot.recreate(out_bkg) as out:
    out[f"background{SUFFIX}"] = {BRANCH: bkg_sampled}
print(f"Wrote {out_bkg}: {len(bkg_sampled)} balanced events")

print("Done: generated balanced discriminant ntuples.")
