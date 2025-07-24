#!/usr/bin/env python3
import os
import uproot
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def load_features_from_file(path, features_list):
    """Load features_list from the 'sel_tree' TTree in the given ROOT file."""
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping")
        return None
    try:
        with uproot.open(path) as root_file:
            if "sel_tree" not in root_file:
                print(f"  WARNING: 'sel_tree' not found in {path}, skipping")
                return None
            tree = root_file["sel_tree"]
            branches = set(tree.keys())
            missing = [b for b in features_list if b not in branches]
            if missing:
                print(f"  WARNING: missing branches {missing} in {path}, skipping")
                return None
            return tree.arrays(features_list, library="pd")
    except Exception as e:
        print(f"  ERROR reading {path}: {e}")
        return None

def main():
    print("--- Starting NTuple processing for stop analysis with selected variables ---")

    NTUPLE_BASE_PATH = "/home/sgoswami/monobcntuples/run3_btag/all"
    MODEL_PATH       = "/home/sgoswami/monobcntuples/ML/best_model_stop.keras"
    suffix           = "_stopana"

    # --- Explicitly defined variables matching the model's input ---
    FEATURES = [
        "jet1_pt", "jet1_eta", "jet1_svmass", "jet1met_dphi", "jet2met_dphi",
        "jet3met_dphi", "jet4met_dphi", "met_pt", "met_phi", "met_sig", "mjj",
        "pTjj", "mbb", "pTbb", "dRjj", "dEtajj", "dPhijj", "dRbb", "dEtabb",
        "dPhibb", "jet2_pt", "jet2_eta", "jet2_svmass"
    ]
    print(f"Using {len(FEATURES)} selected features for model input.")

    # --- Define signal samples (only sT_tN1 variants) ---
    signal_files = {
        "sT_tN1_1000_102_100": os.path.join(NTUPLE_BASE_PATH, "singlestop", "basicSel_sT_tN1_1000_102_100.root"),
        "sT_tN1_1000_202_200": os.path.join(NTUPLE_BASE_PATH, "singlestop", "basicSel_sT_tN1_1000_202_200.root"),
        "sT_tN1_1000_3_1":     os.path.join(NTUPLE_BASE_PATH, "singlestop", "basicSel_sT_tN1_1000_3_1.root"),
    }
    # --- Background samples ---
    background_files = {
        "diboson":   os.path.join(NTUPLE_BASE_PATH, "diboson",   "basicSel_diboson.root"),
        "dijet":     os.path.join(NTUPLE_BASE_PATH, "dijet",     "basicSel_dijet.root"),
        "singletop": os.path.join(NTUPLE_BASE_PATH, "singletop", "basicSel_singletop.root"),
        "ttbar":     os.path.join(NTUPLE_BASE_PATH, "ttbar",     "basicSel_ttbar.root"),
        "wlnu":      os.path.join(NTUPLE_BASE_PATH, "wlnu",      "basicSel_wlnu.root"),
        "zll":       os.path.join(NTUPLE_BASE_PATH, "zll",       "basicSel_zll.root"),
        "znunu":     os.path.join(NTUPLE_BASE_PATH, "znunu",     "basicSel_znunu.root"),
    }
    all_samples = {**signal_files, **background_files}

    # --- Load model & verify input shape matches FEATURES ---
    print(f"\nLoading Keras model from {MODEL_PATH} …")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"FATAL: could not load model: {e}")
        return
    expected_dim = model.input_shape[1]
    if expected_dim != len(FEATURES):
        print(f"FATAL: model expects {expected_dim} features but got {len(FEATURES)}.")
        return
    print("Model input dimension matches selected features.")

    # --- Load & index data ---
    print("\nLoading all data to determine scaling parameters...")
    dfs, indices, pos = [], {}, 0
    for sample, path in all_samples.items():
        df = load_features_from_file(path, FEATURES)
        if df is None or df.empty:
            continue
        n = len(df)
        dfs.append(df)
        indices[sample] = (pos, pos + n)
        pos += n
    if not dfs:
        print("FATAL: no data loaded. Exiting.")
        return
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"  → Loaded {len(combined_df)} total events")

    # --- Scale features ---
    print("\nScaling input features...")
    scaler = StandardScaler()
    X_all = scaler.fit_transform(combined_df[FEATURES])

    # --- Predict ---
    print("\nRunning model predictions on scaled data…")
    preds = model.predict(X_all, batch_size=4096).flatten()

    # --- Write output ROOT ---
    parent = os.path.basename(os.path.dirname(NTUPLE_BASE_PATH))
    output_file = f"{parent}{suffix}.root"
    print(f"\nWriting discriminants to {output_file}")
    with uproot.recreate(output_file) as out:
        for sample, (start, end) in indices.items():
            arr = preds[start:end]
            tree_name = f"{sample}{suffix}"
            out[tree_name] = {f"discriminant{suffix}": arr}
            print(f"  • Wrote {len(arr)} events → TTree '{tree_name}'")
    print(f"\n--- Done: created {output_file} ---")

if __name__ == "__main__":
    main()
