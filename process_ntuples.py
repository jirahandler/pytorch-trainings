import uproot
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import argparse
from sklearn.preprocessing import StandardScaler

def main(analysis_type):
    """
    Processes source ntuples to create discriminant ntuples with correctly scaled inputs.
    """
    print(f"--- Starting NTuple processing for {analysis_type} with input scaling ---")

    # --- Configuration ---
    NTUPLE_BASE_PATH = "/home/sgoswami/monobcntuples/local-samples/trf-workdir/SR/flattenedNTuples"
    FEATURES = ["jet1_pt", "jet1met_dphi", "met_sig", "met_pt"]
    CATEGORIES = ["b_tagged", "untagged"]

    # --- Analysis-Specific Settings ---
    if analysis_type == 'LQ':
        MODEL_PATH = "/home/sgoswami/monobcntuples/ML/best_model_lq.keras"
        output_file = "discriminant_ntuples_lq.root"
        suffix = "_lq"
        signal_files = {
            "LQ_1p6TeV": f"{NTUPLE_BASE_PATH}/sig/lq/flat_tuple_lq_1p6TeV_merged_600K.root",
            "LQ_2TeV":   f"{NTUPLE_BASE_PATH}/sig/lq/flat_tuple_lq_2TeV_merged_600K.root",
            "LQ_2p4TeV": f"{NTUPLE_BASE_PATH}/sig/lq/flat_tuple_lq_2p4TeV_merged_600K.root",
        }
    elif analysis_type == 'DM':
        MODEL_PATH = "/home/sgoswami/monobcntuples/ML/best_model_dm.keras"
        output_file = "discriminant_ntuples_dm.root"
        suffix = "_dm"
        signal_files = {
            "DM_1p0": f"{NTUPLE_BASE_PATH}/sig/dm/flat_tuple_yy_1p0_qcd.root",
            "DM_1p5": f"{NTUPLE_BASE_PATH}/sig/dm/flat_tuple_yy_1p5_qcd.root",
            "DM_2p5": f"{NTUPLE_BASE_PATH}/sig/dm/flat_tuple_yy_2p5_qcd.root",
        }
    else:
        print(f"FATAL: Unknown analysis type '{analysis_type}'. Use 'LQ' or 'DM'.")
        return

    background_files = {
        "znunu": f"{NTUPLE_BASE_PATH}/bkg/flat_tuple_znunu_600K.root",
        "ttbar": f"{NTUPLE_BASE_PATH}/bkg/flat_tuple_ttbar.root",
        "wjets": [f"{NTUPLE_BASE_PATH}/bkg/flat_tuple_wlnu.root"],
    }

    all_samples = {**signal_files, **background_files}

    # --- Step 1: Load all data from all files into a single DataFrame ---
    print("\nLoading all data to determine scaling parameters...")
    all_data_dfs = []
    data_indices = {}
    current_pos = 0

    for sample_name, path in all_samples.items():
        for category in CATEGORIES:
            df = load_features_from_files(path, category, FEATURES)
            if df is not None and not df.empty:
                all_data_dfs.append(df)
                # Store the start and end position of this chunk of data
                data_indices[(sample_name, category)] = (current_pos, current_pos + len(df))
                current_pos += len(df)

    if not all_data_dfs:
        print("FATAL: No data could be loaded. Exiting.")
        return

    combined_df = pd.concat(all_data_dfs, ignore_index=True)
    print(f"Loaded a total of {len(combined_df)} events.")

    # --- Step 2: Scale the features ---
    print("\nScaling input features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combined_df[FEATURES])

    # --- Step 3: Load model and get predictions ---
    print(f"Loading Keras model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"FATAL: Could not load Keras model. Error: {e}")
        return

    print("Running model predictions on scaled data...")
    all_predictions = model.predict(scaled_features, batch_size=4096)
    all_discriminants = all_predictions.flatten()

    # --- Step 4: Write the results to the output ROOT file ---
    print(f"\nWriting scores to output file: {output_file}")
    with uproot.recreate(output_file) as f:
        for (sample_name, category), (start, end) in data_indices.items():
            # Slice the predictions array to get the scores for this specific sample/category
            discriminant_slice = all_discriminants[start:end]

            tree_name = f"{sample_name}_{category}{suffix}"
            f[tree_name] = {f"discriminant{suffix}": discriminant_slice}
            print(f"  -> Wrote {len(discriminant_slice)} events to TTree '{tree_name}'")

    print(f"\n--- Successfully created {output_file} with correct score distributions ---")

def load_features_from_files(file_paths, category_name, features_list):
    """Helper function to load a DataFrame for a given sample/category."""
    if not isinstance(file_paths, list): file_paths = [file_paths]
    dfs = []
    for path in file_paths:
        if not os.path.exists(path): continue
        try:
            with uproot.open(path) as root_file:
                if category_name in root_file:
                    tree = root_file[category_name]
                    if all(b in tree for b in features_list):
                        dfs.append(tree.arrays(features_list, library="pd"))
        except Exception as e:
            print(f"    ERROR processing {path}: {e}")
            return None
    return pd.concat(dfs, ignore_index=True) if dfs else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ntuples with input scaling for LQ or DM analysis.")
    parser.add_argument("--type", type=str, required=True, choices=['LQ', 'DM'], help="Type of analysis to run: 'LQ' or 'DM'")
    args = parser.parse_args()
    main(args.type)
