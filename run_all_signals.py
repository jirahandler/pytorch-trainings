import subprocess
import os

# --- Main Configuration ---

# Path to the universal skeleton config file
SKELETON_CONFIG_PATH = "skeleton-trf-config-ml.txt"
# Tag to append to job/output directory names
ANALYSIS_TAG = "_ctagged"

# Calculation constants
ZNN_TARGET_YIELD = 702063.8
BKG_XSEC_PB = 1063.2
BKG_EFF = 1.0
LQ_MAGNIFICATION = 1000.0
# Using the dictionary for different DM magnifications
DM_MAGNIFICATION = {
    "DM_1p0TeV": 1000.0,
    "DM_1p5TeV": 10000.0,
    "DM_2p5TeV": 100000.0,
}

# Define all signal points with their metadata for the c-tagging analysis
SIGNAL_POINTS = [
    # Leptoquarks
    {"name": "LQ_1p6TeV", "type": "LQ", "mass": "1.6 TeV", "xsec_pb": 0.13,    "n_gen_ntuple": 36504, "survived": 502915, "produced": 600000},
    {"name": "LQ_2TeV",   "type": "LQ", "mass": "2 TeV",   "xsec_pb": 0.05,    "n_gen_ntuple": 35282, "survived": 500561, "produced": 600000},
    {"name": "LQ_2p4TeV", "type": "LQ", "mass": "2.4 TeV", "xsec_pb": 0.03025, "n_gen_ntuple": 34319, "survived": 499548, "produced": 600000},
    # Dark Matter
    {"name": "DM_1p0TeV", "type": "DM", "mass": "1.0 TeV", "xsec_pb": 0.04,    "n_gen_ntuple": 12807, "survived": 100000, "produced": 100000},
    {"name": "DM_1p5TeV", "type": "DM", "mass": "1.5 TeV", "xsec_pb": 0.001615,"n_gen_ntuple": 26885, "survived": 100000, "produced": 100000},
    {"name": "DM_2p5TeV", "type": "DM", "mass": "2.5 TeV", "xsec_pb": 7.831e-6,"n_gen_ntuple": 52144, "survived": 100000, "produced": 100000},
]

def main():
    """
    Loops through all signal points, calculates scale factors, generates a config
    from the skeleton, and runs TrexFitter.
    """
    print(f"--- Starting Standalone TrexFitter Run for All Signal Points ({ANALYSIS_TAG}) ---")

    try:
        with open(SKELETON_CONFIG_PATH, "r") as f:
            base_config = f.read()
    except FileNotFoundError:
        print(f"FATAL: Skeleton config not found at: {SKELETON_CONFIG_PATH}")
        return

    for point in SIGNAL_POINTS:
        point_name_tagged = f"{point['name']}{ANALYSIS_TAG}"
        print(f"\n{'='*50}\nProcessing: {point_name_tagged}\n{'='*50}")

        # --- 1. Scale Factor Calculation ---
        sig_eff = point["survived"] / point["produced"]
        xsec_ratio = point["xsec_pb"] / BKG_XSEC_PB
        eff_ratio = sig_eff / BKG_EFF
        target_signal_yield = ZNN_TARGET_YIELD * xsec_ratio * eff_ratio
        base_sf = target_signal_yield / point["n_gen_ntuple"]

        # --- 2. Determine Analysis-Specific Settings & Placeholders ---
        if point['type'] == 'LQ':
            magnification = LQ_MAGNIFICATION
            suffix = "_lq"
            ntuple_file = f"discriminant_ntuples_lq{ANALYSIS_TAG}"
        else: # DM
            magnification = DM_MAGNIFICATION.get(point['name'], 1.0)
            suffix = "_dm"
            ntuple_file = f"discriminant_ntuples_dm{ANALYSIS_TAG}"

        final_sf = base_sf * magnification
        print(f"Applying magnification: {magnification}x")

        # Define all replacements for the skeleton config
        replacements = {
            "JOB_NAME":             point_name_tagged,
            "OUTPUT_DIRECTORY":     f"./{point_name_tagged}_ML_fit",
            "SIGNAL_LABEL":         f"{point['type']} {point['mass']} (x{int(magnification)})",
            "NTUPLE_FILE_NAME":     ntuple_file,
            "DISCRIMINANT_BRANCH":  f"discriminant{suffix}",
            "SIGNAL_NAME":          point["name"],
            # NOTE: The "SUFFIX" placeholder is no longer used, as the TTree names are simpler now.
            # Make sure your skeleton config has been updated accordingly (e.g., NtupleName: "SIGNAL_NAME_c_tagged")
            "SIGNAL_SCALE_FACTOR":  f"{final_sf:.8f}",
        }

        # --- 3. Generate and Run TrexFitter Config ---
        temp_config = base_config
        for placeholder, value in replacements.items():
            temp_config = temp_config.replace(placeholder, value)

        config_filename = f"config_{point_name_tagged}.txt"
        with open(config_filename, "w") as f:
            f.write(temp_config)
        print(f"Generated config: {config_filename} with FINAL magnified SF = {final_sf:.8f}")

        command = ["trex-fitter", "nwdpf", config_filename]
        print(f"Executing: {' '.join(command)}")

        try:
            subprocess.run(command, check=True, text=True)
            print(f"--- Successfully completed {point_name_tagged} ---")
        except subprocess.CalledProcessError as e:
            print(f"--- ERROR: TRexFitter failed for {point_name_tagged} ---")
            break # Stop if one point fails
        except FileNotFoundError:
            print("\nERROR: 'trex-fitter' command not found. Is it in your PATH?")
            break

if __name__ == "__main__":
    main()
