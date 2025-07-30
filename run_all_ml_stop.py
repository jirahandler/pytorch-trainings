#!/usr/bin/env python3
import subprocess
import os

# --- Main Configuration ---
# Path to the universal skeleton config file
SKELETON_CONFIG_PATH = "trf-stop-ml-config-grouped.txt"

# List of stop analysis points (no scaling or magnification)
STOP_POINTS = [
    "sT_tN1_1000_102_100",
    "sT_tN1_1000_202_200",
    "sT_tN1_1000_3_1",
]

# Fixed signal scale factor (unity)
FIXED_SCALE_FACTOR = 1.0
# Base name of the discriminant ROOT file (no extension)
NTUPLE_FILE = "run3_btag_stopana"
# Suffix used in TTree and branch names
SUFFIX = "_stopana"


def main():
    """
    Generates a TRexFitter config for each stop point without any scale factor adjustments.
    """
    print("--- Generating TRexFitter configs for stop analysis (no scaling) ---")

    # Read skeleton template
    try:
        with open(SKELETON_CONFIG_PATH, "r") as f:
            base_config = f.read()
    except FileNotFoundError:
        print(f"FATAL: Skeleton config not found: {SKELETON_CONFIG_PATH}")
        return

    for point in STOP_POINTS:
        print(f"\n{'='*50}\nPoint: {point}\n{'='*50}")

        # Prepare replacements
        replacements = {
            "JOB_NAME":            point,
            "OUTPUT_DIRECTORY":    f"./{point}_fit",
            "SIGNAL_LABEL":        f"stop {point}",
            "NTUPLE_FILE_NAME":    NTUPLE_FILE,
            "DISCRIMINANT_BRANCH": f"discriminant{SUFFIX}",
            "SIGNAL_NAME":         point,
            "SUFFIX":              SUFFIX,
            "SIGNAL_SCALE_FACTOR": f"{FIXED_SCALE_FACTOR:.8f}",
        }

        # Fill template
        config_text = base_config
        for placeholder, val in replacements.items():
            config_text = config_text.replace(placeholder, val)

        # Write out config file
        cfg_name = f"config_{point}.txt"
        with open(cfg_name, "w") as cfg:
            cfg.write(config_text)
        print(f"Generated config: {cfg_name}")

        # Run TRexFitter
        cmd = ["trex-fitter", "nwdpf", cfg_name]
        print(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"--- Completed: {point} ---")
        except subprocess.CalledProcessError:
            print(f"ERROR: TRexFitter failed for {point}")
            break
        except FileNotFoundError:
            print("ERROR: 'trex-fitter' command not found in PATH.")
            break


if __name__ == "__main__":
    main()
