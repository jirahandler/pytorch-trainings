#!/usr/bin/env python3
"""
Generate and run TRexFitter config for stop analysis with separate signal mass-point ntuples.
This version creates a separate Sample block for each signal file to ensure correct TTree mapping.
"""
import os
import subprocess
import uproot
import numpy as np

# --- Configuration ---
SKELETON             = "trf-stop-ml-config-grouped.txt"
SIGNAL_BASENAMES     = [
    "discriminant_sT_tN1_1000_102_100_stopana",
    "discriminant_sT_tN1_1000_202_200_stopana",
    "discriminant_sT_tN1_1000_3_1_stopana",
]
BACKGROUND_BASENAME  = "discriminant_background_stopana"

BRANCH_NAME          = "discriminant_stopana"
OUTPUT_CONFIG        = "config_stop_combined.txt"
TREXFITTER_COMMAND   = ["trex-fitter", "nwdpf", OUTPUT_CONFIG]
JOB_NAME             = "stop_combined"
OUTPUT_DIRECTORY     = "./stop_combined_fit_balanced_SnB"
SIGNAL_LABEL         = "stop combined"

# --- 1) Count entries and find TTree names ---
sig_count = 0
sig_files_and_trees = []
for base in SIGNAL_BASENAMES:
    filename = base + ".root"
    if not os.path.exists(filename):
        print(f"ERROR: Signal file not found: {filename}")
        exit()
    with uproot.open(filename) as f:
        tree = next(k.split(';')[0] for k in f.keys() if f[k].classname.endswith('TTree'))
        arr = f[tree][BRANCH_NAME].array(library="np")
        sig_count += arr.size
        sig_files_and_trees.append({'basename': base, 'treename': tree})

background_filename = BACKGROUND_BASENAME + ".root"
if not os.path.exists(background_filename):
    print(f"ERROR: Background file not found: {background_filename}")
    exit()
with uproot.open(background_filename) as f:
    bkg_tree = next(k.split(';')[0] for k in f.keys() if f[k].classname.endswith('TTree'))
    bkg_count = f[bkg_tree][BRANCH_NAME].array(library="np").size

if sig_count == 0:
    raise RuntimeError("No signal entries found in discriminant files")
scale_factor = bkg_count / sig_count
print(f"Signal={sig_count}, Background={bkg_count}, Scale={scale_factor:.6f}")

# --- 2) Load skeleton and generate replacement strings ---
with open(SKELETON) as sk:
    cfg = sk.read()

# --- CORRECTED: Generate individual Sample blocks for each signal file with robust formatting ---
sample_blocks = []
signal_sample_names = []
for i, sig in enumerate(sig_files_and_trees):
    sample_name = f"signal_{i}"
    signal_sample_names.append(sample_name)
    # Build the string line-by-line to ensure correct indentation
    block = (
        f'Sample: "{sample_name}"\n'
        f'    Type: SIGNAL\n'
        f'    Title: "{SIGNAL_LABEL} ({i+1})"\n'
        f'    Group: "sig"\n'
        f'    Regions: SR_all\n'
        f'    NtupleFiles: "{sig["basename"]}"\n'
        f'    NtupleNames: "{sig["treename"]}"\n'
        f'    FillColor: 632'
    )
    sample_blocks.append(block)

signal_sample_blocks_str = "\n\n".join(sample_blocks) # Separate blocks with a blank line for readability
signal_sample_names_str = ", ".join(signal_sample_names)

# Prepare background placeholders
background_file   = f'"{BACKGROUND_BASENAME}"'
background_name   = f'"{bkg_tree}"'

# --- 3) Replace all placeholders ---
placeholders = {
    'JOB_NAME':               JOB_NAME,
    'OUTPUT_DIRECTORY':       OUTPUT_DIRECTORY,
    'SIGNAL_LABEL':           SIGNAL_LABEL,
    'SIGNAL_SCALE_FACTOR':    f"{scale_factor:.6f}",
    'SIGNAL_SAMPLE_BLOCKS':   signal_sample_blocks_str,
    'SIGNAL_SAMPLE_NAMES':    signal_sample_names_str,
    'BACKGROUND_NTUPLE_FILE': background_file,
    'BACKGROUND_NTUPLE_NAME': background_name,
    'DISCRIMINANT_BRANCH':    BRANCH_NAME,
}
for k, v in placeholders.items():
    cfg = cfg.replace(k, v)

# --- 4) Write config and run ---
with open(OUTPUT_CONFIG, 'w') as outf:
    outf.write(cfg)
print(f"Wrote {OUTPUT_CONFIG}")

print("Running TRexFitter…")
subprocess.run(TREXFITTER_COMMAND, check=True)
print("Done.")
