import ROOT
import os

def split_ttree(input_file, tree_name, output_dir, region, sample_name, n_splits=5):
    """Splits a TTree into n_splits parts for cross-validation and writes test and training TTrees to separate files."""
    file_in = ROOT.TFile.Open(input_file, "READ")
    if not file_in or file_in.IsZombie():
        print(f"Error: Could not open input file {input_file}")
        return

    tree = file_in.Get(tree_name)
    if not tree:
        print(f"Error: Could not find TTree named {tree_name} in file {input_file}")
        return

    n_entries = tree.GetEntries()
    entries_per_split = n_entries // n_splits
    remainder = n_entries % n_splits

    # Calculate the index ranges for the splits
    indices = []
    start = 0
    for i in range(n_splits):
        end = start + entries_per_split + (1 if i < remainder else 0)
        indices.append(range(start, end))
        start = end

    for fold in range(n_splits):
        test_indices = set(indices[fold])  # Current fold's test set indices
        remove_indices = set(indices[(fold + 1) % n_splits])  # Remove the next fold's indices

        # Define the output file for this fold
        output_file = os.path.join(output_dir, f"{sample_name}_fold{fold + 1}.root")
        file_out = ROOT.TFile(output_file, "RECREATE")

        # Create TTrees for test and training
        test_tree = tree.CloneTree(0)
        test_tree.SetName("test")
        training_tree = tree.CloneTree(0)
        training_tree.SetName("training")
        val_tree = tree.CloneTree(0)
        val_tree.SetName("validation")

        # Fill the test and training TTrees
        for i in range(n_entries):
            tree.GetEntry(i)
            if i in remove_indices:
                val_tree.Fill()
            if i in test_indices:
                test_tree.Fill()
            else:
                training_tree.Fill()

        # Write TTrees to the output file
        file_out.cd()
        test_tree.Write()
        training_tree.Write()
        val_tree.Write()
        file_out.Close()

        print(f"Fold {fold + 1}: Saved test and training TTrees for sample {sample_name} to {output_file}")

    file_in.Close()

# Configuration
samples = ["dijet", "singletop", "diboson", "ttbar", "znunu", "wlnu", "zll", "sT_bC1_1000_202_200", "sT_bC1_1000_102_100"]
region = "SR"
tree_name = "sel_tree"
output_dir = f"/eos/user/m/minlin/monobc/MLinputs/"

# Loop over samples
for sample in samples:
    input_file = f"/afs/cern.ch/work/m/minlin/private/bcoffea_run3/tbc1_untag/all/{sample}/basicSel_{sample}.root"
    print(f"Processing sample {sample}...")
    split_ttree(input_file, tree_name, output_dir, region, sample)
