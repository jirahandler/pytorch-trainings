#!/usr/bin/env python

import os
import sys
import argparse
import math
from array import *
import numpy as np
import ROOT
from ROOT import TMVA, TFile, TTree, TCut, TString, TH1F

ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2(True)

#-------------------------------------------------
def main(arguments):

    # At the moment only working for 5fold

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default='/eos/user/m/minlin/monobc/MLinputs/')
    parser.add_argument('--channel', default='sT_bC1')
    parser.add_argument('--option', default='5fold')
    parser.add_argument('--hp', default='300_10_1_0.01')
    parser.add_argument('--set', default='Validation')
    parser.add_argument('--sample', default='')
    args = parser.parse_args()

    indir = args.indir
    channel = args.channel
    option = args.option
    iset = args.set
    hp = args.hp
    sample = args.sample # "Signal", "Fakes", "Fakes_MC", "Ztt", "Others"

    # Parse hyperparameters
    parameters = hp.split("_")
    optNTrees, optMaxDepth, optMinNodeSize, optLearnRate = parameters
    parameters_str = optNTrees+"_"+optMaxDepth+"_"+optMinNodeSize.replace(".","")+"_"+optLearnRate.replace(".","")

    if iset=="Validation":
        treename = "validation"
    else:
        treename = "test"

    # TMVA and reader
    TMVA.Tools.Instance()
    TMVA.PyMethodBase.PyInitialize()

    input_vars = ["jet1_pt","jet1_eta","jet2_pt","jet2_eta","met_pt","met_sig","meff","mjj","pTjj","dRjj","dEtajj","mbb","pTbb","dRbb","dEtabb","dPhibb","R_met_jet","R_met_meff","sum_jet_pt","jet2_gn2pcbt"]

    input_vars_arrays = {i: array('f', [0]) for i in input_vars}

    hists = TH1F(f"h_score_{sample}", "", 20, -1.0, 1.0)
    hists.SetDirectory(0)

    for n in range(1, 6):
        rfilename = f"{indir}/{sample}_fold{n}.root"
        print(f"rfile: {rfilename}")
        rfile = TFile(rfilename, "READ")
        ttree = rfile.Get(treename)
        nentries = ttree.GetEntries()
        print(f"nentries: {nentries}")

        reader = TMVA.Reader()
        for i in input_vars:
            reader.AddVariable(i, input_vars_arrays[i])

        reader.BookMVA(f"BDT{n}", f"mva/weights/TMVAClassification_{channel}_5fold_BDT{n}_{parameters_str}_all.weights.xml")

        for k in range(nentries):
            ttree.GetEntry(k)

            for j in input_vars:
                input_vars_arrays[j][0] = getattr(ttree, j)

            scores = reader.EvaluateMVA(f"BDT{n}")
            weight = ttree.weight

            hists.Fill(scores, weight)

        del reader
        rfile.Close()

    outfile_name = f"/eos/user/m/minlin/monobc/MLoutputs/reader_output/hist_Run3_BDT_{sample}_{channel}_{option}_Val_{parameters_str}_all.root"
    print(f"rfile_out: {outfile_name}")
    rfile_out = TFile(outfile_name, "RECREATE")
    hists.Write()
    rfile_out.Close()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
