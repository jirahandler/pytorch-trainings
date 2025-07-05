#!/usr/bin/env python
import os
import sys
import argparse
import ROOT
from ROOT import *
from array import *

def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default='/eos/user/m/minlin/monobc/MLinputs/')
    parser.add_argument('--channel', default='sT_bC1')
    parser.add_argument('--option', default='5fold')
    parser.add_argument('--BDT', default='1')
    parser.add_argument('--hp', default='300_10_1_0.01')
    parser.add_argument('--sample', default='all')

    args = parser.parse_args()

    indir = args.indir
    channel = args.channel
    option = args.option
    ibdt = args.BDT
    hp = args.hp
    sam = args.sample
    out_dir = "/eos/user/m/minlin/monobc/MLoutputs/"

    parameters = hp.split("_")
    optNTrees, optMaxDepth, optMinNodeSize, optLearnRate = parameters
    parameters_str = optNTrees+"_"+optMaxDepth+"_"+optMinNodeSize.replace(".","")+"_"+optLearnRate.replace(".","")

    print(indir, channel, option, ibdt, hp)

    # TMVA initialization
    TMVA.Tools.Instance()

    outputstr = channel+"_"+option+"_BDT"+ibdt+"_"+parameters_str+"_"+sam
    fout = ROOT.TFile(out_dir + "Train_out/TrainOutput_BDT_" + outputstr + ".root", "RECREATE")

    factory = ROOT.TMVA.Factory("TMVAClassification", fout,":".join(["!V","!Silent","Color","!DrawProgressBar","AnalysisType=Classification"]))

    loader = TMVA.DataLoader("mva")

    # Define input variables
    input_vars = ["jet1_pt","jet1_eta","jet2_pt","jet2_eta","met_pt","met_sig","meff","mjj","pTjj","dRjj","dEtajj","mbb","pTbb","dRbb","dEtabb","dPhibb","R_met_jet","R_met_meff","sum_jet_pt","jet2_gn2pcbt"]

    for var in input_vars:
        loader.AddVariable(var)
    print("Input variables:", len(input_vars), input_vars)

    # Add samples
    if sam == "all":
        samples = ["sT_bC1_1000_102_100", "sT_bC1_1000_202_200", "ttbar", "singletop", "znunu", "diboson", "dijet", "wlnu", "zll"]
    elif sam == "znunu":
        samples = ["sT_bC1_1000_102_100", "sT_bC1_1000_202_200", "znunu"]
    elif sam == "ttbar":
        samples = ["sT_bC1_1000_102_100", "sT_bC1_1000_202_200", "ttbar"]
    elif sam == "others":
        samples = ["sT_bC1_1000_102_100", "sT_bC1_1000_202_200", "singletop", "diboson", "dijet", "wlnu", "zll"]

    s_train_weights = 0.
    b_train_weights = 0.
    s_test_weights = 0.
    b_test_weights = 0.
    for sample in samples:
        print("")
        print("Sample:",sample)
        rfilename = indir+"/"+sample+"_fold"+ibdt+".root"
        rfile = TFile(rfilename,"READ")
        TrainTree = rfile.Get("training")
        trainEntries = TrainTree.GetEntries()
        print("training nentries",trainEntries)
        for i in range(trainEntries):
            TrainTree.GetEntry(i)
            tvars = ROOT.std.vector('double')()
            for j in input_vars:
                tvars.push_back( getattr(TrainTree,j) )
            if "sT" in sample:
                loader.AddSignalTrainingEvent(tvars,abs(TrainTree.weight))
                s_train_weights+=abs(TrainTree.weight)
            else:
                loader.AddBackgroundTrainingEvent(tvars,abs(TrainTree.weight))
                b_train_weights+=abs(TrainTree.weight)
        ValTree = rfile.Get("validation")
        ValNentries = ValTree.GetEntries()
        print("validation nentries",ValNentries)
        for i in range(ValNentries):
            ValTree.GetEntry(i)
            tvars = ROOT.std.vector('double')()
            for j in input_vars:
                tvars.push_back( getattr(ValTree,j) )
            if "sT" in sample:
                loader.AddSignalTestEvent(tvars,abs(ValTree.weight))
                s_test_weights+=abs(ValTree.weight)
            else:
                loader.AddBackgroundTestEvent(tvars,abs(ValTree.weight))
                b_test_weights+=abs(ValTree.weight)
        rfile.Close()
    print("s_test_weights", s_test_weights)
    print("s_train_weights", s_train_weights)
    print("b_test_weights", b_test_weights)
    print("b_train_weights", b_train_weights)


    loader.PrepareTrainingAndTestTree(TCut(""), "SplitMode=Random:NormMode=EqualNumEvents:!V")

    bdtoptions = "!H:!V:NTrees="+optNTrees+":MaxDepth="+optMaxDepth+":MinNodeSize="+optMinNodeSize+"%:Shrinkage="+optLearnRate+":nCuts=20:BoostType=Grad:UseBaggedBoost=true:BaggedSampleFraction=0.5:NegWeightTreatment=Pray"
    factory.BookMethod(loader, TMVA.Types.kBDT, outputstr, bdtoptions )

    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()

    fout.Close()
    
    print("finished")

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))