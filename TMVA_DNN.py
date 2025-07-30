#!/usr/bin/env python
import os
import sys
import argparse
import ROOT
from ROOT import *
from array import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.regularizers import l2

def main(arguments):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', default='sT_bC1')
    parser.add_argument('--option', default='5fold')
    parser.add_argument('--hp', default='416_160_416_480_0.0001_100_128')
    parser.add_argument('--acti', default='leakyrelu')
    parser.add_argument('--opt', default='Adam')
    parser.add_argument('--NN', default='1')
    parser.add_argument('--sample', default='all')
    args = parser.parse_args()

    channel = args.channel
    option = args.option
    hp = args.hp
    acti = args.acti
    optim = args.opt
    sam = args.sample
    iNN = args.NN

    indir = "/eos/user/m/minlin/monobc/MLinputs/"
    out_dir = "/eos/user/m/minlin/monobc/MLoutputs/"

    # Parse hyperparameters
    parameters = hp.split("_")
    para_len = len(parameters)

    if para_len == 6:
        layer1, layer2, layer3, learnrate, nepochs, batchsize = parameters
        parameters_str = layer1 + "_" + layer2 + "_" + layer3 + "_" + learnrate.replace(".","") + "_" + nepochs+ "_" +batchsize
    elif para_len == 7:
        layer1, layer2, layer3, layer4, learnrate, nepochs, batchsize = parameters
        parameters_str = layer1 + "_" + layer2 + "_" + layer3 + "_" + layer4 + "_" + learnrate.replace(".","") + "_" + nepochs+ "_" +batchsize

    print(indir, option, acti, optim, sam, iNN, hp)

    # TMVA initialization
    TMVA.Tools.Instance()
    TMVA.PyMethodBase.PyInitialize()

    outputstr = "Run3_DNN_" + channel + "_" + option + "_Keras" + iNN + "_" + parameters_str + "_" + sam
    fout = ROOT.TFile(out_dir + "Train_out/TrainOutput_DNN_" + outputstr + ".root", "RECREATE")

    factory = ROOT.TMVA.Factory("TMVAClassification", fout, "!V:!Silent:Color:!DrawProgressBar:AnalysisType=Classification")
    
    loader = TMVA.DataLoader("mva")

    # Define input variables
    input_vars = ["jet1_pt","jet1_eta","met_pt","met_sig","meff","mjj","dRjj","dEtajj","jet2_pt","R_met_jet","R_met_meff","sum_jet_pt"]
    for var in input_vars:
        loader.AddVariable(var)
    print("Input variables:", len(input_vars), input_vars)

    # Add samples
    if sam == "all":
        samples = ["sT_bC1_1000_102_100", "sT_bC1_1000_202_200", "ttbar", "singletop", "znunu", "diboson", "dijet", "wlnu", "zll"]
    elif sam == "znunu":
        samples = ["sT_bC1_1000_102_100", "sT_bC1_1000_202_200", "znunu"]
    elif sam == "top":
        samples = ["sT_bC1_1000_102_100", "sT_bC1_1000_202_200", "ttbar", "singletop"]
    elif sam == "others":
        samples = ["sT_bC1_1000_102_100", "sT_bC1_1000_202_200", "diboson", "dijet", "wlnu", "zll"]

    for sample in samples:
        print("")
        print("Sample:", sample)
        rfilename = indir+"/"+sample+"_fold"+iNN+".root"
        print("reading", rfilename)
        rfile = TFile(rfilename, "READ")
        if "sT" in sample:
            SigTrainTree = rfile.Get("training")
            SigValTree = rfile.Get("validation")
            for i in range(SigValTree.GetEntries()):
                SigValTree.GetEntry(i)
                tvars = ROOT.std.vector('double')()
                for j in input_vars:
                    tvars.push_back(getattr(SigValTree, j))
                loader.AddSignalTestEvent(tvars, abs(SigValTree.weight))
            for i in range(SigTrainTree.GetEntries()):
                SigTrainTree.GetEntry(i)
                tvars = ROOT.std.vector('double')()
                for j in input_vars:
                    tvars.push_back(getattr(SigTrainTree, j))
                loader.AddSignalTrainingEvent(tvars, abs(SigTrainTree.weight))
        else:
            BKGTrainTree = rfile.Get("training")
            BKGValTree = rfile.Get("validation")
            for i in range(BKGValTree.GetEntries()):
                BKGValTree.GetEntry(i)
                tvars = ROOT.std.vector('double')()
                for j in input_vars:
                    tvars.push_back(getattr(BKGValTree, j))
                loader.AddBackgroundTestEvent(tvars, abs(BKGValTree.weight))
            for i in range(BKGTrainTree.GetEntries()):
                BKGTrainTree.GetEntry(i)
                tvars = ROOT.std.vector('double')()
                for j in input_vars:
                    tvars.push_back(getattr(BKGTrainTree, j))
                loader.AddBackgroundTrainingEvent(tvars, abs(BKGTrainTree.weight))
        rfile.Close()

    loader.PrepareTrainingAndTestTree(TCut(""), "SplitMode=Random:NormMode=EqualNumEvents:!V")
    
    if acti == "relu":
        if para_len == 7:
            model = Sequential([
                layers.Input(shape=(len(input_vars),)),
                layers.Dense(int(layer1), activation='relu', kernel_regularizer=l2()),
                BatchNormalization(),
                layers.Dense(int(layer2), activation='relu', kernel_regularizer=l2()),
                BatchNormalization(),
                layers.Dense(int(layer3), activation='relu', kernel_regularizer=l2()),
                BatchNormalization(),
                layers.Dense(int(layer4), activation='relu', kernel_regularizer=l2()),
                BatchNormalization(),
                layers.Dense(2, activation='softmax', kernel_regularizer=l2())
            ])
        if para_len == 6:
            model = Sequential([
                layers.Input(shape=(len(input_vars),)),
                layers.Dense(int(layer1), activation='relu', kernel_regularizer=l2()),
                BatchNormalization(),
                layers.Dense(int(layer2), activation='relu', kernel_regularizer=l2()),
                BatchNormalization(),
                layers.Dense(int(layer3), activation='relu', kernel_regularizer=l2()),
                BatchNormalization(),
                layers.Dense(2, activation='softmax', kernel_regularizer=l2())
            ])
    elif acti == "leakyrelu":
        if para_len == 7:
            model = Sequential([
                layers.Input(shape=(len(input_vars),)),
                layers.Dense(int(layer1), activation=LeakyReLU(), kernel_regularizer=l2()),
                BatchNormalization(),
                layers.Dense(int(layer2), activation=LeakyReLU(), kernel_regularizer=l2()),
                BatchNormalization(),
                layers.Dense(int(layer3), activation=LeakyReLU(), kernel_regularizer=l2()),
                BatchNormalization(),
                layers.Dense(int(layer4), activation=LeakyReLU(), kernel_regularizer=l2()),
                BatchNormalization(),
                layers.Dense(2, activation='softmax', kernel_regularizer=l2())
            ])
        if para_len == 6:
            model = Sequential([
                layers.Input(shape=(len(input_vars),)),
                layers.Dense(int(layer1), activation=LeakyReLU(), kernel_regularizer=l2()),
                BatchNormalization(),
                layers.Dense(int(layer2), activation=LeakyReLU(), kernel_regularizer=l2()),
                BatchNormalization(),
                layers.Dense(int(layer3), activation=LeakyReLU(), kernel_regularizer=l2()),
                BatchNormalization(),
                layers.Dense(2, activation='softmax', kernel_regularizer=l2())
            ])
    else:
        print("No such activation function!")

    if optim == "SGD":
        opti = tf.keras.optimizers.legacy.SGD(learning_rate=float(learnrate), decay=0.0001)
    elif optim == "Adam":
        opti = tf.keras.optimizers.legacy.Adam(learning_rate=float(learnrate), decay=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=opti, metrics=["accuracy"],weighted_metrics=[])

    model.summary()
    modelOutFile = f"{out_dir}model/model_{outputstr}.h5"
    model.save(modelOutFile)

    options = "!H:!V:FilenameModel=" + modelOutFile + ":NumEpochs=" + nepochs + ":BatchSize=" + batchsize + ":Verbose=2" + ":TriesEarlyStopping=5"
    factory.BookMethod(loader, TMVA.Types.kPyKeras, outputstr, options)

    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()

    fout.Close()
    
    print("finished")

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))