#!/usr/bin/python3

import sys
import pickle
from neuronBlock import *
from huffman import *

if len(sys.argv) != 4:
  print("Usage: extract.py networkFile huffmanFile outFile")
  sys.exit(2)

with open(sys.argv[1], 'rb') as f:
  network = pickle.load(f)

with open(sys.argv[2], 'r') as f:
  codings = f.read()

with open("predictions", "rb") as f:
  oldPredictions = pickle.load(f)

out = ""

network.states = network.initStates
nextIn = " "
newPredictions = []

while codings:
  newPrediction = network.step(letterToList(nextIn))
  newPredictions += [newPrediction]
  tree = makeTree(list(zip(letters, normalize(newPrediction))))
  (nextIn, codings) = tree.decode(codings)
  out += nextIn


for prediction in oldPredictions:
  tree = makeTree(list(zip(letters, normalize(prediction))))
  (nextIn, codings) = tree.decode(codings)
  out += nextIn

with open(sys.argv[3], 'w') as f:
  f.write(out)
