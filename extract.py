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

out = ""
network.states = network.initStates
nextIn = " "

while codings:
  prediction = network.step(letterToList(nextIn))
  tree = makeTree(list(zip(letters, normalize(prediction))))
  (nextIn, codings) = tree.decode(codings)
  out += nextIn

with open(sys.argv[3], 'w') as f:
  f.write(out)
