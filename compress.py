#!/usr/bin/python3
import sys
import pickle
from neuronBlock import *
from huffman import *

if len(sys.argv) != 4:
  print("Usage: compress.py inputFile networkFilename huffmanFilename")
  sys.exit(2)

learningRate = 0.004
k1 = 4 
k2 = 32

numInputs = len(letters)
layerSpec = [(len(letters),24)]*6

network = RecurrentNet(numInputs, layerSpec)

with open(sys.argv[1]) as f:
  text = " " + f.read() #The space at the beginning lets us predict and Huffman-code the first letter
  
  inputs = [letterToList(x) for x in text][:-1]
  targets = [letterToList(x) for x in text][1:]
  
  # Train the network
  network.runAndTrain(inputs, targets, k1, k2, learningRate)

  # Serialize and save the trained network
  with open(sys.argv[2]) as f2:
    pickle.dump(network, f2)

  # Get the predictions
  predictions = network.run(inputs) #Don't pass it the last letter, because what would the prediction be for?

  # Huffman-code the letters
  huffmanCodes = []
  for (prediction, target) in zip(predictions, targets):
    tree = makeTree(list(zip(letters, prediction))) #Letter first, likelihood second
    huffmanCodes += tree.encode(target)
  
