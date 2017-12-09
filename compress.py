#!/usr/bin/python3
import sys
import pickle
from neuronBlock import *
from huffman import *

def normalize(xs):
  offset = min(xs)
  xs = [x - offset for x in xs]
  scale = max(xs)
  xs = [x / scale for x in xs]
  return xs

if len(sys.argv) != 4:
  print("Usage: compress.py inputFile networkFilename huffmanFilename")
  sys.exit(2)

learningRate = 0.004
k1 = 1 
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
  with open(sys.argv[2], 'wb') as f2:
    pickle.dump(network, f2)

  # Get the predictions
  predictions = network.run(inputs) #Don't pass it the last letter, because what would the prediction be for?

  # Huffman-code the letters
  huffmanCodes = []
  for (prediction, actual) in zip(predictions, text):
    tree = makeTree(list(zip(letters, normalize(prediction)))) #Letter first, likelihood second
    huffmanCodes += tree.encode(actual)
  
  with open(sys.argv[3], 'w') as f3:
    f3.write(''.join(huffmanCodes))
