#!/usr/bin/python3

import sys
import pickle
from neuronBlock import *
from huffman import *

if len(sys.argv) != 4:
  print("Usage: compress.py inputFile networkFile huffmanFile")
  sys.exit(2)

learningRate = 0.002
k1 = 1 
k2 = 32
learnIterations = 10

numInputs = len(letters)
layerSpec = [(len(letters),24)]*6

network = RecurrentNet(numInputs, layerSpec)

with open(sys.argv[1]) as f:
  text = " " + f.read() #The space at the beginning lets us predict and Huffman-code the first letter

inputs = [letterToList(x) for x in text][:-1]
targets = [letterToList(x) for x in text][1:]

# Train the network
for _ in range(learnIterations):
  network.runAndTrain(inputs, targets, k1, k2, learningRate)

# Serialize and save the trained network
with open(sys.argv[2], 'wb') as f2:
  pickle.dump(network, f2)
with open(sys.argv[2], 'rb') as f2:
  otherNet = pickle.load(f2)

# Get the predictions
predictions = network.run(inputs) #Don't pass it the last letter, because what would the prediction be for?
otherNet.states = otherNet.initStates
otherPredictions = [otherNet.step(x) for x in inputs]

for (first, second) in zip(predictions, otherPredictions):
  print(np.linalg.norm(np.array(first) - np.array(second)))

with open("predictions", 'wb') as f3:
  pickle.dump(predictions, f3)

# Huffman-code the letters
huffmanCodes = []
for (prediction, actual) in zip(predictions, text):
  tree = makeTree(list(zip(letters, normalize(prediction)))) #Letter first, likelihood second
  huffmanCodes += tree.encode(actual)

with open(sys.argv[3], 'w') as f3:
  f3.write(''.join(huffmanCodes))
