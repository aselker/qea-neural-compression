#!/usr/bin/python3
from neuronBlock import *

learningRate = 0.01
k1 = 3
k2 = 2

numInputs = len(letters)
layerSpec = [(len(letters),0)]*1

network = RecurrentNet(numInputs, layerSpec)

with open('moby_dick_cleaned.txt') as f:
  #text = f.read()[1:10000]
  text = "abcd" * 100 #Note that if the string length is not coprime with k1, bad things happen
  
  inputs = [letterToList(x) for x in text][:-1]
  targets = [letterToList(x) for x in text][1:]

  network.runAndTrain(inputs, targets, k1, k2, learningRate)
