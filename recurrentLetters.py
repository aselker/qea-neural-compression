#!/usr/bin/python3
from neuronBlock import *

learningRate = 0.01
k1 = 2 
k2 = 9

numInputs = len(letters)
layerSpec = [(len(letters),16)]*6

network = RecurrentNet(numInputs, layerSpec)

with open('moby_dick_cleaned.txt') as f:
  text = f.read()[1:500000]
  #text = "aaaab" * 1000 #Note that if the string length is not coprime with k1, bad things happen
  
  inputs = [letterToList(x) for x in text][:-1]
  targets = [letterToList(x) for x in text][1:]

  network.runAndTrain(inputs, targets, k1, k2, learningRate)
