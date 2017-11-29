#!/usr/bin/python3
from neuronBlock import *

inputLetters = 3
learningRate = .1

firstLayer = NeuronBlock(len(letters) * inputLetters, len(letters))

with open('moby_dick_short.txt') as f:
  #text = f.read()
  text = "a" * 100
  trainingPairs = textToTrainingPairs(text, inputLetters) # Each pair is (ngram, next letter), in list form
  for trainingPair in trainingPairs:
    firstLayer.evaluate(trainingPair[0])
    errors = firstLayer.outputs - trainingPair[1]

    totalError = sum((errors ** 2)) / 2
    print("Total error: " + str(totalError)[:6] + '    ' + '#'*int(totalError*10)) #Print total error, with a bar to represent visually
    
    firstLayer.backprop(errors, learningRate)
