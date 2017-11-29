#!/usr/bin/python3
from neuronBlock import *

inputLetters = 6
learningRate = 0.1

firstLayer = NeuronBlock(len(letters) * inputLetters, len(letters))

with open('moby_dick_short.txt') as f:
  text = f.read()
  trainingPairs = textToTrainingPairs(text, inputLetters) # Each pair is (ngram, next letter), in list form
  for trainingPair in trainingPairs:
    firstLayer.evaluate(trainingPair[0])
    #errors = firstLayer.outputs - trainingPair[1]
    errorsDeriv = firstLayer.outputs #Most of the error derivatives are just the letter scores, cause most should be zero
    errorsDeriv[letters.index(trainingPair[1])] = -listToLetters(firstLayer.outputs).index(trainingPair[1]) #The one that should be ranked first should have a negative derivative

    #totalError = sum((errorsDeriv ** 2)) / 2
    #print("Total error: " + str(totalError)[:6] + '    ' + '#'*int(totalError*10)) #Print total error, with a bar to represent visually
    print("Total error: " + str(sum(errorsDeriv))[:6] + '    ' + '#'*int(sum(errorsDeriv)*10)) #Print total error, with a bar to represent visually

    
    firstLayer.backprop(errorsDeriv, learningRate)
  
