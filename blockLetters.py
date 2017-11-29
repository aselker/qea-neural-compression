#!/usr/bin/python3
from neuronBlock import *

inputLetters = 8
learningRate = 0.2
hiddenLayer = len(letters)

firstLayer = NeuronBlock(len(letters) * inputLetters, hiddenLayer)
secondLayer = NeuronBlock(hiddenLayer, len(letters))

with open('moby_dick_cleaned.txt') as f:
  text = f.read()[1:3000]
  #text = "abcdefgh" * 300
  trainingPairs = textToTrainingPairs(text, inputLetters) # Each pair is (ngram, next letter), in list form
  for trainingPair in trainingPairs:
    secondLayer.evaluate(firstLayer.evaluate(trainingPair[0]))
    #firstLayer.evaluate(trainingPair[0])

    errors = secondLayer.outputs - trainingPair[1]
    #errors = firstLayer.outputs - trainingPair[1]

    print("Desired letter: " + listToLetters(trainingPair[1])[0])
    print(''.join(listToLetters(secondLayer.outputs)))
    letterPos = listToLetters(secondLayer.outputs).index(listToLetters(trainingPair[1])[0])
    print("Letter position: " + str(letterPos) + (' ' if letterPos>9 else '  ') + '#'*letterPos)
    print("")

    #print("Backprop layer 2...")
    hiddenDerivs = secondLayer.backprop(errors, learningRate)
    #print("Backprop layer 1...")
    firstLayer.backprop(hiddenDerivs, learningRate)
    #firstLayer.backprop(errors, learningRate)

lastN = list("_etymolo")
assert(len(lastN) == inputLetters)

for _ in range(100):
  prediction = listToLetters(secondLayer.evaluate(firstLayer.evaluate(ngramToList(lastN))))
  #if prediction[0] == " ":
  #  prediction = prediction[1]
  #else:
  #  prediction = prediction[0]
  prediction = prediction[0]
  print(prediction[0], end="")
  lastN = lastN[1:] + [prediction[0]]
