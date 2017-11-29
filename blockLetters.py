#!/usr/bin/python3
from neuronBlock import *

inputLetters = 6
learningRate = 0.1
hiddenLayer = len(letters)

firstLayer = NeuronBlock(len(letters) * inputLetters, hiddenLayer)
secondLayer = NeuronBlock(hiddenLayer, len(letters))

with open('moby_dick_short.txt') as f:
  #text = f.read()
  text = "ab" * 300
  trainingPairs = textToTrainingPairs(text, inputLetters) # Each pair is (ngram, next letter), in list form
  for trainingPair in trainingPairs:
    secondLayer.evaluate(firstLayer.evaluate(trainingPair[0]))
    #firstLayer.evaluate(trainingPair[0])

    errors = secondLayer.outputs - trainingPair[1]
    #errors = firstLayer.outputs - trainingPair[1]

    totalError = sum((errors ** 2)) / 2
    print("Total error: " + str(totalError)[:6] + '    ' + '#'*int(totalError*10)) #Print total error, with a bar to represent visually
    print(''.join(listToLetters(secondLayer.outputs)))
    #print(''.join(listToLetters(firstLayer.outputs)))

    print("Backprop layer 2...")
    hiddenDerivs = secondLayer.backprop(errors, learningRate)
    print("Backprop layer 1...")
    firstLayer.backprop(hiddenDerivs, learningRate)
    #firstLayer.backprop(errors, learningRate)

"""
lastN = list("_etymo")
assert(len(lastN) == inputLetters)

for _ in range(100):
  prediction = listToLetters(secondLayer.evaluate(firstLayer.evaluate(ngramToList(lastN))))
  if prediction[0] == " ":
    prediction = prediction[1]
  else:
    prediction = prediction[0]
  print(prediction[0], end="")
  lastN = lastN[1:] + [prediction[0]]
"""
