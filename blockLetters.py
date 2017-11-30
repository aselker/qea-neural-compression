#!/usr/bin/python3
from neuronBlock import *

inputLetters = 16
learningRate = 0.8
hiddenLayer = len(letters)

firstLayer = NeuronBlock(len(letters) * inputLetters, hiddenLayer)
secondLayer = NeuronBlock(hiddenLayer, len(letters))

with open('moby_dick_cleaned.txt') as f:
  text = f.read()[1:8000]
  #text = "abcdefgh" * 300
  trainingPairs = textToTrainingPairs(text, inputLetters) # Each pair is (ngram, next letter), in list form

  #positions = [0] * 10 #The average error over a few iterations

  for trainingPair in trainingPairs:
    secondLayer.evaluate(firstLayer.evaluate(trainingPair[0]))

    errors = secondLayer.outputs - trainingPair[1]

    print("Desired letter: " + listToLetters(trainingPair[1])[0])
    print(''.join(listToLetters(secondLayer.outputs)))
    letterPos = listToLetters(secondLayer.outputs).index(listToLetters(trainingPair[1])[0])
    print("Letter position: " + str(letterPos) + (' ' if letterPos>9 else '  ') + '#'*letterPos)
    print("")
    #positions = positions[1:] + [letterPos]
    #runningAvg = int(sum(positions) / len(positions))
    #print("Avg. position: " + str(runningAvg) + (' ' if runningAvg>9 else '  ') + '#'*runningAvg)


    hiddenDerivs = secondLayer.backprop(errors, learningRate)
    firstLayer.backprop(hiddenDerivs, learningRate)

  lastN = list(text[0:inputLetters])
  for _ in range(100):
    prediction = listToLetters(secondLayer.evaluate(firstLayer.evaluate(ngramToList(lastN))))
    #if prediction[0] == " ":
    #  prediction = prediction[1]
    #else:
    #  prediction = prediction[0]
    prediction = prediction[0]
    print(prediction[0], end="")
    lastN = lastN[1:] + [prediction[0]]
