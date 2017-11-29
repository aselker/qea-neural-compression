import numpy as np #full of numps
from math import e

#letters = "abcdefghijklmnopqrstuvwxyz !\"'()*,-./0123456789:;?_" #Something like ASCII order
letters = "abcde" #For testing

def logistic(x):
  return 1/(1 + e**(-x))

def logDeriv(x):
  return logistic(x) * (1 - logistic(x))

def letterToList(x):
  index = letters.index(x)
  return [0]*index + [1] + [0]*(len(letters)-index-1)

def listToLetter(x):
  return letters[x]

def printLetterWeights(xs, removeZeros = False):
  tuples = [i for i in list(zip(xs,letters)) if (i[0] > 0.01 or not removeZeros)] #Remove the ~zeroes
  tuples = sorted(tuples, key=lambda x: (-x[0],x[1])) #Sort reverse by first value, forward by second
  for i in tuples:
    if len(str(i[0])) > 5:
      print(str(i[0])[0:6] + ":\t" + str(i[1])) #Limit digits
    else:
      print(str(i[0]) + ":\t" + str(i[1])) #All the digits

def ngramToList(ngram):
  out = []
  for i in ngram:
    out.append(letterToList(i))
  return np.array(out).flatten()

def textToTrainingPairs(text, n):
  return [ (ngramToList(text[i:i+n]), letterToList(text[i+n])) for i in range(len(text) - n) ]

class NeuronBlock:
  def __init__(self, numInputs, numOutputs):
    self.inputs = np.array([0.] * numInputs)
    self.outputs = np.array([0.] * numOutputs)
    self.weights = np.array([[1.] * numOutputs] * numInputs)

  def __str__(self):
    return "Weights:\n" + str(self.weights) + "\nLast output:\n" + str(self.outputs)

  def evaluate(self, inputs):
    self.inputs = np.array(inputs) #Does nothing if the input is already an array
    self.outputs = np.array(list(map(logistic, np.dot(self.inputs, self.weights)))) #Multiply by weights, run logistic function
    return self.outputs

  def backprop(self, outDerivs, learnRate):
    print("Out derivatives:  " + str(outDerivs[0:4]))
    connDerivs = np.array(list(map(logDeriv, outDerivs)))
    print("Conn derivatives: " + str(connDerivs[0:4]))
    weightDerivs = np.dot(np.transpose([self.inputs]), [connDerivs])
    print("Weight derivatives: " + str(weightDerivs))
    self.weights -= weightDerivs * learnRate

