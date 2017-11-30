import numpy as np #full of numps
from math import e
from random import random

letters = "abcdefghijklmnopqrstuvwxyz !\"'()*,-./0123456789:;?_" #Something like ASCII order

def logistic(x):
  return 1/(1 + e**(-x))

def logDeriv(x):
  return logistic(x) * (1 - logistic(x))

def letterToList(x):
  index = letters.index(x)
  return [0]*index + [1] + [0]*(len(letters)-index-1)

def listToLetters(xs):
  tuples = [ i for i in list(zip(xs,letters)) ]
  tuples = sorted(tuples, key=lambda x: (-x[0],x[1])) #Sort reverse by first value, forward by second
  return [x[1] for x in tuples] #Return just the letters
  

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
    #self.weights = np.array([[1.] * numOutputs] * numInputs)
    self.weights = np.array([[random() for _ in range(numOutputs)] for _ in range(numInputs)])

  def __str__(self):
    return "Weights:\n" + str(self.weights) + "\nLast output:\n" + str(self.outputs)

  def evaluate(self, inputs):
    self.inputs = np.array(inputs) #Does nothing if the input is already an np.array
    self.outputs = logistic(np.dot(self.inputs, self.weights))
    return self.outputs

  def backprop(self, outDerivs, learnRate):
    connDerivs = outDerivs * logDeriv(self.outputs)
    weightDerivs = np.dot(np.transpose([self.inputs]), [connDerivs])
    #print("Weight derivatives: ")
    #print(weightDerivs)
    inputDerivs = np.dot(self.weights, connDerivs)

    self.weights -= weightDerivs * learnRate
    return inputDerivs


# Defines a recurrent neural net.
# numInputs is the number of inputs to the first layer.
# layerSpecs is a list of tuples.  Each tuple is (recurrent in/outputs, connections to the next layer).
# the last item in layerSpecs also defines the output neuron count (its next-layer connections go to the output).
class RecurrentNet:
  def __init__(self, numInputs, layerSpecs, k1, k2, learnRate):

    # Set some constants
    self.k1 = k1
    self.k2 = k2
    self.learnRate = learnRate

    # Create the layers
    self.layers = []
    for i in range(len(layerSpecs)):
      nIn = numInputs if i == 0 else layerSpecs[i-1][0]
      nRec = layerSpecs[i][0]
      nOut = layerSpecs[i][1]
      layers += [ NeuronBlock(nIn + nRec, nOut + nRec) ]

    # Create the transferred-state arrays for recurrence
    self.states = []
    for spec in layerSpecs:
      self.states += [ np.array([random() for _ in range(spec[0])]) ] #Each is initialized with random numbers, to be optimized later

  # "Step" the network, taking input, consuming and re-producing state, and producting output. Does not train.
  def step(inputs):
    # In here, 'inputs' is the non-recurrent inputs to the next layer
    for i in range(len(self.layers)): #Can't use 'for layer,state in layers,states' cause we have to modify states
      # Do the evaluation
      outputs = layers[i].evaluate(inputs + state[i])
      # Split the outputs into inputs for the next layer, and new state
      inputs = outputs[:len(inputs)]
      state[i] = outputs[len(inputs):]
    
    return inputs #The "inputs" for the next layer are actually the outputs of the whole stack.

