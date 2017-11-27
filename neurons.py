import numpy as np #full of numps
from math import e
from string import ascii_lowercase

letters = ascii_lowercase + " !\"'()*,-./0123456789:;?_" #Something like ASCII order

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


class Conn:
  def __init__(self, source, weight):
    self.source = source
    self.weight = weight
  
  def update(self):
    self.value = self.source.value * self.weight

  def backprop(self, deriv, learnRate):
    weightDeriv = deriv * self.source.value #The derivative of the output of the next neuron, with respect to this conn
    self.weight -= weightDeriv * learnRate #Actually adjust the weight


class Neuron:
  def __init__(self, inputs):
    self.inputs = inputs #'inputs' is False if it has none / this is an input neuron

  def update(self):
    if not self.inputs:
      return
    
    self.total = 0
    for i in self.inputs:
      self.total += i.value

    self.value = logistic(self.total)

  def backprop(self, outDeriv, learnRate):
    if not self.inputs:
      return

    inDeriv = outDeriv * logDeriv(self.total) #Get the derivative of the logistic function, add it to the chain-rule chain
    for i in self.inputs:
      i.backprop(inDeriv, learnRate)

