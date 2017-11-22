import numpy as np
from math import e
from string import ascii_lowercase

def logistic(x):
  return 1/(1 + e**(-x))

def logDeriv(x):
  return logistic(x) * (1 - logistic(x))

def letterToList(x):
  index = ascii_lowercase.index(x)
  return [0]*index + [1] + [0]*(25-index)

def listToLetter(x):
  return ascii_lowercase[x]

def printLetterWeights(xs, removeZeros = False):
  tuples = [i for i in list(zip(xs,ascii_lowercase)) if (i[0] > 0.01 or not removeZeros)] #Remove the ~zeroes
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


inVals = np.array(letterToList('a'))
outDesired = np.array(letterToList('e'))

nIn = []
conn1 = []
nOut = []

for i in range(26):
  nIn.append(Neuron(False))
  nIn[i].value = inVals[i]
  conn1.append([Conn(nIn[i],1) for _ in range(26)])

for i in range(26):
  nOut.append(Neuron([j[i] for j in conn1]))

print("Input values:")
printLetterWeights(inVals, True)
print("Desired output:")
printLetterWeights(outDesired, True)

for _ in range(200):
  for i in conn1:
    for j in i:
      j.update()

  for i in nOut:
    i.update()

  results = [i.value for i in nOut]
  errors = results - outDesired
  totalError = sum((errors ** 2)) / 2
  print("Total error: " + str(totalError))

  for i in range(26):
    nOut[i].backprop(errors[i], 1) #The error is the derivative!  Quadratics are cool


results = [i.value for i in nOut]
print("Results:")
printLetterWeights(results)
print("Errors:")
printLetterWeights(errors)

