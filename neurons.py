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

def printLetterWeights(xs):
  tuples = [i for i in list(zip(xs,ascii_lowercase)) if i[0] > 0.01] #Remove the ~zeroes
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
    weightDeriv = deriv * self.weight #The derivative of the output of the next neuron, with respect to this conn
    self.weight -= weightDeriv * learnRate #Actually adjust the weight

class Neuron:
  def __init__(self, inputs):
    self.inputs = inputs #'inputs' is False if it has none / this is an input neuron

  def update(self):
    if not self.inputs:
      return
    
    total = 0
    for i in self.inputs:
      total += i.value

    self.value = logistic(total)

  def backprop(self, deriv, learnRate):
    if not self.inputs:
      return

    #print("Backpropagating with derivative " + str(deriv))
    inputDeriv = logDeriv(deriv) #Get the derivative of the logistic function, add it to the chain-rule chain
    for i in self.inputs:
      i.backprop(inputDeriv, learnRate)


def main():

  inVals = np.array(letterToList('a'))
  outDesired = np.array(letterToList('e'))
  
  nIn = []
  conn1 = []
  nOut = []

  for i in range(26):
    nIn.append(Neuron(False))
    nIn[i].value = inVals[i]
    conn1.append([Conn(nIn[i],1) for _ in range(26)])

  for _ in range(26):
    nOut.append(Neuron([j[0] for j in conn1]))

  print("Input values:")
  printLetterWeights(inVals)
  print("Desired output:")
  printLetterWeights(outDesired)

  for _ in range(10):
    for i in conn1:
      for j in i:
        j.update()

    for i in nOut:
      i.update()

    results = [i.value for i in nOut]
    errors = ((results - outDesired) ** 2) / 2
    print("Total error: " + str(sum(errors)))
  
    for i in range(26):
      nOut[i].backprop(errors[i], 0.03) #The error is the derivative!  Quadratics are cool


  results = [i.value for i in nOut]
  print("Results:")
  printLetterWeights(results)
  print("Errors:")
  printLetterWeights(errors)


if __name__ == "__main__":
  main()

