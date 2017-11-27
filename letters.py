#!/usr/bin/python3

from neurons import *
"""
inVals = np.array(letterToList('a'))
outDesired = np.array(letterToList('e'))
"""
nIn = []
conn1 = []
nOut = []

for i in range(len(letters)):
  nIn.append(Neuron(False))
  nIn[i].value = 0 #All input weights start at zero
  conn1.append([Conn(nIn[i],1) for _ in range(len(letters))]) #And conn weights start at 1

for i in range(len(letters)):
  nOut.append(Neuron([j[i] for j in conn1]))

def updateAndBackprop(letterIn, letterOut, doPrint = False):

  #First, change the inputs and desired outputs
  inVals = np.array(letterToList(letterIn))
  outDesired = np.array(letterToList(letterOut))

  if doPrint:
    print("Input values:")
    printLetterWeights(inVals, True)
    print("Desired output:")
    printLetterWeights(outDesired, True)

  for i in range(len(letters)):
    nIn[i].value = inVals[i] #Change the input values

  #Update each connection and neuron
  for i in conn1:
    for j in i:
      j.update()

  for i in nOut:
    i.update()

  #Calculate and print errors
  results = [i.value for i in nOut]
  errors = results - outDesired

  if doPrint:
    print(">>>>> Results:")
    printLetterWeights(results)
    print(">>>>> Errors:")
    printLetterWeights(errors)

  totalError = sum((errors ** 2)) / 2
  print("Total error: " + str(totalError)[:6] + '    ' + '#'*int(totalError*10)) #Print total error, with a bar to represent visually

  #And backpropagate!
  for i in range(len(letters)):
    nOut[i].backprop(errors[i], 1) #The error is the derivative!  Quadratics are cool

with open('moby_dick_short.txt') as f:
  prev_char = ' '
  i = 0
  for char in f.read():
    print(' '*(6-len(str(i))) + str(i) + '    ', end='') #Print iteration count, and padding
    updateAndBackprop(prev_char, char, False)
    prev_char = char
    if i == 1000:
      break
    i += 1
updateAndBackprop(' ', ' ', True) #One iteration to print info

