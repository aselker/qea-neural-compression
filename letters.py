#!/usr/bin/python3

from neurons import *
"""
inVals = np.array(letterToList('a'))
outDesired = np.array(letterToList('e'))
"""

inputLetters = 5

nIn = []
conn1 = []
nOut = []

for i in range(len(letters) * inputLetters):
  nIn.append(Neuron(False))
  nIn[i].value = 0 #All input weights start at zero
  conn1.append([Conn(nIn[i],1) for _ in range(len(letters))]) #And conn weights start at 1

for i in range(len(letters)):
  nOut.append(Neuron([j[i] for j in conn1]))

def updateAndBackprop(ngram, doPrint = False):
  letterOut = ngram[len(ngram)-1] # Last character
  lettersIn = ngram[0:len(ngram)-1] # All but the last

  #First, change the inputs and desired outputs
  #inVals = np.array(letterToList(letterIn))
  outDesired = np.array(letterToList(letterOut))

  if doPrint:
    print("Input values:" + " ".join(lettersIn))
    print("Desired output:")
    printLetterWeights(outDesired, True)

  for i in range(inputLetters): #Set the new input values
    inVals = np.array(letterToList(lettersIn[i]))
    for j in range(len(letters)):
      nIn[i*len(letters) + j].value = inVals[j]

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
  """
  prev_char = ' '
  i = 0
  for char in f.read():
    print(' '*(6-len(str(i))) + str(i) + '    ', end='') #Print iteration count, and padding
    updateAndBackprop([prev_char] * 3, char, False)
    prev_char = char
    if i == 1000:
      break
    i += 1
  """
  text = f.read()[0:999]
  ngrams = [ text[i:i+inputLetters+1] for i in range(len(text) - inputLetters - 1) ] # Each should be inputLetters+1 chars long
  for ngram in ngrams:
    updateAndBackprop(ngram) 

updateAndBackprop('the wh', True) #One iteration to print info

