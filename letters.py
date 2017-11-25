from neurons import *

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

