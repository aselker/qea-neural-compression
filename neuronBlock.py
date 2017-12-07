import numpy as np #full of numps
from random import random

#letters = "abcdefghijklmnopqrstuvwxyz !\"'()*,-./0123456789:;?_" #Something like ASCII order
letters = "abcdefghijklmnopqrstuvwxyz _"
#letters = "ab"
lowState = -0.5
highState = 0.5

verboseEval = 0
verboseTrain = 0
printOutputs = 1
quietOutputs = 1
writeInitState = 0

printAvgTime = 256

"""
def activFunc(x):
  return 1/(1 + np.exp(-x))

def activDeriv(x):
  return activFunc(x) * (1 - activFunc(x))
"""
"""
def activFunc(x): 
  return np.log(1 + np.exp(x)) #Softplus / smooth ReLU

def activDeriv(x):
  return 1/(1 + np.exp(-x))
""" 
activFunc = np.tanh #tanh activation function

activDeriv = lambda x: 1 - np.power(np.tanh(x),2)
"""
activFunc = lambda xs: np.array([(0 if x<0 else x) for x in xs])
activDeriv = lambda xs: np.array([(0 if x<0 else 1) for x in xs])
"""
  

def letterToList(x):
  index = letters.index(x)
  return np.array([lowState]*index + [highState] + [lowState]*(len(letters)-index-1))

def listToLetters(xs):
  tuples = [ i for i in list(zip(xs,letters)) ]
  tuples = sorted(tuples, key=lambda x: (-x[0],x[1])) #Sort reverse by first value, forward by second
  return [x[1] for x in tuples] #Return just the letters

def listToFirstLetter(xs):
  tuples = [ i for i in list(zip(xs,letters)) ]
  return max(tuples)[1]
  

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
    self.weights = np.array([[random() - 0.5 for _ in range(numOutputs)] for _ in range(numInputs)]) #Initialize with random weights

  def __str__(self):
    return "Weights:\n" + str(self.weights) + "\nLast output:\n" + str(self.outputs)

  def evaluate(self, inputs):
    self.inputs = np.array(inputs) #Does nothing if the input is already an np.array
    self.outputs = activFunc(np.dot(self.inputs, self.weights))

    if verboseEval:
      print("Layer inputs: ")
      print(self.inputs)
      print("Layer weights: ")
      print(self.weights)
      print("Layer outputs: ")
      print(self.outputs)

    return self.outputs

  # Backpropagate, find the derivatives for internal weights and for inputs. Don't actually change anything.
  def backprop(self, outDerivs):
    outputsBeforeActiv = np.dot(self.inputs, self.weights)

    if verboseTrain:
      print("Output+recurrent derivatives:")
      print(outDerivs)
      print("Outputs before activation function:")
      print(outputsBeforeActiv)
    
    connDerivs = outDerivs * activDeriv(outputsBeforeActiv) 
    weightDerivs = np.dot(np.transpose([self.inputs]), [connDerivs])
    inputDerivs = np.dot(self.weights, connDerivs)

    return (weightDerivs, inputDerivs)


# Defines a multi-layered recurrent neural net.
# numInputs is the number of inputs to the first layer.
# layerSpecs is a list of tuples.  Each tuple is (connections to the next layer, recurrent in/outputs).
# the last item in layerSpecs also defines the output neuron count (its next-layer connections go to the output).
class RecurrentNet:
  def __init__(self, numInputs, layerSpecs):

    # Create the layers
    self.layers = []
    for i in range(len(layerSpecs)):
      nIn = numInputs if i == 0 else layerSpecs[i-1][0]
      nRec = layerSpecs[i][1]
      nOut = layerSpecs[i][0]
      self.layers += [ NeuronBlock(nIn + nRec, nOut + nRec) ]

    # Create starting points for the transferred-state arrays for recurrence
    self.initStates = []
    for spec in layerSpecs:
      self.initStates += [ np.array([random() - 0.5 for _ in range(spec[1])]) ] #Each is initialized with random numbers. Later this might get trained.
    
    if writeInitState:
      with open("initStates.txt", 'w') as f:
        f.write(str(self.initStates))

  # "Step" the network, taking input, consuming and re-producing state, and producting output. Does not train.
  # This changes two pieces of state internal to the object: 'states' is the recurrent thing, and 'midputs' is a record of the outputs of each layer which
  #   are fed into the next layer. Both should be saved if using this to backpropagate. If just using this to evaluate the network, you can ignore both, 
  #   except for initializing 'states' and not modifying it between calls to 'step'.
  def step(self, inputs):
    self.midputs = [inputs]
    # In here, 'midputs' is the non-recurrent inputs to the next layer
    for i in range(len(self.layers)): #Can't use 'for layer,state in layers,states' cause we have to modify states
      # Do the evaluation
      outputs = self.layers[i].evaluate(np.concatenate([self.midputs[i], self.states[i]]))
      # Split the outputs into inputs for the next layer, and new state
      self.midputs += [outputs[:len(inputs)]]
      self.states[i] = outputs[len(inputs):]
    
    return self.midputs[-1] #The last 'midputs' is the output

  # Get the derivatives for training one generation of the network. For use with a more complex, recurrence-conscious training algorithm.
  def getDerivs(self, midputs, statesIn, statesOut, stateDerivsOut, outputDeriv):

    weightDerivs = [] #List of 2d arrays of weight derivatives. Each adjusts one layer.
    stateDerivsIn = [] #List of 1d arrays, representing the state derivatives for the *previous* generation (thus "in")
    # outputDeriv is initialized by the argument, but changes every iteration. 


    for i in reversed(range(len(self.layers))):
      self.layers[i].inputs = np.concatenate([midputs[i], statesIn[i]])
      self.layers[i].outputs = np.concatenate([midputs[i+1], statesOut[i]])
      (weightDeriv, inputDeriv) = self.layers[i].backprop( np.concatenate([outputDeriv, stateDerivsOut[i]]) )

      weightDerivs += [weightDeriv]
      if len(stateDerivsOut[i]) > 0:
        outputDeriv = inputDeriv[:-len(stateDerivsOut[i])]
        stateDerivsIn += [inputDeriv[-len(stateDerivsOut[i]):]]
      else:
        outputDeriv = inputDeriv
        stateDerivsIn += [[]]

    weightDerivs.reverse()
    stateDerivsIn.reverse()

    return(stateDerivsIn, weightDerivs)
  

  def runAndTrain(self, inputs, targets, k1, k2, learnRate):
    
    # First, initialize the state. Might train this at some later date?
    self.states = self.initStates
    
    # Next, set up some variables for training...
    record = [] #Tuples of the form (states, midputs), from a particular generation
    trainTimer = 0 #Iterations since last training; when this hits k1 we train

    if printOutputs:
      printTimer = 0
      letterPosAvg = 0
    # Now loop through the inputs, evaluating, saving states, and sometimes training.
    for (ipt,target) in zip(inputs, targets):

      output = self.step(ipt) #Do the computation

      if printOutputs:
        letterPos = listToLetters(output).index(listToFirstLetter(target))

        if not quietOutputs:
          if verboseEval or verboseTrain:
            print("\nStarting new iteration.")
          print("Desired letter: " + listToFirstLetter(target))
          print("Guesses:           " + ''.join(listToLetters(output)))
          print("Letter position: " + str(letterPos) + (' ' if letterPos>9 else '  ') + '#'*letterPos)
        else:
          letterPosAvg += letterPos
          if printTimer > 256:
            letterPosAvg = letterPosAvg/256
            print("Letter position avg: " + str(int(letterPosAvg)) + (' ' if int(letterPosAvg) > 9 else '  ') + '#'*int(letterPosAvg*5))
            letterPosAvg = 0
            printTimer = 0
          printTimer += 1


      errorDeriv = output - target #Quadratic error function -> linear derivative, like before. This is only used if this happens to be a training iteration.

      record.insert(0, (self.midputs, self.states)) #Add to the records
      if len(record) > k2+1: #If we have enough records, start throwing out the old ones
        record = record[:k2+1] #We keep k2+1 records because we need in and out for k2 iterations.

      trainTimer += 1
      if trainTimer >= k1:
        trainTimer = 0

        if verboseTrain:
          print("\nNew training iteration.")

        stateDerivs = [ np.array([0.] * len(x)) for x in self.states ] #The derivatives of the state outputs are zero to start, because they go off into the future
        weightDerivss = [] #Double plural ftw
        
        #Backpropagate backward through time.
        for i in range(min(k2, len(record)-1)):
          midputs = record[i][0]
          statesIn = record[i+1][1]
          statesOut = record[i][1]
          if i == 0:
            outputDeriv = errorDeriv 
          else:
            outputDeriv = np.array([0] * len(midputs[-1])) #If not the last, don't factor in outputs.
                                                           #  This means that this entire training routine trains only for the most recent output(!)
          (stateDerivs, weightDerivs) = self.getDerivs(midputs, statesIn, statesOut, stateDerivs, outputDeriv)
          weightDerivss += [weightDerivs]
          
        # Next, we average weightDerivss and subtract them to the weights.
        avgWeightDerivs = np.average(weightDerivss,0)
        for (layer, deriv) in zip(self.layers, avgWeightDerivs):
          layer.weights -= deriv * learnRate
