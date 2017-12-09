from queue import PriorityQueue

# A utility function that indents a multiline string by one space
def indent(x, n=1): 
  return (" "*n) + x.replace("\n", "\n" + " "*n)

# First, we define a binary tree class which will become our coding. Each node, leaf or not, also has an attached probability.
class Node:
  def __init__(self, prob=0, value=None, left=None, right=None):
    assert (value == None and prob == 0) or (left == None and right == None) #Leaf or not a leaf, pls
    self.value = value #value is None if this is not a leaf node
    self.left = left
    self.right = right
    if (left and right):
      self.prob = left.prob + right.prob
    else:
      self.prob = prob

  # To help debug and show stuff, we want to be able to visualize trees.  This should do that recursively.
  def __str__(self):
    if self.value:
      return "(" + str(self.prob)[:3] + ") " + str(self.value) 
    else:
      line1 = indent(str(self.left), 4)
      line2 = "(" + str(self.prob)[:3] + ")"
      line3 = indent(str(self.right), 4)
      return line1 + "\n" + line2 + "\n" + line3

  # In order to be useful in the priority queue, nodes have to be orderable.
  def __lt__(self, other):
    return self.prob < other.prob

  def __gt__(self, other):
    return self.prob > other.prob

  def merge(self, other): #Merge this node and another, Huffman-style
    return Node(left = self, right = other)


  def encode(self, value):
    if self.value == value:
      return ""
    elif self.value != None:
      return False
    else:
      leftString = self.left.encode(value)
      if leftString != False: #Can't use "if leftString" because the empty string is cast to false
        return "0" + leftString
      rightString = self.right.encode(value)
      if rightString != False:
        return "1" + rightString
      return False

  def decode(self, bits):
    if bits == "":
      return self.value
    elif bits[0] == "0":
      return self.left.decode(bits[1:])
    elif bits[0] == "1":
      return self.right.decode(bits[1:])


def makeTree(xs):
  # xs is a list of tuples of the form (data, frequency)
  # this returns a Huffman tree
  nodes = [Node(prob = x[1], value = x[0]) for x in xs]
  queue = PriorityQueue(0) #The priority queue to manage the letters. No size limit.

  # Create the queue
  for node in nodes:
    queue.put(node)

  # Collapse the queue
  while not queue.empty():
    first = queue.get(block=False)
    if queue.empty(): #If there was only one element, that^ will succeed but now it'll be empty
      return first
    else:
      newNode = first.merge(queue.get(block=False))
      queue.put(newNode)
"""
x = [(1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4), (5, 0.5), (6, 0.6)]
tree = makeTree(x)
print(tree)

code = tree.encode(2)
print(tree.decode(code))
"""
