from queue import PriorityQueue

# First, we define a binary tree class which will become our coding. Each node, leaf or not, also has an attached probability.
class Node:
  def __init__(self, prob=0, value=None, left=None, right=None):
    assert (value == None) or (left == None and right == None and prob=0) #Leaf or not a leaf, pls
    self.value = value #value is None if this is not a leaf node
    self.left = left
    self.right = right
    if (left and right):
      prob = left.prob + right.prob

  # In order to be useful in the priority queue, nodes have to be orderable.
  def __lt__(self, other):
    return self.prob < other.prob

  def __gt__(self, other):
    return self.prob > other.prob

  def merge(self, other): #Merge this node and another, Huffman-style
    return Node(left = self, right = other)



def makeCode(xs):
  # xs is a list of tuples of the form (data, frequency)
  # this returns a list of tuples of the form (data, coding)
  nodes = [Node(prob = x[1], value = x[0]) for x in xs]
  queue = PriorityQueue(0) #The priority queue to manage the letters. No size limit.

  # Create the queue
  for node in nodes:
    queue.put(node)

