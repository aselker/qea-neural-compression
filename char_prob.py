#!/usr/bin/python3

chars = "abcdefghijklmnopqrstuvwxyz !\"'()*,-./0123456789:;?_"
tally_a = [[0 for _ in range(len(chars))] for _ in range(len(chars))]

"""
def char_to_num(char_in):
  i = 0
  for char in chars:
    if char_in == char:
      return num
    i += 1
"""


prev_char = ' '
with open('moby_dick_short.txt') as f:
  for char in f.read():
    tally_a[chars.index(prev_char)][chars.index(char)] += 1
    prev_char = char

def pad(x):
  if x:
    return (" " * (3-len(str(x))) + str(x))
  else:
    return (" " * 3)

maxValue = max(map(max,tally_a))
tally_scaled_a = list(map(lambda x: map(lambda y: pad(round(100 * y / maxValue)), x), tally_a)) #SCALING

for i in range(len(tally_a)):
  print(chars[i] + " | " + " ".join(tally_scaled_a[i]))
