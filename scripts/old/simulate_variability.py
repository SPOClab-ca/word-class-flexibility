"""
Script to run monte-carlo simulation of expected variation of mean distance from
prototype.

Specifically, in each trial, select N points randomly from a D-dimensional hypercube,
define the prototype as their mean, and find the mean of distances from each point
to the prototype.
"""

import numpy as np

# How many points per simulation
N = 5

# Dimensions of points
D = 2

def gen_vectors():
  return np.random.uniform(low=-1, high=1, size=(N, D))

def do_simulation():
  vecs = gen_vectors()
  v_mean = np.mean(vecs, axis=0)
  scatter = np.sum((vecs - v_mean)**2, axis=1)**0.5
  return np.mean(scatter)

def simulate_many(trials):
  sum_results = 0
  for i in range(trials):
    sum_results += do_simulation()
  return sum_results / trials

for n in range(1, 6):
  for d in range(1, 6):
    N = n
    D = d
    ans = simulate_many(50000)
    print(n, d, ans)
