# https://visualstudiomagazine.com/articles/2021/08/16/wasserstein-distance.aspx

# wasserstein_demo.py
# Wasserstein distance from scratch

import numpy as np
from numba import jit

@jit
def first_nonzero(vec):
  dim = len(vec)
  for i in range(dim):
    if vec[i] > 0.0:
      return i
  return -1  # no empty cells found


@jit
def move_dirt(dirt, di, holes, hi):
  # move as much dirt at [di] as possible to h[hi]
  if dirt[di] <= holes[hi]:   # use all dirt
    flow = dirt[di]
    dirt[di] = 0.0            # all dirt got moved
    holes[hi] -= flow         # less to fill now
  elif dirt[di] > holes[hi]:  # use just part of dirt
    flow = holes[hi]          # fill remainder of hole
    dirt[di] -= flow          # less dirt left
    holes[hi] = 0.0           # hole is filled
  dist = np.abs(di - hi)
  return flow * dist          # work


@jit
def my_wasserstein(p, q):
  dirt = np.copy(p) 
  holes = np.copy(q)
  tot_work = 0.0

  while True:  # TODO: add sanity counter check
    from_idx = first_nonzero(dirt)
    to_idx = first_nonzero(holes)
    if from_idx == -1 or to_idx == -1:
      break
    work = move_dirt(dirt, from_idx, holes, to_idx)
    tot_work += work
  return tot_work  


@jit
def kullback_leibler(p, q):
  n = len(p)
  sum = 0.0
  for i in range(n):
    sum += p[i] * np.log(p[i] / q[i])
  return sum


@jit
def symmetric_kullback(p, q):
  a = kullback_leibler(p, q)
  b = kullback_leibler(q, p)
  return a + b

def main():
  print("\nBegin Wasserstein distance demo ")

  P =  np.array([0.6, 0.1, 0.1, 0.1, 0.1])
  Q1 = np.array([0.1, 0.1, 0.6, 0.1, 0.1])
  Q2 = np.array([0.1, 0.1, 0.1, 0.1, 0.6])

  kl_p_q1 = symmetric_kullback(P, Q1)
  kl_p_q2 = symmetric_kullback(P, Q2)

  wass_p_q1 = my_wasserstein(P, Q1)
  wass_p_q2 = my_wasserstein(P, Q2)

  print("\nKullback-Leibler distances: ")
  print("P to Q1 : %0.4f " % kl_p_q1)
  print("P to Q2 : %0.4f " % kl_p_q2)

  print("\nWasserstein distances: ")
  print("P to Q1 : %0.4f " % wass_p_q1)
  print("P to Q2 : %0.4f " % wass_p_q2)

  print("\nEnd demo ")

if __name__ == "__main__":
  main()