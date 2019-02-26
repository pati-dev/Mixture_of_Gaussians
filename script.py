#!/usr/bin/env python3
# script.py: Implementing GMM from scratch.
# Ankit Mathur, February 2019

import sys
import random
import numpy as np

# Read data
x = []
with open('data1.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        x += [d for d in line.split()]

# Initiate Gaussians
k = int(sys.argv[1])
n = len(x)
np.random.seed(69)
mu = list(np.random.random(k))
sig = list(np.ones(k))
pc = list(np.ones(k)/k)

def outer(thres=0.001):
    n_iter, lhood, prev_lhood = 0, 1
    while(lhood - prev_lhood > thres):
        prev_lhood = cal_lhood()
        e_step()
        m_step()
        n_iter += 1
        lhood = cal_lhood()
        print("Log likelihood for iteration " + str(n_iter) +" is " + str(round(lhood,5)))


def e_step():
    # foo

def m_step():
    # foo

import pdb; pdb.set_trace()
