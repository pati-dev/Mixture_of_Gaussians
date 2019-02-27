#!/usr/bin/env python3
# script.py: Implementing GMM from scratch.
# Ankit Mathur, February 2019

import sys
import random
import numpy as np
from scipy.stats import norm

# Read data
x = []
with open('data1.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        x += [float(d) for d in line.split()]

# Initiate Gaussians
global pc
global gauss
k = int(sys.argv[1])
n = len(x)
np.random.seed(69)
mu = list(np.random.random(k))
sig = list(np.ones(k))
pc = list(np.ones(k)/k)
gauss = []
for i in range(k):
    gauss += [ norm(loc=mu[i], scale=sig[i]) ]

def outer(thres=0.001):
    global pc
    global gauss
    n_iter, llhood, prev_llhood = 0, 1, 0
    while(llhood - prev_llhood > thres):
        prob = cal_prob()
        prev_llhood = cal_llhood(prob)
        n_pts = e_step(gauss, pc, prob)
        pc, gauss = m_step(score)
        n_iter += 1
        llhood = cal_llhood()
        print("Log likelihood for iteration " + str(n_iter) +" is " + str(round(llhood,5)))
    print("Stopped at iteration " + str(n_iter))

def e_step(gauss, pc, prob):
    pc_arr = np.array( [ [ pc[i] ] for i in range(k) ] )
    numerator = prob * pc_arr
    denominator = np.sum(numerator, axis=0)
    score = numerator / denominator
    temp = np.sum(score, axis=1)
    return n_pts

def m_step(n_pts):
    global pc
    global gauss
    for i in range(k):
        temp_mean, temp_std = 0, 0
        # for j in range(n):

    return pc, gauss

def cal_llhood(prob):
    pc_arr = np.array( [ [ pc[i] ] for i in range(k) ] ).transpose()
    sum_prod = np.dot(pc_arr, prob)
    log_val = np.log(sum_prod)
    llhood = np.sum(log_val)
    return llhood

def cal_prob():
    prob = np.array( [ [ gauss[i].pdf(x[j]) for j in range(n) ] for i in range(k) ] )
    return prob
outer()
