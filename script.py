#!/usr/bin/env python3
# script.py: Implement EM algorithm from scratch.
# To run, please pass the number of Gaussians as the system argument variable.
# Ankit Mathur, February 2019

import sys
import random
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Read data
x = []
with open('data1.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        x += [ float(d) for d in line.split() ]

# Initiate Gaussians
global pc
global gauss

k = int(sys.argv[1])
n = len(x)

np.random.seed(10)
mu = np.array(np.random.random(k))
sig = np.array(np.ones(k))
pc = np.array(np.ones(k)/k)

gauss = []
for i in range(k):
    gauss += [ norm(loc=mu[i], scale=sig[i]) ]

def outer(thres=0.001):
    global pc
    global gauss

    llhood_arr = []

    n_iter, llhood, prev_llhood = 0, 1, 0
    while(llhood - prev_llhood > thres):
        prob = cal_prob(gauss)
        prev_llhood = cal_llhood(prob)

        score, score_agg = e_step(gauss, pc, prob)
        pc, gauss = m_step(score, score_agg)

        n_iter += 1
        prob = cal_prob(gauss)
        llhood = cal_llhood(prob)
        llhood_arr += [llhood]

    print("Converged at iteration " + str(n_iter))
    print("Details:")
    for i in range(k):
        print("Gaussian " + str(i+1) + ":")
        print("mean: " + str(gauss[i].mean()))
        print("std dev: " + str(gauss[i].std()))
        print("weight: " + str(pc[i]))
    plt.plot(range(1, n_iter+1), llhood_arr, color='green', marker='+', markersize=5)
    plt.xlabel('number of iterations')
    plt.ylabel('Log-likelihood')
    plt.show()

def e_step(gauss, pc, prob):
    pc_arr = np.array( [ [ pc[i] ] for i in range(k) ] )
    numerator = prob * pc_arr
    denominator = np.sum(numerator, axis=0)
    score = numerator / denominator
    score_agg = np.sum(score, axis=1)
    return score, score_agg

def m_step(score, score_agg):
    x_arr = np.array(x)
    new_mu = np.sum(x_arr * score, axis=1) / score_agg

    t1 = np.sum(np.square(x_arr) * score, axis=1) / score_agg
    t2 = np.square( np.sum(x_arr * score, axis=1) / score_agg )
    new_sig = np.sqrt(t1-t2)

    new_pc = score_agg / np.sum(score_agg, axis=0)

    new_gauss = []
    for i in range(k):
        new_gauss += [ norm(loc=new_mu[i], scale=new_sig[i]) ]
    return new_pc, new_gauss

def cal_llhood(prob):
    pc_arr = np.array( [ [ pc[i] ] for i in range(k) ] ).transpose()
    sum_prod = np.dot(pc_arr, prob)
    log_val = np.log(sum_prod)
    llhood = np.sum(log_val)
    return llhood

def cal_prob(gauss):
    prob = np.array( [ [ gauss[i].pdf(x[j]) for j in range(n) ] for i in range(k) ] )
    return prob

outer()
