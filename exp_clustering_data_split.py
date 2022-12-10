#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 17:46:38 2022

@author: soominkwon
"""

from utils import compute_autocovariance, compute_separation, generate_mixed_lds
from subspace_est import subspace_estimation
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from clustering import clustering_fast

# Initial setup 
Ntrial = 30
d = 40
K =  2
rho = 0.5
delta_A = 0.12
Mclustering = 5*d
Msubspace = 5*d
Tclusterings = np.array([10*i for i in range(1,7)])


# To store the parameters of the linear models
As = [[] for _ in range(K)]
Whalfs = [[] for _ in range(K)]
Ws = [[] for _ in range(K)]

R =  linalg.orth(np.random.rand(d,d)) #Random Orthogonal matrix

# Generate linear models using a random orthogonal matrix R
for k in range(K):
    rho_A = rho + ((-1)**k)*delta_A
    As[k] = rho_A * R
    Whalfs[k] = np.identity(d)
    Ws[k] = Whalfs[k]@Whalfs[k]

# To store errors without/with subspace estimation and dimension reduction
error_list_without = np.zeros([len(Tclusterings), Ntrial])
error_list_with = np.zeros([len(Tclusterings), Ntrial])

# Compute Gamma's, Y's, and delta_{Gamma,Y}'s
Gammas, Ys = compute_autocovariance(As,Whalfs)
delta_gy  = compute_separation(Gammas, Ys)
tau = delta_gy/4

for k_T in range(len(Tclusterings)):
    Tclustering = Tclusterings[k_T]
    for k_trial in range(Ntrial): 
        # Generate a mixed LDS
        true_labels = np.random.randint(K,size=[Mclustering,1])
        Ts = np.ones([Mclustering,1])*Tclustering
        data = generate_mixed_lds(As, Whalfs,true_labels,Ts)
        
        #Subspace estimation with independent data
        true_labels_sub = np.random.randint(K,size=[Msubspace,1])
        Ts_sub = np.ones([Msubspace,1])*Tclustering
        data_sub = generate_mixed_lds(As, Whalfs,true_labels_sub,Ts_sub)
        Vs, Us = subspace_estimation(data_sub,K)
        
        # #Subspace estimation
        #Vs, Us = subspace_estimation(data_sub,K)
        
        #0/1 clustering with/without dim reduction
        print("Tclustering:", Tclustering, ",  k_trial:", k_trial, ", No subspace")
        labels_without, S_original, S = clustering_fast(data,Vs,Us, K, tau, no_subspace=1)
        print("Tclustering:", Tclustering, "  k_trial:", k_trial, "With subspace")
        labels_with, S_original, S =  clustering_fast(data,Vs,Us, K, tau, no_subspace=0)
        
        #Note: We are taking the minimum of these two quantities as the predicted labels might be flipped 
        mis_without = min(np.mean(abs(labels_without.squeeze() - true_labels.squeeze())), np.mean(abs(1-labels_without.squeeze() - true_labels.squeeze())))
        mis_with = min(np.mean(abs(labels_with.squeeze() - true_labels.squeeze())), np.mean(abs(1-labels_with.squeeze() - true_labels.squeeze())))
        
        error_list_without[k_T,k_trial] = mis_without 
        error_list_with[k_T,k_trial] = mis_with
        
errors_without = np.mean(error_list_without,axis=1)
errors_with = np.mean(error_list_with,axis=1)

# Plot the errors
plt.plot(Tclusterings,errors_without, 'b--o')
plt.plot(Tclusterings,errors_with,'r-o')
plt.legend(["Without Subspace", "With Subspace"])
plt.xlabel('$T_{\mathrm{clustering}}$')
plt.ylabel("Clustering Error")
plt.grid()
plt.savefig('clustering_error_sample_split.pdf')
plt.show()


