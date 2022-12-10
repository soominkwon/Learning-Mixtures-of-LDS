#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:37:52 2022

@author: sanal
"""

from utils import compute_autocovariance, compute_separation, generate_mixed_lds
from subspace_est import subspace_estimation
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from clustering import clustering_fast

# Initial setup 
Ntrial = 30
d = 5
K =  2
rho = 0.5
delta_A = 0.12
Mclustering = 5*d
Tclusterings = np.array([10*i for i in range(1,10)])


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
error_list_random = np.zeros([len(Tclusterings), Ntrial])
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
                
        #Random subspace clustering
        print("Tclustering:", Tclustering, ",  k_trial:", k_trial, " Random subspace")
        Us, Vs = np.zeros((d,d,K)), np.zeros((d,d,K)) 
        for i in range(d): #Randomly pick Us[i] and Vs[i] as a dxK orthonormal matrix  
            Us[i,:,:] = linalg.orth(np.random.randn(d,d))[:,:K]
            Vs[i,:,:] = linalg.orth(np.random.randn(d,d))[:,:K]
        labels_random,_,_ = clustering_fast(data,Vs,Us, K, tau, no_subspace=0)
        
        #Subspace clustering
        Vs, Us = subspace_estimation(data,K) #Subspace estimation
        print("Tclustering:", Tclustering, "  k_trial:", k_trial, " With subspace")
        labels_with,_,_ =  clustering_fast(data,Vs,Us, K, tau, no_subspace=0)
        
        #Note: We are taking the minimum of these two quantities as the predicted labels might be flipped 
        mis_random = min(np.mean(abs(labels_random.squeeze() - true_labels.squeeze())), np.mean(abs(1-labels_random.squeeze() - true_labels.squeeze())))
        mis_with = min(np.mean(abs(labels_with.squeeze() - true_labels.squeeze())), np.mean(abs(1-labels_with.squeeze() - true_labels.squeeze())))
        
        error_list_random[k_T,k_trial] = mis_random 
        error_list_with[k_T,k_trial] = mis_with
        
errors_random = np.mean(error_list_random,axis=1)
errors_with = np.mean(error_list_with,axis=1)


# Plot the errors
plt.plot(Tclusterings,errors_random, 'b--o')
plt.plot(Tclusterings,errors_with,'r-o')
plt.legend(["random subspace", "with subspace"])
plt.xlabel('Tclustering')
plt.ylabel("clustering error")
plt.show()