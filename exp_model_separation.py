#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:34:29 2022

@author: soominkwon
"""

from utils import compute_autocovariance, compute_separation, generate_mixed_lds
from subspace_est import subspace_estimation
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from clustering import clustering_fast

# Initial setup 
Ntrial = 15
d = 40
K =  2
rho = 0.5

delta_As = np.linspace(0.001, 0.25, 12)

#delta_A = 0.12
Mclustering = 5*d
Msubspace = 5*d


Tclusterings = np.array([10*i for i in range(1,7)])
Tclustering = 40

# To store the parameters of the linear models
As = [[] for _ in range(K)]
Whalfs = [[] for _ in range(K)]
Ws = [[] for _ in range(K)]

R =  linalg.orth(np.random.rand(d,d)) #Random Orthogonal matrix

error_list_without = np.zeros([delta_As.shape[0], Ntrial])
error_list_with = np.zeros([delta_As.shape[0], Ntrial])
    
# Generate linear models using a random orthogonal matrix R
for i in range(delta_As.shape[0]):
    for k in range(K):
        rho_A = rho + ((-1)**k)*delta_As[i]
        As[k] = rho_A * R
        Whalfs[k] = np.identity(d)
        Ws[k] = Whalfs[k]@Whalfs[k]
    
    # To store errors without/with subspace estimation and dimension reduction
    #error_list_without = np.zeros([len(Tclusterings), Ntrial])
    #error_list_with = np.zeros([len(Tclusterings), Ntrial])

    
    # Compute Gamma's, Y's, and delta_{Gamma,Y}'s
    Gammas, Ys = compute_autocovariance(As,Whalfs)
    delta_gy  = compute_separation(Gammas, Ys)
    tau = delta_gy/4

    #for k_T in range(len(Tclusterings)):
        #Tclustering = Tclusterings[k_T]
    Tclustering = Tclustering
    
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
        
        
        #0/1 clustering with/without dim reduction
        print("Tclustering:", Tclustering, ",  k_trial:", k_trial, ", No subspace", "model sep:", delta_As[i])
        labels_without, S_original, S = clustering_fast(data,Vs,Us, K, tau, no_subspace=1)
        print("Tclustering:", Tclustering, "  k_trial:", k_trial, "With subspace")
        labels_with, S_original, S =  clustering_fast(data,Vs,Us, K, tau, no_subspace=0)
        
        #Note: We are taking the minimum of these two quantities as the predicted labels might be flipped 
        mis_without = min(np.mean(abs(labels_without.squeeze() - true_labels.squeeze())), np.mean(abs(1-labels_without.squeeze() - true_labels.squeeze())))
        mis_with = min(np.mean(abs(labels_with.squeeze() - true_labels.squeeze())), np.mean(abs(1-labels_with.squeeze() - true_labels.squeeze())))
        
        error_list_without[i,k_trial] = mis_without 
        error_list_with[i,k_trial] = mis_with
        
errors_without = np.mean(error_list_without,axis=1)
errors_with = np.mean(error_list_with,axis=1)


# Plot the errors
#plt.plot(Tclusterings,errors_without, 'b--o')
#plt.plot(Tclusterings,errors_with,'r-o')
plt.plot(delta_As,errors_without, 'b--o')
plt.plot(delta_As,errors_with,'r-o')
plt.legend(["Without Subspace", "With Subspace"])
plt.xlabel('Model Separation')
plt.ylabel("Clustering Error")
plt.grid()
plt.savefig('model_sep_t40_eigh.pdf')
plt.show()