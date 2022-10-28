#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 21:43:31 2022

@author: soominkwon
"""

import numpy as np
from scipy import linalg

def classification(data_classification, Ahats, Whats):
    """
    Infers the label of the m-th trajectory.
    
    Parameters:
        data_classification:    Trajectories corresponding to classification
        Ahats:                 List of estimated linear models
        Whats:                 List of estimated covariance matrices
    
    Returns:
        labels:                 New labels for cluster
    """
    
    # initializing
    M_class = len(data_classification)
    K = len(Ahats)
    labels = np.zeros((M_class, ))
    
    # for each trajectory
    for m in range(M_class):
        X = data_classification[m]
        losses = np.zeros((K, 1))
        
        for k in range(K):
            Ahat = Ahats[k]
            What = Whats[k]
            
            T = X.shape[1] - 1
            
            res = X[:, 1:(X.shape[1])] - Ahat @ X[:, 0:(X.shape[1]-1)]
            tmp = linalg.lstsq(What, res)

            loss = T * np.log(linalg.det(What)) + np.trace(res.T @ tmp[0])
            
            losses[k] = loss

        # finding index of minimum loss
        labels[m] = np.argmin(losses)

    return labels

