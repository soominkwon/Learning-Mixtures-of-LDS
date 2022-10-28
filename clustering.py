from utils import compute_autocovariance, compute_separation, generate_mixed_lds
from subspace_est import subspace_estimation
import numpy as np
from scipy import linalg
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

def clustering_fast(data, Vs, Us, K, tau, no_subspace):
    """
    Function for the clustering algorithm.
    
    Parameters:
        data:        Mixed LDS data
        Vs, Us:      List of estimated subspaces, Vs.shape = (d,d,K), Us.shape=(d,d,K)
        K:           Number of clusters
        tau:         Separation parameter
        no_subspace: 0/1 depending on whether we want to cluster with or without subspace estimation
        
    Returns:
        labels:      Predicted labels of shape (M,1)
    """
    
    # initializing parameters
    M = len(data)
    T = data[0].shape[1] - 1
    N = int(np.floor(T/4))
    
    tmp1_Gammas = []
    tmp2_Gammas = []
    tmp1_Ys = []
    tmp2_Ys = []
     
    for m in range(M):
        X = data[m]  # X.shape = (d,T+1)
        X1 = X[:, N:2*N]  # X1.shape = (d,N)
        X2 = X[:, 3*N:4*N]  # X2.shape = (d,N)
        Xp1 = X[:, N+1:2*N+1]  # Xp1.shape = (d,N)
        Xp2 = X[:, 3*N+1:4*N+1]  # Xp2.shape = (d,N)
        
        tmp1_Gammas.append(X1 @ X1.T / N) #tmp1_Gammas[i].shape = (d,d)
        tmp2_Gammas.append(X2 @ X2.T / N) #tmp2_Gammas[i].shape = (d,d)
        tmp1_Ys.append(X1 @ Xp1.T / N) #tmp1_Ys[i].shape = (d,d)
        tmp2_Ys.append(X2 @ Xp2.T / N) #tmp2_Ys[i].shape = (d,d)
    
    # Convert to numpy arrays
    tmp1_Gammas = np.array(tmp1_Gammas) #tmp1_Gammas.shape = (M,d,d)
    tmp2_Gammas = np.array(tmp2_Gammas) #tmp2_Gammas.shape = (M,d,d)
    tmp1_Ys = np.array(tmp1_Ys) #tmp1_Ys.shape = (M,d,d)
    tmp2_Ys = np.array(tmp2_Ys) #tmp2_Ys.shape = (M,d,d)
    
    S_original = np.zeros([M,M])
    
    if no_subspace: # if dimension reduction is not used
        t1G_repeat = tmp1_Gammas[:,:,:,np.newaxis].repeat(M,axis=3)
        t2G_repeat = tmp2_Gammas[:,:,:,np.newaxis].repeat(M,axis=3)
        t1Y_repeat = tmp1_Ys[:,:,:,np.newaxis].repeat(M,axis=3)
        t2Y_repeat = tmp2_Ys[:,:,:,np.newaxis].repeat(M,axis=3)
        S_original += np.sum((t1G_repeat-t1G_repeat.transpose(3,1,2,0))*(t2G_repeat-t2G_repeat.transpose(3,1,2,0)), axis = (1,2))
        S_original += np.sum((t1Y_repeat-t1Y_repeat.transpose(3,1,2,0))*(t2Y_repeat-t2Y_repeat.transpose(3,1,2,0)), axis = (1,2))
    else: # with dimension reduction
        Vs_repeat = Vs[np.newaxis,:,:,:].repeat(M,axis=0).transpose(0,2,1,3) #Vs_repeat.shape = (M,d,d,K)
        Us_repeat = Us[np.newaxis,:,:,:].repeat(M,axis=0).transpose(0,2,1,3) #Vs_repeat.shape = (M,d,d,K)    
        t1G_repeat = np.repeat(tmp1_Gammas[:,:,:,np.newaxis],K,axis=3) #t1G_repeat.shape = (M,d,d,K)
        t2G_repeat = np.repeat(tmp2_Gammas[:,:,:,np.newaxis],K,axis=3)#t2G_repeat.shape = (M,d,d,K)
        t1Y_repeat = np.repeat(tmp1_Ys[:,:,:,np.newaxis],K,axis=3) #t1Y_repeat.shape = (M,d,d,K)
        t2Y_repeat = np.repeat(tmp2_Ys[:,:,:,np.newaxis],K,axis=3)     #t1Y_repeat.shape = (M,d,d,K)   
        V_times_t1G = np.repeat(np.sum(Vs_repeat*t1G_repeat,axis=1)[:,:,:,np.newaxis],M,axis=3) #V_times_t1G.shape =  (M,d,K,M)
        V_times_t2G = np.repeat(np.sum(Vs_repeat*t2G_repeat,axis=1)[:,:,:,np.newaxis],M,axis=3) #V_times_t2G.shape =  (M,d,K,M)
        U_times_t1Y = np.repeat(np.sum(Us_repeat*t1Y_repeat,axis=1)[:,:,:,np.newaxis],M,axis=3) #V_times_t1Y.shape =  (M,d,K,M)
        U_times_t2Y = np.repeat(np.sum(Us_repeat*t2Y_repeat,axis=1)[:,:,:,np.newaxis],M,axis=3) #V_times_t2Y.shape =  (M,d,K,M)
        V_times_t1Gdiff = V_times_t1G - V_times_t1G.transpose(3,1,2,0) #V_times_t1Gdiff.shape =  (M,d,K,M) 
        V_times_t2Gdiff = V_times_t2G - V_times_t2G.transpose(3,1,2,0) #V_times_t1Gdiff.shape =  (M,d,K,M)
        U_times_t1Ydiff = U_times_t1Y - U_times_t1Y.transpose(3,1,2,0) #V_times_t1Gdiff.shape =  (M,d,K,M)
        U_times_t2Ydiff = U_times_t2Y - U_times_t2Y.transpose(3,1,2,0) #V_times_t1Gdiff.shape =  (M,d,K,M)
        S_original += np.sum(V_times_t1Gdiff*V_times_t2Gdiff, axis=(1,2)) #S_original.shape = (M,M)
        S_original += np.sum(U_times_t1Ydiff*U_times_t2Ydiff, axis=(1,2))

    # Compute the zero-one similarity matrix
    def less_than_tau(x): #Auxilliary function
        return 0 if x > tau else 1
    
    S = np.vectorize(less_than_tau)(S_original)

    # Manual KMeans
    eig_vals, eig_vecs = linalg.eigh(S)
    K_largest_mag_idx = sorted(range(M), key = lambda idx: -abs(eig_vals[idx]))[:K]
    U = eig_vecs[:,K_largest_mag_idx]
    centers = np.zeros([K,K])
    
    for k in range(K):
        centers[k][k] = U[:,k][np.argmax(np.abs(U[:,k]))]

    U = U.T #U.shape = (K,M)
    distances = np.zeros((K,M))
    for t in range(50):
        for k in range(K):
            center = centers[:,[k]] #center.shape = (K,1)
            res = U - np.repeat(center,M,axis=1) #res.shape = (K,M)
            distances[k,:] = np.sum(res**2,axis=0)
        labels = np.argmin(distances,axis=0) #labels.shape = (M,)
        for k in range(K):
            subU = U[:,labels == k]
            centers[:,k] = np.mean(subU, axis=1)
    
    labels = labels.reshape(-1,1) #shape = (M,1)
    
    return labels, S_original, S

