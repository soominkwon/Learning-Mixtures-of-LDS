import numpy as np
from scipy import linalg

def subspace_estimation_helper(data_sub_est):
    """
    Function used to estimate the subspaces.
    
    Parameters:
        data_sub_est: Dataset for subspace estimation with dims (M,d,T+1)

    Returns:
        H_i, G_i:     Eigenspaces each with dimensions (d,d,d)
    """
    
    # Set up data matrix X and dimensions
    X = np.array(data_sub_est)
    X = X.transpose(0,2,1)
    M, Tp1, d = X.shape
    N = int((Tp1-1)/4)
    
    # Initialize h_mij and g_mij
    h_mij = np.zeros([M,d,2,d])
    g_mij = np.zeros([M,d,2,d])
    
    # Make a helper for h_mij that vectorizes the multiplication involved
    h_mij_helper = (np.broadcast_to(X, (d,M,Tp1,d)).transpose(3,1,2,0)*np.broadcast_to(X, (d,M,Tp1,d)))
    
    # Take the means of the helper along the time axis to get h_mij
    h_mij[:,:,0,:] = np.mean(h_mij_helper[:, :, N:2*N, :], axis=2).transpose(1,0,2)
    h_mij[:,:,1,:] = np.mean(h_mij_helper[:, :, 3*N:4*N, :], axis=2).transpose(1,0,2)
    
    # Shift X forward by one timestep
    X_time_shifted = np.concatenate((X[:, 1:, :], np.zeros([M,1,d])), axis = 1)
    
    # Repeat what we did for h_mij to get g_mij
    g_mij_helper = (np.broadcast_to(X_time_shifted, (d,M,Tp1,d)).transpose(3,1,2,0)*np.broadcast_to(X, (d,M,Tp1,d)))
    
    g_mij[:,:,0,:] = np.mean(g_mij_helper[:, :, N:2*N, :], axis=2).transpose(1,0,2)
    g_mij[:,:,1,:] = np.mean(g_mij_helper[:, :, 3*N:4*N, :], axis=2).transpose(1,0,2)
    
    # Vectorizes the sum for H_i and G_i
    H_i = ((h_mij[:,:,0,:]).transpose(1,2,0))@(h_mij[:,:,1,:].transpose(1,0,2))
    G_i = ((g_mij[:,:,0,:]).transpose(1,2,0))@(g_mij[:,:,1,:].transpose(1,0,2))

    # Add transpose
    H_i = (H_i + H_i.transpose(0,2,1))
    G_i = (G_i+ G_i.transpose(0,2,1))
    
    return H_i, G_i
    
def subspace_estimation(data_sub_est, K):
    """
    Function to perform subspace estimation using helper function.
    
    Parameters:
        data_sub_est: Dataset for subspace estimation with dims (M,d,T+1)

    Returns:
        V_i, U_i:     Orthogonal projectors to the desired subspaces with dims (d,d,K)
    """
    
    H_i, G_i = subspace_estimation_helper(data_sub_est)

    X = np.array(data_sub_est).transpose(0,2,1)
    M,Tp1,d = X.shape
    
    # Initialize V_i and U_i
    V_i = np.zeros([d,d,K])
    U_i = np.zeros([d,d,K])
    
    # Get eigenvectors
    for i in range(d):
        V_i[i, :, :] = linalg.svd(H_i[i, :, :])[0][:,:K]
        U_i[i,:,:] = linalg.svd(G_i[i,:,:])[0][:,:K]

    return V_i, U_i
