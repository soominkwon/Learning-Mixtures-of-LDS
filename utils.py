import numpy as np
from scipy import linalg


def generate_models(d, K, rho):
    """
    Creates the latent variables A and W.
    
    Parameters:
        d:      Dimension of each trajectory
        K:      Number of labels
        rho:    Contraction rate (bound on A)
    
    Returns:
        As:      List of linear systems
        Whalfs:  List of square root of covariance matrices
    """
    
    # initialize lists of latent variables
    As = []
    Whalfs = []
    
    # generate variables for each label
    for i in range(K):
        # generate A matrix
        R = linalg.orth(np.random.randn(d, d))
        A = rho * R
        
        # generate W matrix
        U = linalg.orth(np.random.randn(d, d))
        L = np.diag((1.0 + np.random.randn(d, )).squeeze())
        
        # eigendecomposition of W
        W = U @ L @ U.T
        
        # gathering all matrices
        As.append(A)
        Whalfs.append(W)
        
    return As, Whalfs


def compute_autocovariance(As, Whalfs):
    """
    Computes the order 0 and 1 autocovariance matrices Gamma and Y.
    
    Parameters:
        As:         List of linear systems
        Whalfs:     List of square root of covariance matrices
        
    Returns:
        Gammas:     List of order-0 autocovariance matrices
        Ys:         List of order-1 autocovariance matrices
    """

    # initializing variables
    K = len(As)
    Gammas = []
    Ys = []
    
    # generating for all labels K
    for i in range(K):
        W = Whalfs[i] @ Whalfs[i]
        Gamma = linalg.solve_discrete_lyapunov(As[i], W)
        Y = As[i] @ Gamma
        
        # gathering all matrices
        Gammas.append(Gamma)
        Ys.append(Y)
        
    return Gammas, Ys


def compute_separation(Gammas, Ys):
    """
    Computes separation parameter of latent variables.
    
    Parameters:
        Gammas:     List of order-0 autocovariance matrices
        Ys:         List of order-1 autocovariance matrices
    
    Returns:
        delta_gy:   Separation parameter
    
    """
    
    # initializing
    K = len(Gammas)
    delta_gy = np.inf
    
    # computing for each label
    for k in range(K):
        gamma_k = Gammas[k]
        Y_k = Ys[k]
        
        for l in range(k+1, K):
            gamma_l = Gammas[l]
            Y_l = Ys[l]
            
            # computing delta_gy
            temp = np.linalg.norm(gamma_k - gamma_l, 'fro')**2 + np.linalg.norm(Y_k - Y_l, 'fro')**2
            
            delta_gy = min(delta_gy, temp) # updating delta_gy
            
    return delta_gy


def generate_lds(A, Whalf, x0, T):
    """
    Generate a sample trajectories by the LDS model A and Whalf.
    
    Parameters:
        A:              Latent variable or linear system
        Whalf:          Square root of covariance matrix
        x0:             Initial starting point
        T:              Number of trajectories for model A and Whalf
    
    Returns:
        trajectories:   Sample trajectories from model A and Whalf
    """
    
    # initializing
    T = int(T)
    d = x0.shape[0]
    trajectories = np.zeros((d, T+1))
    trajectories[:, 0] = x0.squeeze() # initializing first trajectory
    
    x = x0 # initializing trajectory
    
    # generating trajectories
    for t in range(T):
        x = A @ x + Whalf @ np.random.randn(d, ) # updating trajectory
        trajectories[:, t+1] = x.squeeze()
        
    return trajectories


def generate_mixed_lds(As, Whalfs, true_labels, Ts):
    """
    Generates a mixture of sample trajectories given latent variables.
    
    Parameters:
        As:             List of linear systems
        Whalfs:         List of square root of covariance matrices
        true_labels:    Array of true labels for each trajectory
        Ts:             Array of trajectory lengths for each subroutine         
    
    Returns:
        data:           Data of mixture of sample trajectories
    """

    # initializing
    M = len(true_labels) # total number of sample trajectories
    d = As[0].shape[0] # dimension of each trajectory
    data = []
    x0 = np.zeros((d, ))

    # generating all trajectories
    for m in range(M):
        k_m = true_labels[m][0] # label of the mth trajectory
        T_m = int(Ts[m][0]) # length of the mth trajectory
        traj = generate_lds(As[k_m], Whalfs[k_m], x0, T_m)

        # updating trajectory
        data.append(traj)
        x0 = traj[:, T_m]

    return data


if __name__ == "__main__":
    # initializing parameters
    d   = 30
    K   = 4
    rho = 0.5
    
    Msubspace        = 30  * d
    Mclustering      = 10  * d
    Mclassification  = 50 * d
    M = Msubspace + Mclustering + Mclassification
    
    Tsubspace        = 20
    Tclustering      = 20
    Tclassification  = 5
    
    # generating labels and lengths of trajectories
    true_labels = np.random.randint(1, K, (M, 1))
    Ts = np.concatenate([np.ones((Msubspace,1))*Tsubspace, np.ones((Mclustering,1))*Tclustering,
                         np.ones((Mclassification,1))*Tclassification], axis=0)
    
    # generating synthetic data
    As, Whalfs = generate_models(d=d, K=K, rho=rho)
    Gammas, Ys = compute_autocovariance(As=As, Whalfs=Whalfs)
    # delta_gy = compute_separation(Gammas=Gammas, Ys=Ys)
    data_matrix = generate_mixed_lds(As=As, Whalfs=Whalfs, true_labels=true_labels, Ts=Ts)

        
    