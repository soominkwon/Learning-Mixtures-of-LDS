import numpy as np

def model_estimation(clusters):
    """
    Function for least squares and covariance estimation.
    
    Parameters:
        clusters:   List of lists of size m of data matrices of size (d, T)
            
    Returns:
        Ahats:      List of estimated A models
        Whats:      List of estimated W covariance matrices
    """
    
    # initializing
    K = len(clusters)
    d = clusters[0][0].shape[0]

    Ahats = []
    Whats = []
    
    # estimating matrices for each class
    for k in range(K):
        # computing Ahat
        xxt = np.zeros((d, d))
        xpxt = np.zeros((d, d))

        # computing for each data matrix in each cluster
        for m in range(len(clusters[k])):
            T = clusters[k][m].shape[1]
            tmp = clusters[k][m][:, :T-1]
            tmpp = clusters[k][m][:, 1:]
            xxt = xxt + tmp @ tmp.T
            xpxt = xpxt + tmpp @ tmp.T

        Ahat = xpxt @ np.linalg.inv(xxt)
        Ahats.append(Ahat)

        # computing What
        tmpT = 0
        What = np.zeros((d, d))

        # computing for each data matrix in each cluster
        for m in range(len(clusters[k])):
            T = clusters[k][m].shape[1]
            tmp = clusters[k][m][:, :T-1]
            tmpp = clusters[k][m][:, 1:]
            whats = tmpp - Ahat @ tmp
            What = What + whats @ whats.T
            tmpT = tmpT + tmp.shape[1]

        What = What / tmpT
        Whats.append(What)

    return Ahats, Whats