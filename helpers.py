import numpy as np

def get_clusters(data, labels, K):
    """
    Function to obtain the data that correspond the labels.
    
    Parameters:
        data:           data to separate into predicted labels
        labels:         predicted labels used to cluster the data
        K:              number of classes
    
    Returns:
        clustered_data: list of clustered data
    """
    
    # initializing
    clustered_data = []
    
    # clustering for each class
    for k in range(K):
       class_label = []
       
       # looping through each labels
       for i in range(len(labels)):
           if labels[i] == k:
               class_label.append(data[i])
       
       # turning into an array before adding to list
       clustered_data.append(class_label)
    
    return clustered_data

def model_errors(Ahats, As, Whats, Ws, invperm):
    """
    Function to obtain errors for estimated A and W.
    
    Parameters:
        Ahats:      List of estimated models A
        As:         List of true A
        Whats       List of estimated covariance matrices W
        Ws:         List of true W
        
    Returns:
        max_A_error:    Model error for A
        max_W_error:    Covariance error for W
        
    """

    # initializing
    K = len(Ahats)
    A_errors = []
    W_errors = []
    
    for k in range(K):
        invk = int(invperm[k])
        A_error = np.linalg.norm(Ahats[k] - As[invk], 2)
        A_errors.append(A_error)
    
        W_error = np.linalg.norm(Whats[k] - Ws[invk], 2) / np.linalg.norm(Ws[invk], 2)
        W_errors.append(W_error)
    
    max_A_error = max(A_errors)
    max_W_error = max(W_errors)    
    
    return max_A_error, max_W_error
    
    
    