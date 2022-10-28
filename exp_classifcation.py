import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from utils import compute_autocovariance, compute_separation, generate_mixed_lds
from subspace_est import subspace_estimation
from clustering import clustering_fast
from model_estimation import model_estimation
from classification import classification
import time

# recording time
start_time = time.time()

# initializing parameters
Ntrial = 24
d = 40
K = 2
rho = 0.5
delta_A = 0.12

Mclustering = 10 * d
Mclassification = 30

Tclustering = 30
Tclassifications = range(4,50,4)

errors = np.zeros([len(Tclassifications), Ntrial])

As = []
Whalfs = []
Ws = []
R = linalg.orth(np.random.rand(d,d))

for k in range(K):
    rho_A = rho + ((-1)**k)*delta_A
    As.append(rho_A*R)
    Whalfs.append(np.identity(d))
    Ws.append(Whalfs[k]@Whalfs[k])
    
Gammas, Ys = compute_autocovariance(As,Whalfs)
delta_gy = compute_separation(Gammas, Ys)

# clustering and model estimation
true_labels_clustering = np.random.randint(K,size=[Mclustering,1])
Ts_clustering = np.ones([Mclustering,1])*Tclustering
data_clustering = generate_mixed_lds(As, Whalfs, true_labels_clustering, Ts_clustering)

# Note: Chen and Poor's code uses clustering data for subspace estimation
Vs, Us = subspace_estimation(data_clustering, K)
tau = delta_gy/4
labels_clustering = clustering_fast(data_clustering, Vs, Us, K, tau, no_subspace=0)[0]

# check whether labels have been permuted or not
tmp1 = np.mean(np.abs(labels_clustering-true_labels_clustering))
tmp2 = np.mean(np.abs(1-labels_clustering-true_labels_clustering))

if tmp1 < tmp2: 
    label_perm = [0,1]
else:
    label_perm = [1,0]

clusters = []
for k in range(K):
    clusters.append(np.array(data_clustering)[(labels_clustering == k).reshape(-1,)])

# printing clustering Error 
print('Clustering Error: ', min(tmp1,tmp2))

# coarse model estimation
Ahats, Whats = model_estimation(clusters)

for k_classification, Tclassification in enumerate(Tclassifications):
    for k_trial in range(Ntrial):
        true_labels_classification = np.random.randint(K,size=[Mclassification,1])
        Ts_classification = np.ones([Mclassification,1])*Tclassification
        data_classification = generate_mixed_lds(As, Whalfs, true_labels_classification, Ts_classification)
        labels_classification = classification(data_classification,Ahats,Whats)
        
        true_labels_perm = np.zeros([Mclassification,1])
        for k in range(K):
            true_labels_perm[true_labels_classification == k] = label_perm[k] 
        
        misclassification = np.mean(np.abs(true_labels_perm.squeeze() - labels_classification.squeeze()))
        errors[k_classification,k_trial] = misclassification

mean_error = np.mean(errors,axis=1)

# plotting
plt.figure()
plt.plot(Tclassifications,mean_error,'-o')
plt.xlabel('$T_{\mathrm{classification}}$')
plt.ylabel('Classification Error')
plt.grid()
plt.savefig('classification_error.pdf')
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
