import csv
import matplotlib.pyplot as plt
import numpy as np
from subspace_est import subspace_estimation
from clustering import clustering_fast
from sklearn.cluster import SpectralClustering

"""Preprocessing"""

# Initialize d, K, M, T
d = 12
K = 2
T = 400
M=12

# Open the files    
file1 = open('jog.csv')
file2 = open('wlk_sub_23.csv')

# Initialize the readers and skip the headers
jog_reader = csv.reader(file1)
walk_reader = csv.reader(file2)
jog_header = next(jog_reader)
walk_header = next(walk_reader)

# Accumulate data from the rows, skip the serial number (first element)
jog_rows = []
for row in jog_reader:
    jog_rows.append([float(i) for i in row[1:]])
walk_rows = []
for row in walk_reader:
    walk_rows.append([float(i) for i in row[1:]])

# Convert to arrays
jog_rows = np.array(jog_rows)
walk_rows = np.array(walk_rows)



# Initialize a numpy array containing 24 (12 x 400) matrices 
# Shape (24, 12, 400)
combined_data_subspace = np.zeros([M,d,T])
combined_data_clustering = np.zeros([M,d,T])

M_half = int(M/2)


# Take blocks of 400 timesteps from jog_rows and walk_rows to add to the combined data for subspace estimation
for i in range(M_half):
    combined_data_subspace[i, :, :] = np.array(jog_rows[400*i:400*(i+1),:]).transpose()
for i in range(M_half):
    combined_data_subspace[(M_half+i), :, :] = np.array(walk_rows[400*i:400*(i+1),:]).transpose()

# Take blocks of 400 timesteps from jog_rows and walk_rows to add to the combined data for clustering
for i in range(M_half):
    combined_data_clustering[i, :, :] = np.array(jog_rows[400*(M_half+i):400*(M_half+i+1),:]).transpose()
for i in range(M_half):
    combined_data_clustering[(M_half+i), :, :] = np.array(walk_rows[400*(M_half+i):400*(M_half+i+1),:]).transpose()
    
"""End of Preprocessing"""

# Choose sample trajectories for jogging and walking, to be plotted below
sample_jog_traj = combined_data_subspace[0, :, :]
sample_walk_traj = combined_data_clustering[M-1, :, :]

# Plot and save a figure for the sample jogging trajectory
fig, ax = plt.subplots()

for i in range(d):
    ax.plot(sample_jog_traj[i,:], linewidth = 0.5)
ax.set(xlabel = "Time step", ylabel = "Measurement", title = "Mode 1: jogging")
fig.savefig("jog_fig.png")
plt.close()


# Plot and save a figure for the sample walking trajectory
fig, ax = plt.subplots()

for i in range(d):
    ax.plot(sample_walk_traj[i,:], linewidth = 0.5)
ax.set(xlabel = "Time step", ylabel = "Measurement", title = "Mode 2: walking")
fig.savefig("walk_fig.png")
plt.close()

# Color map of the original matrix
Vs, Us = subspace_estimation(combined_data_subspace,K)
tau = 50
labels, S_original, S = clustering_fast(combined_data_clustering, Vs, Us, K, tau, 0)
fig, ax = plt.subplots()
ax.set(xlabel = "Trajectories", ylabel = "Trajectories", title = "Color map of the original matrix")
cax = ax.matshow(S_original, cmap = "gray")
ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
fig.colorbar(cax, ax=ax)
fig.savefig("gray_color_map_split.pdf")
plt.close()

# Histogram of computed distances 
newdata = S_original.reshape((combined_data_clustering.shape[0])**2)
fig, ax = plt.subplots()
ax.hist(newdata, bins = 40)
ax.set(xlabel = "Computed Distance", ylabel= "Frequency", title="Histogram of Computed Distances")
ax.grid()
fig.savefig("hist_computed_dists_split.pdf")
plt.close()
# Suggests tau = 110

# Color map of the true S
fig, ax = plt.subplots()
ax.set(xlabel = "Trajectories", ylabel = "Trajectories", title = "Color Map of the Thresholded Matrix")
cax = ax.matshow(S, cmap = "gray")
ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
fig.colorbar(cax, ax=ax)
fig.savefig("gray_color_map_true_split.pdf")
plt.close()

# Misclassification rate
clustering = SpectralClustering(n_clusters=K,affinity='precomputed').fit(S)
labels = clustering.labels_.reshape(-1,)
true_labels = np.concatenate((np.zeros(6), np.ones(6)))
print("The misclassification rate is", min(np.mean(abs(labels- true_labels)), np.mean(abs((1-labels) - true_labels))))
