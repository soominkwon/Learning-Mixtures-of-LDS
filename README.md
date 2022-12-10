# Learning Mixtures of Linear Dynamical Systems
Implementation of Learning Mixtures of Linear Dynamical Systems in Python. 

Author(s): Soo Min Kwon, Chinmaya Kausik, Nancy Wang, Sanal Shivaprasad

This implementation reproduces the figures in Chen and Poor's paper "Learning Mixtures of Linear Dynamical Systems." If you have all the dependencies installed (i.e. matplotlib), then simply running the code should generate the figures. The following is a brief explanation of the associated files:

- exp_clustering.py: Used to generate Figure 1 of reproducibility paper
- exp_classification.py: Used to generate Figure 2 of reproducibility paper
- exp_validate.py: Used to generate Figure 3 of reproducibility paper
- exp_motion_sense.py: Used to generate Figure 4 of reproducibility paper
- exp_motion_sense_split.py: Used to generate Figure 5 of reproducibility paper
- exp_model_separation.py: Used to generate Figure 6 of reproducibility paper
- exp_random_subspace.py: Used to generate Figure 7 of reproducibility paper
- exp_clustering_data_split.py: Used to generate Figure 9 of reproducibility paper

To generate Figure 8, one can simply save the eigenvalues and singular values from subspace_est.py and plot using matplotlib. The following are data for MotionSense:
- jog.csv
- wlk_sub_23.csv

The rest of the code is the implementation of the algorithm.

We lost track of who wrote which code in the codebase, but it was an even distribution of work!

