# %%
import cvxpy as cp
import numpy as np

def run_sdp_clustering(task_affinities, k, use_exp=True, temperature=1.0):
    if use_exp:
        task_affinities = np.exp(task_affinities/temperature)
    def sdp_clustering(T, k):
        n = T.shape[0]

        A = []
        b = []
        # first constraint 
        A.append(np.eye(n))
        b.append(k)

        # second constraint
        for i in range(n):
            tmp_A = np.zeros((n, n))
            tmp_A[:, i] = 1
            A.append(tmp_A)
            b.append(1)

        # Define and solve the CVXPY problem.
        # Create a symmetric matrix variable.
        X = cp.Variable((n,n), symmetric=True)
        # The operator >> denotes matrix inequality.
        constraints = [X >> 0, X>=0]
        constraints += [
            cp.trace(A[i] @ X) == b[i] for i in range(len(A))
        ]
        prob = cp.Problem(cp.Minimize(cp.trace(T @ X)),
                        constraints)
        prob.solve()

        # Print result.
        print("The optimal value is", prob.value)
        X_final = X.value
        X_final = X_final > 1/n
        return X_final, X.value

    maximum = np.max(task_affinities)
    X_final, X_value = sdp_clustering(maximum-task_affinities, k)

    # generate cluster labels
    assignment = {}; cluster_idx = 0; assigned_before = np.zeros(X_final.shape[0])
    for i in range(X_final.shape[0]):
        assigned_count = 0
        for j in range(i, X_final.shape[1]):
            if X_final[i, j] and assigned_before[j] == 0:
                if assigned_before[i] == 0: 
                    if cluster_idx in assignment:
                        assignment[cluster_idx].append(i) 
                    else:
                        assignment[cluster_idx] = [i]
                    assigned_count += 1
                    assigned_before[i] = 1
                if assigned_before[j] == 0:
                    if cluster_idx in assignment:
                        assignment[cluster_idx].append(j) 
                    else:
                        assignment[cluster_idx] = [j]
                    assigned_count += 1
                    assigned_before[j] = 1
        if assigned_count > 0:
            cluster_idx += 1

    for cluster_idx in assignment:
        print(" ".join([str(idx) for idx in assignment[cluster_idx]]))

from sklearn.cluster import SpectralClustering
from sklearn.cluster import SpectralCoclustering

def run_spectral_clustering(task_affinities, k, use_exp=True):
    if use_exp:
        task_affinities = np.exp(task_affinities)
    num_task = task_affinities.shape[0]
    sym_task_models = (task_affinities + task_affinities.T)/2

    clustering = SpectralClustering(
        n_clusters=k,
        affinity="precomputed", 
        n_init=100).fit(sym_task_models)

    groups = []
    for i in range(k):
        group = np.arange(num_task)[clustering.labels_ == i]
        groups.append(group)
        print(group)


def run_extended_spectral_clustering(task_affinities, k, use_exp=True):
    ''' 
    Extended Spectral Coclustering:
    [(\Theta + \Theta.T)/2, \Theta,
    \Theta.T, 0      ]
    '''
    num_task = task_affinities.shape[0]
    if use_exp:
        task_affinities = np.exp(task_affinities)
    A_1 = np.concatenate([(task_affinities+task_affinities.T)/2, task_affinities], axis=1)
    A_2 = np.concatenate([task_affinities.T, np.zeros_like(task_affinities)], axis=1)
    A = np.concatenate([A_1, A_2], axis=0)

    clustering = SpectralClustering(
        n_clusters=k,
        affinity="precomputed", 
        n_init=100).fit(A)

    groups = []
    for i in range(k):
        group = np.arange(num_task*2)[clustering.labels_ == i]
        new_group = set(); has_target = False
        for task_id in group:
            if task_id<num_task:
                new_group.add(task_id)
                has_target = True
            else:
                new_group.add(task_id - num_task)
        if has_target:
            groups.append(np.array(list(new_group)))

    for group in groups:
        print(" ".join([str(idx) for idx in group]))

def run_spectral_coclustering(task_affinities, k, use_exp=True):
    if use_exp:
        task_affinities = np.exp(task_affinities)
    num_task = task_affinities.shape[0]

    clf = SpectralCoclustering(n_clusters=k, random_state=0, n_init=100).fit(task_affinities)

    row_clusters = clf.row_labels_
    col_clusters = clf.column_labels_

    row_shuffle_idxes = np.concatenate([
        np.arange(num_task)[row_clusters == i] for i in range(k)
        ])
    col_shuffle_idxes = np.concatenate([
        np.arange(num_task)[col_clusters == i] for i in range(k)
        ])

    for i in range(k):
        group = set(list(np.arange(num_task)[row_clusters == i]))
        group.update(list(np.arange(num_task)[col_clusters == i]))
        print(np.array(list(group)))
    

# %%
task_affinities = np.load('task_affinities_pairwise_youtube_10.npy')
std = np.std(task_affinities, keepdims=True, axis=1)
std[std==0] = 1
normalized_task_affinities = (task_affinities - np.mean(task_affinities, keepdims=True, axis=1))/std
run_sdp_clustering(task_affinities, 20, use_exp=False)