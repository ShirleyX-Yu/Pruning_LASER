import torch
import numpy as np
import matplotlib.pyplot as plt
from laser.q_vendi import score, sequential_maximize_score


# Helper functions for abs weight pruning
def sorted_mat(matrix):
    temp = list(abs(matrix).flatten())
    temp.sort()
    return temp


def prune(matrix, mat_sort, to_prune):
    if to_prune != 0:
        alpha = mat_sort[int(to_prune * 0.1 * len(mat_sort))]
        matrix[abs(matrix) <= alpha] = 0
    return matrix


def rank(matrix):
    np_matrix = np.array(matrix)
    return np.linalg.matrix_rank(np_matrix)/min(list(np_matrix.shape))


# What percentage can be pruned by weight
def sparsity(matrix, alpha):
    abs_matrix = abs(matrix)
    filtered_matrix = abs_matrix[abs_matrix < alpha]
    return len(filtered_matrix)/matrix.size


def viz_rank_change(rank_list,name):
    fig = plt.figure()
    plt.plot(rank_list)
    plt.savefig(name)


# Helper functions for rank reduction
# replace do_low_rank with VENDI score
# k referes to bottom k% of the singular vectors (pg 8 of paper)
def do_low_rank(weight, k, debug=False, niter=2):
    assert weight.ndim == 2

    max_rank = min(weight.shape[0], weight.shape[1])
    desired_rank = int(max_rank * k) 

    if debug:
        print(f"Shape is {weight.shape} and shape is {weight.dtype} => desired rank {desired_rank}")

    results = torch.svd_lowrank(weight,
                                q=desired_rank,
                                niter=niter)

    # weight_approx is a pytorch tensor with the same shape as weight 
    weight_approx = results[0] @ torch.diag(results[1]) @ results[2].T

    if debug:
        print(f"New matrix has shape {weight_approx.shape}")

    assert weight_approx.shape[0] == weight.shape[0] and weight_approx.shape[1] == weight.shape[1]

    # wraps the weight_approx tensor into a torch.nn.Parameter object
    weight_approx = torch.nn.Parameter(weight_approx)

    return weight_approx

'''
# function to do low rank approx using VENDI scores
def do_vendi_approx(weight, k, debug=False): 
    assert weight.ndim == 2

    max_rank = min(weight.shape[0], weight.shape[1])
    desired_rank = int(max_rank * k) 

    if debug:
        print(f"Shape is {weight.shape} and shape is {weight.dtype} => desired rank {desired_rank}")

    # vendi score approx steps below
    # convert to a numpy array
    weight0 = weight
    weight1 = weight.detach().numpy()

    def centered_gaussian(x):
        return np.exp(np.sum(-x**2, axis=-1))


    def rbf_k(x1, x2):
        return np.exp(np.sum(-(x1 - x2)**2) / 2)

    print("calculating selected_xs...")
    selected_xs, qVS = sequential_maximize_score(weight1, rbf_k, centered_gaussian, desired_rank)

    B = selected_xs
    C = weight1 

    print(f"B shape {B.shape}")
    print(f"C shape {C.shape}")

    print("solving closed form solution of A...")
    # closed form formula for A, such that A . B best approximates C 
    A = C @ B.T @ (np.linalg.inv(B @ B.T))

    print("solving for weight_approx...")
    weight_approx = A @ B 

    if debug:
        print(f"New matrix has shape {weight_approx.shape}")

    assert weight_approx.shape[0] == weight.shape[0] and weight_approx.shape[1] == weight.shape[1]
    # wraps the weight_approx tensor into a torch.nn.Parameter object
    weight_approx = torch.nn.Parameter(weight_approx)

    return weight_approx
'''













