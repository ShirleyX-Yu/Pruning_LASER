import torch
import numpy as np
import matplotlib.pyplot as plt
from laser.q_vendi import score, sequential_maximize_score, sequential_maximize_score_i
import pickle


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


# function to do low rank approx using VENDI scores
def do_vendi_approx(weight, k, debug=False): 
    assert weight.ndim == 2

    max_rank = min(weight.shape[0], weight.shape[1])
    desired_rank = int(max_rank * k) 

    if debug:
        print(f"Shape is {weight.shape} and shape is {weight.dtype} => desired rank {desired_rank}")

    # vendi score approx steps below
    # convert to a numpy array
    numpyWeight = weight.detach().numpy()

    def centered_gaussian(x):
        return np.exp(np.sum(-x**2, axis=-1))


    def rbf_k(x1, x2):
        return np.exp(np.sum(-(x1 - x2)**2) / 2)

    print("calculating selected_xs...")
    

    selected_xs_i = None
    filename = "processedIndexGPTJ.pickle"

    
    
    try:
        print("Looking for pre-processed pickle model...")
        # load from pickle file to get the top k vectors chosen
        with open(filename, 'rb') as handle:
            selected_xs_i = pickle.load(handle)
    except:
        print("Model not found... calculating vendi score...")
        # store in pickle file to not rerun the picking of the vectors
        selected_xs_i, qVS = sequential_maximize_score_i(numpyWeight, rbf_k, centered_gaussian, desired_rank)
        with open(filename, 'wb') as handle:
            pickle.dump(selected_xs_i, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    # change the indices into a tensor format for usage
    indicesTensor = torch.from_numpy(np.array(selected_xs_i, dtype=np.int32))
    B = torch.index_select(weight,0,indicesTensor)
    C = weight 

    print(f"B shape {B.shape}")
    print(f"C shape {C.shape}")

    print("solving closed form solution of A...")
    
    # closed form formula for A, such that A . B best approximates C 
    A = C @ B.T @ (torch.inverse(B @ B.T))

    print("solving for weight_approx...")
    weight_approx = A @ B 

    if debug:
        print(f"New matrix has shape {weight_approx.shape}")

    assert weight_approx.shape[0] == weight.shape[0] and weight_approx.shape[1] == weight.shape[1]
    # wraps the weight_approx tensor into a torch.nn.Parameter object
    # weight_approx = torch.nn.Parameter(weight_approx)

    return weight_approx