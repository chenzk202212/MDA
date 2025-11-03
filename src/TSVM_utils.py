from hashlib import new
import torch
import math
import numpy as np
import torch.nn.functional as F
from src.task_vectors import *

def compute_svd_dict(task_vectors, config):
    """
    Computes the Singular Value Decomposition (SVD) for each task vector in the provided datasets and stores the results in a dictionary.

    Args:
        task_vectors (list): A list of task vector objects, where each task vector contains a dictionary of matrices to be decomposed.
        config (object): Configuration object containing the list of datasets under the attribute `DATASETS`.

    Returns:
        dict: A dictionary where each key is a dataset name and the value is another dictionary containing the SVD components ('u', 's', 'v') for each matrix in the task vector.
        If a matrix is not 2-dimensional or contains 'text_projection' in its key, it is stored under the key 'dim1' without decomposition.
    """
    sv_reduction = 1 / len(config.DATASETS)
    with torch.no_grad():
        svd_dict = {}
        for i, (task_vector, dataset) in enumerate(zip(task_vectors, config.DATASETS)):
            svd_dict[dataset] = {}
            print(f"Computing SVD for {dataset}...")
            for key in task_vector.vector:
                svd_dict[dataset][key] = {}
                if (
                    len(task_vector.vector[key].shape) == 2
                    and "text_projection" not in key
                ):
                    u, s, v = torch.linalg.svd(
                        task_vector.vector[key], full_matrices=False
                    )
                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    temp_u = torch.zeros_like(u)
                    # select only the first reduced_index_s columns of u and place them
                    temp_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    svd_dict[dataset][key]["u"] = temp_u

                    temp_s = torch.zeros_like(s)
                    temp_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]

                    svd_dict[dataset][key]["s"] = temp_s  # s_reduced

                    # select only the first reduced_index_s rows of v and place them
                    temp_v = torch.zeros_like(v)
                    temp_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                    svd_dict[dataset][key]["v"] = temp_v

                    # temp_mat = temp_u @ torch.diag_embed(temp_s) @ temp_v
                else:
                    svd_dict[dataset][key]["dim1"] = task_vector.vector[key]
    return svd_dict


def sum_svd_dict(svd_dict, config):
    """
    Sums the Singular Value Decomposition (SVD) components from multiple datasets and computes a new vector.

    Args:
        svd_dict (dict): A dictionary containing SVD components for multiple datasets. The structure of the dictionary is expected to be:
                         {
                             dataset_name: {
                                 key: {
                                     "u": tensor,
                                     "s": tensor,
                                     "v": tensor,
                                     "dim1": tensor (optional)
                                 }
                             }
                         }
        config (object): A configuration object that contains a list of dataset names under the attribute `DATASETS`.

    Returns:
        dict: A dictionary containing the merged SVD components or averaged "dim1" values for each key.
    """
    print("Summing SVD...")
    new_vector = {}
    for key in svd_dict[config.DATASETS[0]]:
        if "u" in svd_dict[config.DATASETS[0]][key].keys():
            sum_u = sum([svd_dict[dataset][key]["u"] for dataset in config.DATASETS])
            sum_s = sum([svd_dict[dataset][key]["s"] for dataset in config.DATASETS])
            sum_v = sum([svd_dict[dataset][key]["v"] for dataset in config.DATASETS])

            u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
            u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
            new_vector[key] = torch.linalg.multi_dot(
                (
                    u_u,
                    v_u,
                    torch.diag(sum_s),
                    u_v,
                    v_v,
                )
            )
        else:
            for i, dataset in enumerate(config.DATASETS, start=1):
                if i == 1:
                    new_vector[key] = svd_dict[dataset][key]["dim1"]
                else:
                    new_vector[key] += (
                        svd_dict[dataset][key]["dim1"] - new_vector[key]
                    ) / i
    return new_vector


###############
##### LOSSLESS Orthogonalization
def compute_and_sum_svd_mem_reduction_lossless(task_vectors, config):
    """
    Computes the Singular Value Decomposition (SVD) for each task vector and merge the results.

    This function performs the following steps:
    1. Iterates over each layer in the task vectors.
    2. For each layer, it computes the SVD of the corresponding matrix if it is a 2D tensor excluding "text_projection".
    3. Concatenate the U_i, S_i, and V_i matrices from the SVD across all tasks.
    4. If the vector is not a 2D tensor or is "text_projection", it computes the mean of the vectors.
    5. After concatenating the SVD components, recomputes the SVD of the summed U and V matrices and constructs the merged layer.

    Args:
        task_vectors (list): A list of task vectors, where each task vector is a dictionary containing the vectors for each task.
        config (object): A configuration object containing the device and dataset information.

    Returns:
        dict: A dictionary containing the new vectors after summing the SVD components.
    """
    # becareful wit vit-l on 20 task it does not fit in GPU or in 64 GB RAM (try without last layer)
    device = config.device
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            for i, (task_vector, dataset) in enumerate(
                zip(task_vectors, config.DATASETS)
            ):
                vec = task_vector.vector[key].to(device)

                if (
                    len(task_vector.vector[key].shape) == 2
                    and "text_projection" not in key
                ):

                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros(
                            u.shape[0], u.shape[1] * config.num_tasks, device=device
                        )
                        sum_s = torch.zeros(
                            s.shape[0] * config.num_tasks, device=device
                        )
                        sum_v = torch.zeros(
                            v.shape[0] * config.num_tasks, v.shape[1], device=device
                        )
                    reduced_index_s = s.shape[0]

                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

                new_vector[key] = torch.linalg.multi_dot(
                    (
                        u_u,
                        v_u,
                        torch.diag(sum_s),
                        u_v,
                        v_v,
                    )
                )

    return new_vector


###############
##### LOSSLESS EIGENDECOMP
def compute_and_sum_svd_mem_reduction_lossless_eigen(task_vectors, config):
    """
    Computes the Singular Value Decomposition (SVD) for each task vector and merge the results.

    This function performs the following steps:
    1. Iterates over each layer in the task vectors.
    2. For each layer, it computes the SVD of the corresponding matrix if it is a 2D tensor excluding "text_projection".
    3. Concatenate the U_i, S_i, and V_i matrices from the SVD across all tasks.
    4. If the vector is not a 2D tensor or is "text_projection", it computes the mean of the vectors.
    5. After concatenating the SVD components, recomputes the eigendecomposition of the summed U and V matrices and constructs the merged layer.

    Args:
        task_vectors (list): A list of task vectors, where each task vector is a dictionary containing the vectors for each task.
        config (object): A configuration object containing the device and dataset information.

    Returns:
        dict: A dictionary containing the new vectors after merging the SVD components.
    """
    # becareful wit vit-l on 20 task it does not fit in GPU or in 64 GB RAM (try without last layer)
    device = config.device
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            for i, (task_vector, dataset) in enumerate(
                zip(task_vectors, config.DATASETS)
            ):
                vec = task_vector.vector[key].to(device)

                if (
                    len(task_vector.vector[key].shape) == 2
                    and "text_projection" not in key
                ):

                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros(
                            u.shape[0], u.shape[1] * config.num_tasks, device=device
                        )
                        sum_s = torch.zeros(
                            s.shape[0] * config.num_tasks, device=device
                        )
                        sum_v = torch.zeros(
                            v.shape[0] * config.num_tasks, v.shape[1], device=device
                        )
                    reduced_index_s = s.shape[0]

                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                sum_s, indices = torch.sort(sum_s, stable=True)

                sum_u = torch.index_select(sum_u, 1, indices)
                l_u, q_u = torch.linalg.eigh(sum_u.mT @ sum_u)
                u_orth = (
                    q_u
                    @ torch.diag(1.0 / (torch.sqrt(torch.abs(l_u)) + 1e-12))
                    @ q_u.mT
                )

                sum_v = torch.index_select(sum_v, 0, indices)

                l_v, q_v = torch.linalg.eigh(sum_v @ sum_v.mT)
                v_orth = (
                    q_v
                    @ torch.diag(1.0 / (torch.sqrt(torch.abs(l_v)) + 1e-12))
                    @ q_v.mT
                )

                new_vector[key] = torch.linalg.multi_dot(  # bool_mask *
                    (
                        u_orth,
                        torch.diag(sum_s),
                        v_orth,
                    )
                )

    return new_vector

def build_etf_svd(d: int, c: int) -> torch.Tensor:
    I = torch.eye(c)
    J = torch.ones((c, c)) / c
    P = I - J
    U, _, _ = torch.linalg.svd(P)
    etf = U[:, :d].T * np.sqrt(c / (c - 1))
    return etf.T  # shape: [C, D]

def build_etf_qr(d: int, c: int) -> torch.Tensor:
    X = torch.randn(d, c)
    Q, _ = torch.linalg.qr(X)  # Q shape: [d, c]
    etf = Q.T  # [C, D]
    return F.normalize(etf, dim=1)

import torch
import torch.nn as nn
import torch.nn.functional as F

rotation_modules = {} 

class LearnableSO(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.R = nn.Parameter(torch.eye(dim))

    def forward(self):
        u, _, v = torch.linalg.svd(self.R)
        return u @ v.T 

def sanitize_key(key: str) -> str:
    return key.replace(".", "__")  # or use "_" or another safe token




import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull


def matrix_cosine(a, b):
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    return F.cosine_similarity(a_flat, b_flat, dim=0).item()


# def compute_and_sum_svd_mem_reduction(task_vectors, config, save_path="task_vector_partition.png"):

#     if isinstance(task_vectors, dict):
#         task_vectors = list(task_vectors.values())
#     if hasattr(task_vectors[0], 'vector'):
#         task_vectors = [tv.vector for tv in task_vectors]

#     device = config.device
#     sv_reduction = 1/8
#     num_classes = 2400
#     alpha = 1.0

#     new_vector = {}
#     diff_before_all = []
#     diff_after_all = []

#     for key in task_vectors[0]:
#         original_shape = task_vectors[0][key].shape

#         if len(original_shape) == 2 and "text_projection" not in key:
#             output_dim = original_shape[0]
#             feature_dim = original_shape[1]
#             reduced_len = int(min(original_shape) * sv_reduction)

#             sum_u = torch.zeros((output_dim, reduced_len * len(task_vectors)), device=device)
#             sum_s = torch.zeros((reduced_len * len(task_vectors)), device=device)
#             sum_v = torch.zeros((reduced_len * len(task_vectors), feature_dim), device=device)

#             for i, task_vector in enumerate(task_vectors):
#                 vec = task_vector[key].to(device)
#                 u, s, v = torch.linalg.svd(vec, full_matrices=False)
#                 start = i * reduced_len
#                 end = (i + 1) * reduced_len
#                 sum_u[:, start:end] = u[:, :reduced_len]
#                 sum_s[start:end] = s[:reduced_len]
#                 sum_v[start:end, :] = v[:reduced_len, :]

#             u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
#             u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
#             reconstructed = torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v))

#             if feature_dim >= num_classes:
#                 etf = build_etf_svd(d=num_classes, c=feature_dim).to(device)
#                 etf = etf.T
#             else:
#                 etf = build_etf_svd(d=feature_dim, c=num_classes).to(device)
#             etf = F.normalize(etf, dim=0)

#             rotated = reconstructed
#             etf_proj = rotated @ etf.T @ etf
#             # print('rec', rotated.shape)
#             # print('etf', etf_proj.shape)
#             # print('eee', etf.shape)
#             # final =  etf_proj
#             # final1 =  rotated
#             # diff_before_all.append(matrix_cosine(final1, etf_proj))
#             # diff_after_all.append(matrix_cosine(final, etf_proj))

#             new_vector[key] = final[:, :feature_dim] if final.shape[1] > feature_dim else final
#         else:
#             new_vector[key] = torch.mean(torch.stack([v[key].to(device) for v in task_vectors]), dim=0)

#     diff_before = sum(diff_before_all) / len(diff_before_all) if diff_before_all else 0.0
#     diff_after = sum(diff_after_all) / len(diff_after_all) if diff_after_all else 0.0

#     print('diff_before', diff_before)
#     print('diff_after', diff_after)

#     return new_vector


def compute_mda(task_vectors, config):

    if isinstance(task_vectors, dict):
        task_vectors = list(task_vectors.values())

    if hasattr(task_vectors[0], 'vector'):
        task_vectors = [tv.vector for tv in task_vectors]


    device = config.device
    sv_reduction = 1 / 8
    num_classes = 2400
    alpha = 1.0

    new_vector = {}

    for key in task_vectors[0]:
        original_shape = task_vectors[0][key].shape

        if len(original_shape) == 2 and "text_projection" not in key:
            output_dim = original_shape[0]
            feature_dim = original_shape[1]
            print('feature_dim', feature_dim)
            reduced_len = int(min(original_shape) * sv_reduction)

            sum_u = torch.zeros((output_dim, reduced_len * len(task_vectors)), device=device)
            sum_s = torch.zeros((reduced_len * len(task_vectors)), device=device)
            sum_v = torch.zeros((reduced_len * len(task_vectors), feature_dim), device=device)

            for i, task_vector in enumerate(task_vectors):
                vec = task_vector[key].to(device)
                u, s, v = torch.linalg.svd(vec, full_matrices=False)
                start = i * reduced_len
                end = (i + 1) * reduced_len
                sum_u[:, start:end] = u[:, :reduced_len]
                sum_s[start:end] = s[:reduced_len]
                sum_v[start:end, :] = v[:reduced_len, :]

            u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
            u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
            reconstructed = torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v))  # 重建
            if feature_dim >= num_classes:
                etf = build_etf_qr(d=feature_dim, c=num_classes).to(device)  # [758, feature_dim]
                # etf = build_etf_svd(d=num_classes, c=feature_dim).to(device) 
                etf = etf.T
            else:
                etf = build_etf_svd(d=feature_dim, c=num_classes).to(device)  # [758, feature_dim]
            etf = F.normalize(etf, dim=0)


            rotated = reconstructed 

            etf_proj = rotated @ etf.T @ etf
            final = (1 - alpha) * rotated + alpha * etf_proj

            new_vector[key] = final[:, :feature_dim] if final.shape[1] > feature_dim else final

        else:
            new_vector[key] = torch.mean(torch.stack([
                v[key].to(device) for v in task_vectors
            ]), dim=0)

    return new_vector


###############
#### TSV Merge Eigendecomp
def compute_and_sum_svd_mem_reduction_2(task_vectors, config):
    """
    Computes the Singular Value Decomposition (SVD) for each vector in the task_vectors,
    reduces the dimensionality of the vectors based on the sv_reduction factor, and concatenate
    the low-rank matrices. If the vector is not a 2D tensor or is "text_projection", it computes the mean of the vectors.
    Computation of the eigendecomposition is performed instead of the SVD for the second operation.

    Args:
        task_vectors (list): A list of task vector objects, where each object contains a
                             dictionary of vectors.
        config (object): Configuration object containing the following attributes:
                         - DATASETS (list): List of datasets.
                         - device (torch.device): The device to perform computations on.

    Returns:
        dict: A dictionary containing the new vectors after SVD computation and merging.
    """
    sv_reduction = 1 / len(config.DATASETS)
    device = config.device
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            for i, (task_vector, dataset) in enumerate(
                zip(task_vectors, config.DATASETS)
            ):
                vec = task_vector.vector[key].to(device)

                if (
                    len(task_vector.vector[key].shape) == 2
                    and "text_projection" not in key
                ):
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                sum_s, indices = torch.sort(sum_s, stable=True)

                sum_u = torch.index_select(sum_u, 1, indices)
                l_u, q_u = torch.linalg.eigh(sum_u.mT @ sum_u)
                u_orth = (
                    q_u
                    @ torch.diag(1.0 / (torch.sqrt(torch.abs(l_u)) + 1e-12))
                    @ q_u.mT
                )

                sum_v = torch.index_select(sum_v, 0, indices)

                l_v, q_v = torch.linalg.eigh(sum_v @ sum_v.mT)
                v_orth = (
                    q_v
                    @ torch.diag(1.0 / (torch.sqrt(torch.abs(l_v)) + 1e-12))
                    @ q_v.mT
                )

                new_vector[key] = torch.linalg.multi_dot(  # bool_mask *
                    (
                        sum_u,
                        u_orth,
                        torch.diag(sum_s),
                        v_orth,
                        sum_v,
                    )
                )

    return new_vector


###############
#### Rank Reduction TV
def compute_and_sum_svd_mem_reduction_rank_reduction(task_vectors, config):
    """
    Compute and sum the Singular Value Decomposition (SVD) of task vectors with rank reduction.

    This function performs SVD on the vectors in `task_vectors` and reduces their rank based on the
    number of tasks specified in `config.DATASETS`. The reduced vectors are then summed together.

    Args:
        task_vectors (list): A list of task vector objects. Each object should have a `vector` attribute
                             which is a dictionary where keys are vector names and values are tensors.
        config (object): Configuration object containing the following attributes:
                         - DATASETS (list): List of datasets.
                         - device (torch.device): The device to perform computations on.

    Returns:
        dict: A dictionary containing the new vectors after SVD computation and summation.
    """
    sv_reduction = 1 / len(config.DATASETS)
    device = config.device
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            for i, (task_vector, dataset) in enumerate(
                zip(task_vectors, config.DATASETS)
            ):
                vec = task_vector.vector[key].to(device)

                if (
                    len(task_vector.vector[key].shape) == 2
                    and "text_projection" not in key
                ):
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                new_vector[key] = torch.linalg.multi_dot(
                    (
                        sum_u,
                        torch.diag(sum_s),
                        sum_v,
                    )
                )
    return new_vector


def compute_and_sum_svd_mem_reduction_dummy(task_vectors, config):
    """To perform dummy operations."""
    sv_reduction = 1 / 8
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            for i in range(0, 8):
                if (
                    len(task_vectors[0].vector[key].shape) == 2
                    and "text_projection" not in key
                ):
                    if i == 0:
                        u, s, v = torch.linalg.svd(
                            task_vectors[0].vector[key], full_matrices=False
                        )
                        reduced_index_s = int(s.shape[0] * sv_reduction)

                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros_like(u)
                        sum_s = torch.zeros_like(s)
                        sum_v = torch.zeros_like(v)

                        # select only the first reduced_index_s columns of u and place them
                        sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                            :, :reduced_index_s
                        ]
                        sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                            :reduced_index_s
                        ]
                        # select only the first reduced_index_s rows of v and place them
                        sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                            :reduced_index_s, :
                        ]
                    else:
                        # generate u vectors orthogonal to the previous ones
                        # generate v vectors orthogonal to the previous ones
                        print("dummy")
                        u = torch.nn.functional.normalize(
                            torch.randn_like(sum_u), p=2, dim=-2
                        )
                        v = torch.nn.functional.normalize(
                            torch.randn_like(sum_v), p=2, dim=-1
                        )

                        # select only the first reduced_index_s columns of u and place them
                        sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                            :, :reduced_index_s
                        ]
                        sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                            :reduced_index_s
                        ]
                        # select only the first reduced_index_s rows of v and place them
                        sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                            :reduced_index_s, :
                        ]

                else:
                    if i == 0:
                        new_vector[key] = task_vectors[0].vector[key]
                    # else:
                    #     new_vector[key] += (
                    #         task_vector.vector[key] - new_vector[key]
                    #     ) / (i + 1)

            if (
                len(task_vectors[0].vector[key].shape) == 2
                and "text_projection" not in key
            ):

                new_vector[key] = torch.linalg.multi_dot(
                    (
                        sum_u,
                        torch.diag(sum_s),
                        sum_v,
                    )
                )

    return new_vector
