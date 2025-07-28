import pickle
import shutil
import torch

import numpy as np

from torch import Tensor

def load_from_pkl(pkl_path):
    """
    Load data from a pickle file. 

    Args: 
        pkl_path (str): pickle file path.

    Returns: 
        object: data
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    f.close()
    return data

def save_as_pkl(data, pkl_path): 
    """
    Save data as a pickle file. 

    Args: 
        data (object): object to be saved.
        pkl_path (str): pickle file save path.
    """
    with open(pkl_path, 'wb') as f: 
        pickle.dump(data, f)
    f.close()

def clear_tmp_dir(tmp_dir): 
    """
    Remove a temporary directory.

    Args: 
        tmp_dir (str): temporary directory path.
    """
    try: 
        shutil.rmtree(tmp_dir)
    except FileNotFoundError:
        assert False, f'Directory [{tmp_dir}] is not found!'

def tensor_to_ndarray(t: Tensor) -> np.ndarray:
    """
    Convert a tensor on GPU to a numpy array on CPU.

    Args: 
        t (Tensor): tensor on GPU.

    Returns: 
        ndarray: a numpay array
    """
    return t.detach().cpu().numpy()

def get_sparse_matrix_alpha(indices: torch.Tensor, values: torch.Tensor, size: tuple, data_type, device):
    """
    Get a sparse matrix (COO) given indices, values and matrix size.

    Args:
        indices (Tensor): Initial data for the tensor. Can be a Tensor. The indices are the coordinates of the non-zero values in the matrix, and thus
            should be two-dimensional where the first dimension is the number of tensor dimensions and
            the second dimension is the number of non-zero values.
        values (Tensor): Initial values for the tensor. Can only be a tuple.
        size (tuple): Size of the sparse tensor.
        dtype (:class:`torch.dtype`): the desired data type of returned tensor.
        device (:class:`torch.device`): the desired device of returned tensor.

    Returns: 
        Tensor: a sparse tensor of predictions.
    """
    assert indices.shape[-1] == values.shape[0], f'(2, E) = {indices.shape} | (E, ) = {values.shape}'
    return torch.sparse_coo_tensor(indices=indices, values=values, size=size, dtype=data_type, device=device).to_sparse_csr()
