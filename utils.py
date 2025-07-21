import pickle
import shutil
import torch

import numpy as np

from torch import Tensor

def load_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    f.close()
    return data

def save_as_pkl(data, pkl_path): 
    with open(pkl_path, 'wb') as f: 
        pickle.dump(data, f)
    f.close()

def clear_tmp_dir(tmp_dir): 
    try: 
        shutil.rmtree(tmp_dir)
    except FileNotFoundError:
        assert False, f'Directory [{tmp_dir}] is not found!'

def tensor_to_ndarray(t: Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()

def get_sparse_matrix_alpha(indices: torch.Tensor, values: torch.Tensor, size: tuple, data_type, device):
    assert indices.shape[-1] == values.shape[0], f'(2, E) = {indices.shape} | (E, ) = {values.shape}'
    return torch.sparse_coo_tensor(indices=indices, values=values, size=size, dtype=data_type, device=device).to_sparse_csr()
