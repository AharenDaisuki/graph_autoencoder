import pickle
import shutil

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