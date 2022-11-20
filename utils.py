import pymeshlab
import os
import pandas as pd
dataset_to_path  = {
    "raw": "meshes_raw",
    "preprocessed": "meshes_preprocessed",
    "featureVectors": "feature_vectors/PRINCETON/train"
}

def encode_bounding_box(bb):
    return {
        "diagonal" :bb.diagonal(),
        "dim_x" :  bb.dim_x(),
        "dim_y" : bb.dim_y(),
        "dim_z" : bb.dim_z(),
        "max" : bb.max(),
        "min" : bb.min()
    }

def all_paths(dataset):
    cwd = os.getcwdb()
    path = dataset_to_path[dataset]
    result = []
    files = os.listdir(path)
    for file in files:
        if file.endswith(".off") or file.endswith("obj"):
            result.append(cwd.decode('UTF-8').replace('\\', '/') + "/" + path + "/" + file)
    return result

def get_id_to_idx(matrix_name):
    df = pd.read_csv(f"feature_matrices/{matrix_name}.csv", index_col=0) 
    id_to_idx = {}
    idx_to_id = {}
    for i , id in enumerate(df['ID']):
        id_to_idx[id] = i
        idx_to_id[i] = id
    return id_to_idx, idx_to_id

    

