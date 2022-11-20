import pymeshlab
import utils
import pandas as pd
import numpy as np
import math


def distance_from_origin(path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    barycenter = ms.get_geometric_measures()["barycenter"]
    dist = np.linalg.norm(barycenter - [0,0,0])
    return dist

def max_side_bounding_box(path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    bb = ms.current_mesh().bounding_box()
    return np.max([bb.dim_x(), bb.dim_y(), bb.dim_z()])

def diff_from_ideal_faces(path, ideal=4_000):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    return np.abs(ms.current_mesh().face_number() - ideal)


def calculate_statistics(dataset):
    ms = pymeshlab.MeshSet()
    paths = utils.all_paths(dataset)

    ids, face_numbers, vertex_numbers, classes, distance_to_origin, filepaths, angle, flip = [],[],[],[],[], [],[], []
    bounding_boxes = {}
    for key in ["dim_x", "dim_y", "dim_z"]:
        bounding_boxes[key] = []

    print(f"Calculating statistics on {dataset}")

    for i, path in enumerate(paths):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(paths)}")

        ms.load_new_mesh(path)
        id = path.split("/")[-1].split(".")[0]
        ids.append(id)
        classes.append(path.split('/')[-2])
        face_numbers.append(ms.current_mesh().face_number())
        vertex_numbers.append(ms.current_mesh().vertex_number())

        # refactor 
        d = (abs(ms.get_geometric_measures()['barycenter'][0])+abs(ms.get_geometric_measures()['barycenter'][1])+abs(ms.get_geometric_measures()['barycenter'][2]))
        vertex_matrix = ms.current_mesh().vertex_matrix().T
        covariance_matrix = np.cov(vertex_matrix)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        angle.append(absolute_angle(eigenvectors[np.argmax(eigenvalues)],[1,0,0]))


        distance_to_origin.append(d)
        flip.append(flip_mesh(ms))
        bounding_box = ms.current_mesh().bounding_box()
        encoded = utils.encode_bounding_box(bounding_box)

        for key in ['dim_x',"dim_y", "dim_z"]:
            bounding_boxes[key].append(encoded[key])



        filepaths.append(path)
    print(f"Progress: {len(paths)}/{len(paths)}")
    data = {
            "class": classes, 
            "n_faces": face_numbers, 
            "n_vertexes": vertex_numbers,
            "distance_origin": distance_to_origin,
            "path": filepaths,
            "ID" : ids,
            "absolute_angle": angle,
            "flipped" : flip
            }
    for key in ['dim_x',"dim_y", "dim_z"]:
        data[key] = bounding_boxes[key]
    df = pd.DataFrame(data=data)

    df.to_csv(f"tabular_data/statistics_{dataset}")
    return df

def diff_from_ideal_vertices(path, ideal=5_000):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    return np.abs(ms.current_mesh().vertex_number()-ideal)

def absolute_angle(a, b):
    
    
    return np.degrees(np.arccos(np.dot(abs(a), b) / (np.linalg.norm(
        abs((a))) * np.linalg.norm(b))))
    
    
def flip_mesh(ms):
    fx = 0
    fy = 0
    fz = 0
    vertex_matrix = ms.current_mesh().vertex_matrix()
    for f in ms.current_mesh().face_matrix():
        
        center_i =(vertex_matrix[f[0]]+vertex_matrix[f[1]]+vertex_matrix[f[2]]) /3
        
        fx += np.sign(center_i[0])*np.square(center_i[0])
        fy += np.sign(center_i[1])*np.square(center_i[1])
        fz += np.sign(center_i[2])*np.square(center_i[2])
    
    fx += 0.000000001   #This is to account for rounding errors. sometimes fx = -1e-24
    fy += 0.000000001
    fz += 0.000000001
    return([np.sign(fx),np.sign(fy),np.sign(fz)])




    
if __name__ == "__main__":
    calculate_statistics("preprocessed")
    #calculate_statistics("raw")