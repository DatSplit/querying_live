from msilib.schema import MsiAssembly
from turtle import distance
import utils
import pymeshlab
import numpy as np
from stats import distance_from_origin, max_side_bounding_box
import random
from preprocessing import preprocess, centering, scaling

def test_centering(amount=885, threshold=0.1):
    ms = pymeshlab.MeshSet()
    paths = utils.all_paths("preprocessed")
    rand_idx = np.random.choice(len(paths),amount, replace=False)

    incorrect = 0 
    for idx in rand_idx:
        distance = distance_from_origin(paths[idx])
        try:
            assert distance < threshold, f"Incorrect centering for {paths[idx].split('/')[-1]}"
        except AssertionError as e:
            incorrect += 1
            print(e)
    print(f"{incorrect}/{amount} incorrect centering")

def test_scaling(amount=885, threshold=0.1):
    ms = pymeshlab.MeshSet()
    paths = utils.all_paths("preprocessed")
    rand_idx = np.random.choice(len(paths),amount, replace=False)

    incorrect = 0 
    for idx in rand_idx:
        max_bb = max_side_bounding_box(paths[idx])
        try:
            assert max_bb > 1 - threshold and max_bb < 1 + threshold ,f"{paths[idx]}"
        except AssertionError as e:
            incorrect += 1
            print(e)
    print(f"{incorrect}/{amount} incorrect scaling")
    
def test_topology(amount):
    ms = pymeshlab.MeshSet()
    paths = utils.all_paths("raw")
    rand_idx = np.random.choice(len(paths),amount, replace=False)

    incorrect = 0 
    for idx in rand_idx:
        ms.load_new_mesh(paths[idx])
        print(ms.get_topological_measures())
        ms.show_polyscope()
        #max_bb = max_side_bounding_box(paths[idx])
        #try:
        #    assert max_bb > 1 - threshold and max_bb < 1 + threshold ,f"{paths[idx]}"
        #except AssertionError as e:
        #    incorrect += 1
        #    print(e)
        ms = pymeshlab.MeshSet()
    #print(f"{incorrect}/{amount} incorrect scaling")

#def randomize()

if __name__ == "__main__":
    #test_centering()
    #test_scaling()
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh("meshes_raw/m41.obj")
    print("mesh loaded")
    ms.show_polyscope()
    
    for i in range(5):
        ms.clear()
        ms.load_new_mesh("meshes_raw/m41.obj")
        ms.compute_matrix_from_rotation(rotaxis="X axis",rotcenter="barycenter",angle=random.randint(0,360))
        ms.compute_matrix_from_rotation(rotaxis="Y axis",rotcenter="barycenter",angle=random.randint(0,360))
        ms.compute_matrix_from_rotation(rotaxis="Z axis",rotcenter="barycenter",angle=random.randint(0,360))
        ms.compute_matrix_from_scaling_or_normalization(scalecenter="barycenter",axisx=random.randint(0,10)/10 + 0.5)
        ms.compute_matrix_from_translation(axisx=random.randint(0,10)/1000,axisy=random.randint(0,10)/1000,axisz=random.randint(0,10)/1000)
        ms.show_polyscope()
        ms = preprocess(ms)
        ms.show_polyscope()

        

