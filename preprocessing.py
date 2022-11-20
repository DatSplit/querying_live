import pymeshlab
import os
from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd
import seaborn as sb
from collections import defaultdict
import utils
from multiprocessing import Process, Manager
import logging 

logging.basicConfig(level=logging.DEBUG,
                    format='(%(processName)s) %(message)s',)

def centering(ms):
    ms.compute_matrix_from_translation(traslmethod=3, neworigin=ms.get_geometric_measures()['barycenter'])
    return ms

def aligning(ms):
    # Aligning meshes
    ms.compute_matrix_by_principal_axis() # This prints the determinant to the console. Which is not a problem but looks ugly. 2nd edit: but only sometimes????
    ms.compute_matrix_from_rotation(rotaxis=1,angle=90)
    return ms

def scaling(ms):
    bbox = ms.get_geometric_measures()['bbox']
    max_side = np.max([bbox.dim_x(), bbox.dim_y(), bbox.dim_z()])
    ms.compute_matrix_from_scaling_or_normalization(axisx=1/max_side, axisy=1/max_side,axisz=1/max_side)
    return ms

def resampling(ms, upper_threshold=4_000, lower_threshold=2_000):
    """
    Resamples the meshes so that the amount of faces is within a range. Uses Subdivision midpoint and meshing decimation clustering.

        Parameters:
                ms (pymeshlab.MeshSet): A MeshSet with as current mesh the mesh which needs to be resampled.
                upper_threshold (int): The highest amount of faces the mesh is allowed to have if after resampling.
                lower_threshold (int): The lowest amount of faces the mesh is allowed to have if after resampling.

        Returns:
                ms (pymesh.MeshSet): A MeshSet with as current mesh the resampled mesh.
    """

    # Supersampling

    i = 0
    while (ms.current_mesh().face_number() < lower_threshold):
        if i > 30:
            raise ValueError
        try:
           ms.meshing_repair_non_manifold_vertices()
           ms.meshing_repair_non_manifold_edges()
        except:
           pass
        try:
            ms.meshing_surface_subdivision_midpoint(iterations=1)

            # Other Options:
            #ms.meshing_surface_subdivision_butterfly(iterations=1)
            #ms.meshing_surface_subdivision_catmull_clark()
            #ms.meshing_surface_subdivision_ls3_loop()
            #ms.meshing_surface_subdivision_midpoint()
            #ms.meshing_surface_subdivision_loop(iterations=1)
            
            pass
        except Exception as e:
            raise ValueError
        i += 1   
     

    #subsampling
    i = 1
    while (ms.current_mesh().face_number() > upper_threshold ):
        if i > 50:
           raise ValueError
        #ms.meshing_decimation_quadric_edge_collapse(targetfacenum=4_000)
        ms.meshing_decimation_clustering(threshold=pymeshlab.Percentage(float(i)*0.1))
        i += 1
    
    return ms

def preprocess(ms):
    """
    Preprocesses a mesh using resampling, centering, aligning, flipping and scaling.

        Parameters:
                ms (pymeshlab.MeshSet): A MeshSet with as current mesh the mesh which needs to be preprocessed.

        Returns:
                ms (pymesh.MeshSet): A MeshSet with as current mesh the preprocessed mesh.
    """

    # Preparing mesh for resampling
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_close_holes(maxholesize=30,newfaceselected=False )
    # ms.meshing_snap_mismatched_borders()
    # ms.meshing_re_orient_faces_coherentely()
    # ms.meshing_close_holes(maxholesize=30,selfintersection =False, newfaceselected=False)
    try:
        ms.meshing_repair_non_manifold_vertices()
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_close_holes(maxholesize=30,newfaceselected=False)
    except:
        pass
    try:
        ms.meshing_snap_mismatched_borders()
    except:
        pass
    try:
        ms.meshing_repair_non_manifold_vertices()
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_re_orient_faces_coherentely()
    except:
        pass
    # resampling meshes
    try:
        ms = resampling(ms)
    except ValueError:
        return None
    try:
        ms.meshing_repair_non_manifold_vertices()
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_close_holes(maxholesize=30,newfaceselected=False )
    except:
        pass
    try:
        ms.meshing_snap_mismatched_borders()
    except:
        pass
    try:
        ms.meshing_repair_non_manifold_vertices()
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_re_orient_faces_coherentely()
    except:
        pass

    ms = centering(ms)
    ms = aligning(ms)
    ms = flip_mesh(ms)
    ms = scaling(ms)
    
    return ms

def flip_mesh(ms):
    """
    Flips the mesh based on the moment test. (04 MR Features Slide 21)

        Parameters:
                ms (pymeshlab.MeshSet): A MeshSet with as current mesh the mesh which needs to be flipped.

        Returns:
                ms (pymesh.MeshSet): A MeshSet with as current mesh the preprocessed mesh.
    """
    fx = 0
    fy = 0
    fz = 0
    vertex_matrix = ms.current_mesh().vertex_matrix() 
    for f in ms.current_mesh().face_matrix():
        center_i = (vertex_matrix[f[0]]+vertex_matrix[f[1]]+vertex_matrix[f[2]]) / 3
        fx += np.sign(center_i[0])*np.square(center_i[0])
        fy += np.sign(center_i[1])*np.square(center_i[1])
        fz += np.sign(center_i[2])*np.square(center_i[2])
    if(np.sign(fx) < 0):
        ms.apply_matrix_flip_or_swap_axis(flipx  = True)
    if(np.sign(fz) < 0):
        ms.apply_matrix_flip_or_swap_axis(flipz  = True)
    if(np.sign(fy) < 0):
        ms.apply_matrix_flip_or_swap_axis(flipy  = True)
    return ms

def preprocess_all_data(dataset):
    errors = 0
    
    paths = utils.all_paths(dataset)
    print("Preprocessing..")
    for i, path in enumerate(paths):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(paths)}")
    
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(path)
        ms = preprocess(ms)
        if not ms:
            print("Error")
            errors += 1
            continue

        ms.save_current_mesh(path.replace("raw","preprocessed")) # This is a bug waiting to happen

def thread_job(low, high):
    errors = 0
    logging.info(f"{low}-{high}")
    paths = utils.all_paths("raw")
    for i, path in enumerate(paths[low:high]):
        if i % 50 == 0:
            logging.info(f"Progress: {i}/{len(paths[low:high])}")
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(path)
        ms = preprocess(ms)
        if not ms:
            errors += 1
            continue

        ms.save_current_mesh(path.replace("raw","preprocessed")) # This is a bug waiting to happen
    logging.info(f"Done! with {errors} errors")
    return 

def preprocess_multiprocess(n=None, n_processes=4):
    """
    Uses multi-processing to calculate the feature vectors of the meshes in the database.

        Parameters:
                n (int): The amount of meshes to calculate the feature vectors from
                n_processes (int): The amount of processes to start

        Returns:
                feature_matrix (pandas.DataFrame): Dataframe with meshes as rows and features as columns
    """
    if n == None:
        n = len(utils.all_paths("raw"))
    

    splits = np.arange(0, n + n/n_processes, n/n_processes)
    splits = [int(split) for split in splits]

    # Creating processes
    processes = []
    for i in range(n_processes):
        processes.append(Process(name=f"t{i}", target=thread_job,args=(splits[i],splits[i+1])))

    for process in processes:
        process.start()
    
    for process in processes:
        process.join()
    
    return


if __name__ == "__main__":
    preprocess_multiprocess()