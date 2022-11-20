import pymeshlab
import pandas as pd
import logging
import utils
import json
import numpy as np
from multiprocessing import Process, Manager
from multiprocessing import Pool
from numpy import arange as nparange
from features import volume, area, rectangularity, diameter, eccentricity, a3, d1, d2, d3, d4, compactness, diameter_2, convexity
import pickle
import time


logging.basicConfig(level=logging.DEBUG,
                    format='(%(processName)s) %(message)s',)

with open(r"C:\Users\niels\MR\INFOMR-Multimedia-Retrieval\labels\id_to_label_49.json", "rb") as file:
    ID_TO_LABEL = json.load(file)


def extract_features(path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)

    mesh_volume = volume(ms)

    vector = {
        "mesh_area": area(ms),
        "mesh_rectangularity": rectangularity(ms,mesh_volume),
        "mesh_eccentricity": eccentricity(ms),
        "mesh_volume": volume(ms),
        "mesh_a3": a3(ms, 50_000),
        "mesh_d1": d1(ms, 50_000),
        "mesh_d2": d2(ms, 50_000),
        "mesh_d3": d3(ms, 50_000),
        "mesh_d4": d4(ms, 50_000),
        }

    mesh_compactness = compactness(vector["mesh_area"], mesh_volume)
    vector["mesh_compactness"] = mesh_compactness
    vector["mesh_diameter"] =  diameter_2(ms)
    ms.clear()
    ms.load_new_mesh(path)
    vector["mesh_convexity"] = convexity(ms, mesh_volume)
    ms.clear()
    ms.load_new_mesh(path)
    id = path.split("/")[-1].split(".")[0][1:]
    vector['ID'] = id
    vector["class"] = ID_TO_LABEL[id]
    return vector

def feature_vectors_all_data(n=999999):

    feature_vector_dataframe = pd.DataFrame(columns = ["mesh_class", "mesh_area", "mesh_compactness", "mesh_rectangularity", "mesh_diameter", "mesh_eccentricity","mesh_volume", "mesh_convexity","mesh_a3", "mesh_d1", "mesh_d2",
                                                       "mesh_d3", "mesh_d4"])

    paths = utils.all_paths("preprocessed")
    print(f"Calculating feature vectors...")
    for i, path in enumerate(paths):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(paths)}")
        if i > n:
            return
        print(path)
        feature_vector = extract_features(path)

        feature_vector_dataframe = feature_vector_dataframe.append(feature_vector, ignore_index=True)
    feature_vector_dataframe.to_excel("feature_vectors.xlsx")


def standardize_features(df):
    """
    Standardizes the features in the feature matrix by subtracting the mean and dividing by the standard deviation. ()

        Parameters:
                df (pandas.DataFrame): the feature matrix

        Returns:
                df (pandas.DataFrame): the feature matrix with standardized values.
    """
    # This might be faster if we used a numpy array.
    standardizer = []
    for feature in ['mesh_area', "mesh_compactness", "mesh_rectangularity","mesh_diameter","mesh_convexity", "mesh_eccentricity", "mesh_volume"]:
        standardizer.append((df[feature].mean(), df[feature].std()))
        df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()
        
    return df, standardizer



def thread_job(low, high, return_dict):
    logging.info("Started thread!")
    paths = utils.all_paths("preprocessed")
    i = 0
    starttime = time.time()
    for path in paths[low: high]:
        if i % 10 == 0 and i > 0:
            current_time = time.time()
            time_per_mesh = (current_time - starttime)/i
            logging.info(f"""done {i} meshes with average {time_per_mesh} seconds per mesh \n
                             estimated total time = {time_per_mesh * (high - low)/60.0} minutes \n
                             estimated time left = {time_per_mesh * (high - low - i)/60.0} minutes \n""")
            
        feature_vector = extract_features(path)
        return_dict.append(feature_vector)
        
        logging.info(f"{i} / {len(paths[low:high])}")
        i += 1


def calculate_features_multiprocess(n=None, n_processes=4):
    """
    Uses multi-processing to calculate the feature vectors of the meshes in the database.

        Parameters:
                n (int): The amount of meshes to calculate the feature vectors from
                n_processes (int): The amount of processes to start

        Returns:
                feature_matrix (pandas.DataFrame): Dataframe with meshes as rows and features as columns
    """
    if n == None:
        n = len(utils.all_paths("preprocessed"))
    
    # Splitting the database in n equal parts
    splits = nparange(0, n + n/n_processes, n/n_processes)
    splits = [int(split) for split in splits]

    # Global data source
    manager = Manager()
    return_dict = manager.list()

    # Creating processes
    processes = []
    for i in range(n_processes):
        processes.append(Process(name=f"t{i}", target=thread_job,args=(splits[i],splits[i+1],return_dict,)))

    for process in processes:
        process.start()
    
    for process in processes:
        process.join()
 
    assert len(return_dict) == n

    feature_vector_dataframe = pd.DataFrame(columns = ["mesh_class","mesh_area", "mesh_compactness", "mesh_rectangularity","mesh_diameter","mesh_eccentricity", "mesh_a3", "mesh_d1", "mesh_d2", 
                                                       "mesh_d3", "mesh_d4"])

    feature_vector_dataframe = pd.DataFrame.from_records(return_dict)

    #feature_vector_dataframe = standardize_features(feature_vector_dataframe)
    

    return feature_vector_dataframe 

def test_diameter():
    paths = utils.all_paths("preprocessed")
    print(f"Calculating feature vectors...")
    for i, path in enumerate(paths):
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(path)

        dia1 = diameter(ms)
        dia2 = diameter_2(ms)
        if dia1 != dia2:
            print("YO")
        else:
            print("correct")
        if i > 20:
            return


if __name__ == "__main__":
    # import cProfile
    # import pstats
    # cProfile.run('test_diameter()', "profiling")
    # p = pstats.Stats("profiling")
    # p.strip_dirs().print_callees("test_diameter")
    
    
    df = calculate_features_multiprocess()
    df.to_csv("feature_matrices/14-11-2022-test.csv")
    #df = pd.read_csv("feature_matrices/improved_feature_matrix_non_standard.csv").dropna()
    #new_df, standardizer = standardize_features(df)
    #with open("standardizer.p", "wb") as file:
    #    pickle.dump(standardizer, file)
    

