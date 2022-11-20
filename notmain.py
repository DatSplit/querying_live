from preprocessing import preprocess_all_data, preprocess_multiprocess
from feature_extraction import calculate_features_multiprocess
from stats import calculate_statistics
from visualization import generate_plots
#from stats import pose_normalisation
import pymeshlab
import polyscope as ps
from query import standardize_features, df_to_feature_matrix
import numpy as np
import pickle
from dissimilarity import dissimilarity_matrix
from evaluation import query_all_shapes_KNN, confusion_matrix_top_k, topk_precision_recall
import json
import pandas as pd
from visualization import feature_histograms

if __name__ == "__main__": # This is needed because we are doing multiprocessing

    FOLDER_NAME = "run_18-11-2022"

    # PREPROCESSING
    print("preprocessing..")
    #preprocess_multiprocess()

    # FEATURE EXTRACTION
    df = calculate_features_multiprocess()
    df.to_csv(f"{FOLDER_NAME}/non_standard_df.csv")

    df = pd.read_csv(f"{FOLDER_NAME}/non_standard_df.csv")

    # FEATURE CLEANING AND STANDARDIZATION
    new_df = df.dropna(axis=0)
    new_df, standardizer = standardize_features(new_df)
    new_df.to_csv(f"{FOLDER_NAME}/df.csv")
    df = pd.read_csv(f"{FOLDER_NAME}/df.csv")
    with open(f"{FOLDER_NAME}/standardizer.p", "wb") as file:
        pickle.dump(standardizer, file)
    feature_matrix = df_to_feature_matrix(new_df)
    with open(f"{FOLDER_NAME}/feature_matrix.p", "wb") as file:
        pickle.dump(feature_matrix, file)

    for feature in ["a3", "d1", "d2", "d3", "d4"]:
        feature_histograms(feature, df, FOLDER_NAME)

    # # BUILDING DISSIMILARITY MATRIX
    dissim_matrix, id_to_index, index_to_id = dissimilarity_matrix(feature_matrix)
    with open(f"{FOLDER_NAME}/dissim_matrix.p", "wb") as file:
        pickle.dump(dissim_matrix, file)
    with open(f"{FOLDER_NAME}/id_to_index.p", "wb") as file:
        pickle.dump(id_to_index, file)
    with open(f"{FOLDER_NAME}/index_to_id.p", "wb") as file:
        pickle.dump(index_to_id, file)
    with open(f"{FOLDER_NAME}/dissim_matrix.p", "rb") as file:
        dissim_matrix = pickle.load(file)
    with open(f"{FOLDER_NAME}/index_to_id.p", "rb") as file:
        index_to_id = pickle.load(file)


    # EVALUATION
    query_results = query_all_shapes_KNN(dissim_matrix,index_to_id, k=1500)
    with open('labels/id_to_label_49.json', 'rb') as file:
        id_to_label = json.load(file)


    labels = list(set(list(query_results.keys())))
    confusion_matrix_top_k(query_results, id_to_label, k=5)
    confusion_matrix_top_k(query_results, id_to_label, k=30)
    confusion_matrix_top_k(query_results, id_to_label, k=50)
    topk_precision_recall(df, query_results, id_to_label, k=100)
