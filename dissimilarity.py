from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cityblock,cosine
import numpy as np
from operator import itemgetter
from utils import get_id_to_idx
import pandas as pd
import pickle
from numba import jit
from wasserstein_manual_import import my_wasserstein
import pynndescent

def dissimilarity(vector1, vector2, weights = [1,1,1,1,1,1]):
    """
    Returns the dissimilarity between two feature vectors. (05 MR Matching Slide 3)

        Parameters:
                vector1 (numpy.1darray): feature vector 1
                vector2 (numpy.1darray): feature vector 2

        Returns:
                dissimilarity (float): the dissimilarity between the two vectors
    """
    dissimilarity = 0
    dissimilarity += l2_distance(vector1[1:8], vector2[1:8])*weights[0]
    
    dissimilarity += earth_movers_distance(vector1[8:18], vector2[8:18])   * weights[1] #a3
    dissimilarity += earth_movers_distance(vector1[18:28], vector2[18:28]) * weights[2] #d1
    dissimilarity += earth_movers_distance(vector1[28:38], vector2[28:38]) * weights[3] #d2
    dissimilarity += earth_movers_distance(vector1[38:48], vector2[38:48]) * weights[4] #d3
    dissimilarity += earth_movers_distance(vector1[48:58], vector2[48:58]) * weights[5] #d4
    return dissimilarity

@jit(nopython=True)
def dissimilarity_jit(vector1, vector2):
    """
    Returns the dissimilarity between two feature vectors. (05 MR Matching Slide 3)

        Parameters:
                vector1 (numpy.1darray): feature vector 1
                vector2 (numpy.1darray): feature vector 2

        Returns:
                dissimilarity (float): the dissimilarity between the two vectors
    """
    dissimilarity = 0
    dissimilarity += l2_distance_jit(vector1[1:8], vector2[1:8])*2
    
    dissimilarity += earth_movers_distance_jit(vector1[8:18], vector2[8:18])   * 1    #a3
    dissimilarity += earth_movers_distance_jit(vector1[18:28], vector2[18:28]) * 0.1 #d1
    dissimilarity += earth_movers_distance_jit(vector1[28:38], vector2[28:38]) * 0.8 #d2
    dissimilarity += earth_movers_distance_jit(vector1[38:48], vector2[38:48]) * 0.2 #d3
    dissimilarity += earth_movers_distance_jit(vector1[48:58], vector2[48:58]) * 0.2 #d1

    return dissimilarity

def l1_distance(vector1,vector2):
    """
    Returns the Manhattan distance between two feature vectors. (05 MR Matching Slide 16)

        Parameters:
                vector1 (numpy.1darray): (subset of a) feature vector 1
                vector2 (numpy.1darray): (subset of a) feature vector 2

        Returns:
                distance (float): the Manhatten distance between the two vectors.
    """
    return cityblock(vector1, vector2)

def cosine_distance(vector1,vector2):
    """
    Returns the cosine distance between two feature vectors. (05 MR Matching Slide 19)

        Parameters:
                vector1 (numpy.1darray): (subset of a) feature vector 1
                vector2 (numpy.1darray): (subset of a) feature vector 2

        Returns:
                distance (float): the cosine distance between the two vectors.
    """
    return cosine(vector1,vector2)

# TODO Quadratic Form Distance ?
def quadratic_form_distance(vector1,vector2):
    """
    Returns the quadratic form distance between two feature vectors. (05 MR Matching Slide 20)

        Parameters:
                vector1 (numpy.1darray): (subset of a) feature vector 1
                vector2 (numpy.1darray): (subset of a) feature vector 2

        Returns:
                distance (float): the quadratic form distance between the two vectors.
    """
    raise NotImplementedError

@jit(nopython=True)
def earth_movers_distance_jit(histogram1,histogram2):
    """
    Returns the earth movers distance between two histograms. (05 MR Matching Slide 21)

        Parameters:
                histogram1 (numpy.1darray): histogram represented as a 1d array where the values are the normalized counts
                vector2 (numpy.1darray): histogram represented as a 1d array where the values are the normalized counts

        Returns:
                distance (float): the earth movers distance between the two histograms.
    """
    return my_wasserstein(histogram1, histogram2) #TODO Change this back

def earth_movers_distance(histogram1,histogram2):
    """
    Returns the earth movers distance between two histograms. (05 MR Matching Slide 21)

        Parameters:
                histogram1 (numpy.1darray): histogram represented as a 1d array where the values are the normalized counts
                vector2 (numpy.1darray): histogram represented as a 1d array where the values are the normalized counts

        Returns:
                distance (float): the earth movers distance between the two histograms.
    """
    return my_wasserstein(histogram1, histogram2) #TODO Change this back

# TODO Transportation Distance


def l2_distance(vector1,vector2):
    """
    Returns the Euclidian distance between two feature vectors. (05 MR Matching Slide 16)

        Parameters:
                vector1 (numpy.ndarray): feature vector 1
                vector2 (numpy.ndarray): feature vector 2

        Returns:
                distance (float): the Manhatten distance between the two vectors.
    """

    dist = np.linalg.norm(vector1-vector2)
    return dist

@jit(nopython=True)
def l2_distance_jit(vector1,vector2):
    """
    Returns the Euclidian distance between two feature vectors. (05 MR Matching Slide 16)

        Parameters:
                vector1 (numpy.ndarray): feature vector 1
                vector2 (numpy.ndarray): feature vector 2

        Returns:
                distance (float): the Manhatten distance between the two vectors.
    """

    dist = np.linalg.norm(vector1-vector2)
    return dist



def dissimilarity_matrix(feature_matrix, weights=[1,1,1,1,1,1]):
    """
    Returns the dissimilarity matrix (05 MR Matching Slide 16)

        Parameters:
                feature_matrix (numpy.ndarray): a matrix with meshes as rows and features as columns

        Returns:
                dissimilarity_matrix (numpy.ndarray): a matrix with at index [x,y] the dissimilarity between mesh x and mesh y.
    """
    dissim_matrix = np.zeros((len(feature_matrix),len(feature_matrix)))
    
    index_to_id = {}
    id_to_index = {}
    for mesh1_index in range(len(feature_matrix)):
        id = feature_matrix[mesh1_index][0]
        index_to_id[mesh1_index] = id
        id_to_index[id] = mesh1_index
        print(mesh1_index)
        for mesh2_index in range(mesh1_index + 1,len(feature_matrix)):

            dissim_matrix[mesh1_index][mesh2_index] = dissimilarity(feature_matrix[mesh1_index],feature_matrix[mesh2_index], weights=weights)
            if np.isnan(dissimilarity(feature_matrix[mesh1_index],feature_matrix[mesh2_index])):
                print(f"{mesh1_index} {mesh2_index} ")
                continue
            dissim_matrix[mesh2_index][mesh1_index] = dissim_matrix[mesh1_index][mesh2_index]

    return dissim_matrix, id_to_index, index_to_id


def most_similar_meshes_query(query_vector,feature_matrix,weights = [1,1,1,1,1,1], k=5):
    """
    Calculates the top k most similar meshes for an unseen query.

        Parameters:
                query_vector (numpy.1darray): vector representing the features of the query
                feature_matrix (numpy.ndarray): matrix containing the feature vectors of the database
        
        Returns:
                top_k (list<tuple>): a list sorted for the smallest distance as a tuple with 0 being the distance and 1 being the ID
    """
    sim_vector = list(tuple())
    list_vectors = feature_matrix

    for i in range(len(list_vectors)):
        sim_vector.append((dissimilarity(query_vector,feature_matrix[i], weights),int(feature_matrix[i][0])))
    return sorted(sim_vector,key=itemgetter(0))[:k]

def most_similar_meshes_query_ANN(query_vector, feature_matrix, k=5):    
    with open("run_18-11-2022/ANN_index.p", "rb") as file:
        index = pickle.load(file)
    indexes, distances = index.query([query_vector])
    indexes = indexes[0]
    distances = distances[0]
    result = []
    for i in range(k):
        print(indexes[i])
        tuple = (distances[i], int(feature_matrix[indexes[i]][0]))
        result.append(tuple)
    return result


