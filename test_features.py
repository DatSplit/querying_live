import pandas as pd
from query import df_to_feature_matrix
from features import d1, d2, d3, d4, volume, a3
import pymeshlab
import utils
import numpy as np
from feature_extraction import standardize_features, extract_features
import utils
from preprocessing import preprocess, centering, scaling
import matplotlib.pyplot as plt
from dissimilarity import earth_movers_distance
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
from dissimilarity import dissimilarity
import pickle

def histograms_normalization(feature_matrix="feature_vectors_2.csv"):
    df = pd.read_csv(f"feature_matrices/{feature_matrix}", index_col=0)
    feature_matrix = df_to_feature_matrix(df)
    for feature_vector in feature_matrix:
        for l_range in range(6,56,10):
            hist_sum = sum(feature_vector[l_range:l_range + 10]) 
            if feature_vector[0] != 1432:
                assert hist_sum > 1 - 0.1 and hist_sum < 1 + 0.1, f"histogram sums to {hist_sum} for ID {feature_vector[0]}"


def feature_standardization(feature_matrix="feature_vectors_2.csv"):
    df = pd.read_csv(f"feature_matrices/{feature_matrix}", index_col=0)
    for col in df.columns:
        if col not in ['mesh_a3', 'mesh_d1', "mesh_d2", "mesh_d3", "mesh_d4", "ID", "class"]:
            assert df[col].std() > 1 - 0.0001 and df[col].std() < 1 + 0.0001, str(df[col].std())
            assert df[col].mean() > -0.0001 and df[col].mean() < 0.0001, df[col].mean()


def histograms_normalization_online(n=5):
    i = 0
    for path in utils.all_paths("preprocessed"):
        if i > n:
            return
        i += 1
        vector = extract_features(path)
        for histo in ['mesh_a3', "mesh_d1", "mesh_d2", "mesh_d3", "mesh_d4"]:
            hist_sum = sum(vector[histo]) 
            print(hist_sum)
            assert hist_sum > 1 - 0.000001 and hist_sum < 1 + 0.000001, f"histogram sums to {hist_sum} for ID {vector[0]}"

def test_volume():
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh("meshes_debug/cube.obj")
    ms = centering(ms)
    ms = scaling(ms)
    print(volume(ms))
    ms.load_new_mesh("meshes_debug/rectangle.obj")
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_snap_mismatched_borders()
    ms.meshing_re_orient_faces_coherentely()
    ms.meshing_close_holes(maxholesize=20000,selfintersection =False)
    ms = preprocess(ms)
    print(volume(ms))

def test_feature_consistency():
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh("meshes_preprocessed/m48.obj")
    for i in range(5):
        hist = a3(ms)
        plt.plot(hist)
    plt.show()

def confusion_matrix_feature(feature_matrix):
    feature_matrix = feature_matrix[feature_matrix[:,0].argsort()]
    heatmap = np.zeros(shape=(len(feature_matrix),len(feature_matrix)))
    distances = []
    for mesh1_index in range(len(feature_matrix)):
        print(mesh1_index)
        for mesh2_index in range(mesh1_index + 1,len(feature_matrix)):
            fv1 = feature_matrix[mesh1_index]
            fv2 = feature_matrix[mesh2_index]
            distance = np.abs(fv1[5] - fv2[5])
            #distance = earth_movers_distance(fv1[16:26], fv2[16:26])#d1
            heatmap[mesh1_index, mesh2_index] = distance
            heatmap[mesh2_index, mesh1_index] = distance
            distances.append(distance)
    print(heatmap)
    sns.heatmap(heatmap, norm=LogNorm())
    plt.show()
    plt.hist(distances)
    plt.show()
    

def within_class_between_class_confusion_matrix(df, distance_function=dissimilarity):
    labels = df["class"].unique()
    confusion_matrix = np.zeros((len(labels), len(labels)))
    
    # Iterate over all classes
    for label_index_1 in range(len(labels)):
        print(labels[label_index_1])
        df_subset_1 = df[df['class'] == labels[label_index_1]]
        feature_matrix_1 = df_to_feature_matrix(df_subset_1)

        within_distances = []
        for within_vector_1 in feature_matrix_1:
            for within_vector_2 in feature_matrix_1:
                within_distance = distance_function(within_vector_1, within_vector_2)
                within_distances.append(within_distance)
        confusion_matrix[label_index_1,label_index_1] = np.mean(within_distances)

            

        # Iterate over all class combinations
        for label_index_2 in range(label_index_1 + 1,len(labels)):
            df_subset_2 = df[df['class'] == labels[label_index_2]]
            feature_matrix_2 = df_to_feature_matrix(df_subset_2)
            between_distances = []
            for vector1 in feature_matrix_1:
                for vector2 in feature_matrix_2:
                    distance = distance_function(vector1,vector2)
                    between_distances.append(distance)
            confusion_matrix[label_index_1,label_index_2] = np.mean(between_distances)
            confusion_matrix[label_index_2,label_index_1] = np.mean(between_distances)
    print(confusion_matrix)
    # for i in range(len(confusion_matrix)):
    #     std = np.std(confusion_matrix[i])
    #    confusion_matrix[i] = confusion_matrix[i]/ std
    #     confusion_matrix[:,i] = confusion_matrix[:,i] / std
    # #     confusion_matrix[i,i] * std

    c_map = sns.color_palette("cubehelix", as_cmap=True)
    sns.heatmap(confusion_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap=c_map)
    
    plt.show()
    

def boxplot_features(df, feature):
    labels = df["class"].unique()
    
    boxplots= []
    # Iterate over all classes
    for label_index_1 in range(len(labels)):
        print(labels[label_index_1])
        df_subset_1 = df[df['class'] == labels[label_index_1]]
        boxplots.append(df_subset_1[feature])
    plt.boxplot(boxplots, labels=labels)
    plt.show()    


def test_sample_size(feature_function):
    paths = utils.all_paths("preprocessed")
    selected_paths = np.random.choice(paths, size=100, replace=False)
    dists_per_n = []
    sample_sizes = [100,200,300,400,500,800, 1_000,3_000, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 100_000]
    for n_samples in sample_sizes:
        print(n_samples)
        distances = []
        for path in selected_paths:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(path)
            hist1 = feature_function(ms, samples=n_samples, seed=np.random.randint(0,100))
            hist2= feature_function(ms, samples=n_samples, seed=np.random.randint(0,100))
            distances.append(earth_movers_distance(hist1, hist2))
        dists_per_n.append(distances)
    
    x = sample_sizes
    y = [np.mean(y) for y in dists_per_n]
    yerr = [np.std(y) for y in dists_per_n]
    with open(f"{feature_function.__name__}_plot.p", "wb") as file:
        pickle.dump((x,y,yerr), file)
    with open(f"{feature_function.__name__}_plot.p", "rb") as file:
        x,y, yerr = pickle.load(file)
    
    plt.errorbar(x=x, y=y, yerr=yerr)
    with open(f"{feature_function.__name__}_plot.png", "wb") as file:
        plt.savefig(file)
    plt.show()



# Test standardization
if __name__ == "__main__":
    for function in [d4]:
        # RERUN D2
        test_sample_size(function)


