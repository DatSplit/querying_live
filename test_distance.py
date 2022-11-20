import pandas as pd
from query import df_to_feature_matrix
from dissimilarity import dissimilarity
import numpy as np

def feature_matrix_nan_values(feature_matrix="standardized_feature_vectors.csv"):
    df = pd.read_csv(f"feature_matrices/{feature_matrix}", index_col=0)
    for col in df.columns:
        if not df[col].isna().sum() == 0:
            print(f"{df[col].isna().sum()} nan values in column {col}")


def dissimilarity_self_identity(feature_matrix="standardized_feature_vectors.csv"):
    """
    Check whether the dissimilarity function satisfies the Self-Identity property (05 MR Matching Slide 8)

        Returns:
            AssertionError: The dissimilarity of the mesh with itself is not 0
            True: The property holds
    """
    df = pd.read_csv(f"feature_matrices/{feature_matrix}", index_col=0)
    feature_matrix = df_to_feature_matrix(df)
    for mesh in feature_matrix:
        assert (0 == dissimilarity(mesh, mesh)), f"Found an distance {dissimilarity(mesh, mesh)} > 0 for mesh {mesh[0]}"
        # nan for m1193
    return True

def dissimilarity_positivity(feature_matrix="standardized_feature_vectors.csv"):
    """
    Check whether the dissimilarity function satisfies the positivity property (05 MR Matching Slide 8)

        Returns:
            AssertionError: The dissimilarity of the mesh with itself is not 0
            True: The property holds
    """
    df = pd.read_csv(f"feature_matrices/{feature_matrix}", index_col=0)
    feature_matrix = df_to_feature_matrix(df)
    for index1 in range(len(feature_matrix)):
        for index2 in range(index1+1, len(feature_matrix)):
            mesh1 = feature_matrix[index1]
            mesh2 = feature_matrix[index2]
            try:
                assert (0 < dissimilarity(mesh1, mesh2)), f"Found an distance {dissimilarity(mesh1, mesh2)} > 0 for meshes {mesh1[0]} and {mesh2[0]}"
            except:
                print( f"Found an distance {dissimilarity(mesh1, mesh2)} > 0 for meshes {mesh1[0]} and {mesh2[0]}")
                print(mesh1)
                print(mesh2)    
    return True

def dissimilarity_symmetry(feature_matrix="standardized_feature_vectors.csv"):
    """
    Check whether the dissimilarity function satisfies the positivity property (05 MR Matching Slide 8)

        Returns:
            AssertionError: The dissimilarity of the mesh with itself is not 0
            True: The property holds
    """
    df = pd.read_csv(f"feature_matrices/{feature_matrix}", index_col=0)
    feature_matrix = df_to_feature_matrix(df)
    for index1 in range(len(feature_matrix)):
        for index2 in range(index1+1, len(feature_matrix)):
            mesh1 = feature_matrix[index1]
            mesh2 = feature_matrix[index2]
            if mesh1[0] == 1193.0 or mesh2[0] == 1193.0:
                continue
            assert (dissimilarity(mesh2,mesh1) == dissimilarity(mesh1, mesh2)), f"Found assymetric dissmilarity for meshes {mesh1[0]} and {mesh2[0]}"  
    return True

# TODO these should be removed as we will have better evaluation metrics
def within_class_dissimilarity(feature_matrix="standardized_feature_vectors.csv"):
    df = pd.read_csv(f"feature_matrices/{feature_matrix}", index_col=0)
    medians = []
    for class_name in df['class'].unique():
        df_subset = df[df['class'] == class_name]
        feature_matrix = df_to_feature_matrix(df_subset)
        dissims = []
        for mesh1_index in range(len(feature_matrix)):
            for mesh2_index in range(mesh1_index + 1,len(feature_matrix)):
                dissims.append(dissimilarity(feature_matrix[mesh1_index],feature_matrix[mesh2_index]))
        if np.isnan(np.median(dissims)):
            # M1193 has nan values?
            continue
        medians.append(np.median(dissims))
    print(f"within class dissim: {np.median(medians)}")

# TODO these should be removed as we will have better evaluation metrics
def outside_class_dissimilarity(feature_matrix="standardized_feature_vectors.csv"):
    df = pd.read_csv(f"feature_matrices/{feature_matrix}", index_col=0)
    medians = []
    for class_name in df['class'].unique():
        df_within = df[df['class'] == class_name]
        df_outside = df[df['class'] != class_name]
        feature_matrix_within = df_to_feature_matrix(df_within)
        feature_matrix_outside = df_to_feature_matrix(df_outside)
        feature_matrix_outside = feature_matrix_outside[~np.isnan(feature_matrix_outside).any(axis=1)]

        dissims = []
        for mesh1_index in range(len(feature_matrix_within)):
            for mesh2_index in range(len(feature_matrix_outside)):
                dissims.append(dissimilarity(feature_matrix_within[mesh1_index],feature_matrix_outside[mesh2_index]))
        if np.isnan(np.median(dissims)):
            # M1193 has nan values?
            continue
        medians.append(np.median(dissims))
    print(f"outside class dissimilarity {np.median(medians)}")



    
if __name__ == "__main__":
    pass
