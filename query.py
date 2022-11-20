import pandas as pd
import numpy as np
import streamlit as st
from feature_extraction import extract_features, standardize_features
import pymeshlab
import os
import meshio
import numpy as np
import plotly.graph_objects as go
from preprocessing import preprocess
from dissimilarity import dissimilarity, dissimilarity_matrix, most_similar_meshes_query, most_similar_meshes_query_ANN
import pickle

FEATURES = ['mesh_area', 'mesh_compactness', 'mesh_rectangularity', 'mesh_diameter', 'mesh_eccentricity', 'mesh_volume', 'mesh_convexity']

def histograms_to_vector(df):
    column_list = ["mesh_a3", "mesh_d1", "mesh_d2","mesh_d3", "mesh_d4"]
    histogram_lists = []
    for column in column_list:
        histogram_list = np.vstack(df[column].apply(lambda x: 
                           np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' ')))
        histogram_lists.append(histogram_list)
    histograms = np.hstack(histogram_lists)
               
    return histograms


def dict_to_feature_vector(dictionary):
    """
    Transforms a dictionary as received from extract_features to a feature vector.

        Parameters:
            dictionary (dict): A dictionary as created in extract_features
        
        Returns:
            feature_vector (numpy.1darray) numpy array with only the values representing the features and the ID
    """
    histograms = []
    for histo in ["mesh_a3", "mesh_d1", "mesh_d2","mesh_d3", "mesh_d4"]:
        histograms.append(dictionary[histo])
    histograms = np.concatenate(histograms)
    dataframe = pd.DataFrame.from_dict(dictionary)
    dataframe = dataframe[['ID'] + FEATURES].iloc[0] # TODO this needs to be refactored. It makes a dataframe of length 10 because of the 10 values of the histograms.
    feature_vector = dataframe.to_numpy()
    feature_vector = np.append(feature_vector, histograms)
    return feature_vector

def df_to_feature_matrix(dataframe):
    """
    Transforms a pandas DataFrame into a numpy feature matrix. 

        Parameters:
                dataframe (pandas.DataFrame) the dataframe as created by features.py

        Returns:
                feature_matrix (numpy.ndarray) numpy array with only the values representing the features and the ID
    """
    histogram_vectors = histograms_to_vector(dataframe)
    dataframe = dataframe[['ID'] + FEATURES]
    feature_matrix = dataframe.to_numpy()
    feature_matrix = np.hstack((feature_matrix, histogram_vectors))
    return feature_matrix


def streamlit_show_mesh(st, filename, dist, df_row):
    print(f'meshes_raw/{filename}')
    msh = meshio.read(f'meshes_raw/{filename}')
    verts = msh.points
    I, J, K =  msh.cells_dict["triangle"].T
    x, y, z = verts.T
    largest_side = max(max(x),max(y),max(z))
    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z,i=I, j=J, k=K, color='lightpink', opacity=0.50)])
    fig.update_layout(
    scene = dict(
        xaxis = dict(nticks=4, range=[0,largest_side],),
                     yaxis = dict(nticks=4, range=[0,largest_side],),
                     zaxis = dict(nticks=4, range=[0,largest_side],),),)
    fig.update_layout(scene_aspectmode='cube')
    # st.write('Original mesh') TODO write a description
    col1, col2 = st.columns([1,2])
    with col2:
        st.plotly_chart(fig)
    with col1:
        st.header(filename)
        st.text(f"Dissimilarity: {dist} \nClass label: {df_row['class'].values[0]}")
        #st.text(filename)
        #st.text(f"Dissimilarity: {dist}")
        #st.text(f"Class label: {df_row['class'].values[0]}")

def standardized_using_standardizer(query_dict):
    with open("run_18-11-2022/standardizer.p", "rb") as file:
        standardizer = pickle.load(file)
    features = ['mesh_area', "mesh_compactness", "mesh_rectangularity","mesh_diameter", "mesh_eccentricity", "mesh_volume", "mesh_convexity"]
    
    for i in range(len(features)):
        
        query_dict[features[i]] = (query_dict[features[i]] - standardizer[features[i]][0]) /standardizer[features[i]][1]
    
    return query_dict

def console_query_interface():
    df = pd.read_csv("non_standardized_feature_vectors.csv", index_col=0)    
    feature_matrix = df_to_feature_matrix(df)

    mesh_name = input("Which model would you like to query?")
    ("Preprocessing..")
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(f'meshes_raw/{mesh_name}.obj')
    print("Preprocessing..")
    ms = preprocess(ms)
    ms.save_current_mesh(f'tempDir/{mesh_name}.obj')
    print("Extracting features..")
    query_features = extract_features(f'tempDir/{mesh_name}.obj') #returns dict -> df
    query_features = dict_to_feature_vector(query_features)
    print("Matching..")
    a = most_similar_meshes_query(query_features,feature_matrix)
    print(a)
    id1 = a[0][1] 
    id2 = a[1][1]
    id3 = a[2][1]
    print(df[df['ID'] == id1])
    print(df[df['ID'] == id1]['class'])
    print(df[df['ID'] == id2])
    print(df[df['ID'] == id2]['class'])
    print(df[df['ID'] == id3])
    print(df[df['ID'] == id3]['class'])

    os.remove(f'tempDir/{mesh_name}.obj')

def streamlit_query_interface(folder_name):
    import os
    st.title("3D Mesh Query")
    query_object = st.file_uploader(label='Upload mesh here')
    df = pd.read_csv(f"{folder_name}/df.csv", index_col=0)    

    feature_matrix = df_to_feature_matrix(df)
    
    bool_query = False
    weights = None
    neigh_col1, col2 = st.columns([1,4]) 
    with neigh_col1:
        neighbour_method = st.selectbox("Neighbour method",["KNN","ANN"])
        k = st.number_input("Amount of results", min_value=1, max_value=30, value=5,step=1) 

    if neighbour_method == "KNN":
        with col2:
            w1 = st.slider("Elementary features",min_value=0.0,max_value=1.0,value=0.5,step=0.1, help="Adjust the weight to get a new value")
            w2 = st.slider("A3",min_value=0.0,max_value=1.0,value=0.5,step=0.1, help="Adjust the weight to get a new value")
            w3 = st.slider("D1",min_value=0.0,max_value=1.0,value=0.5,step=0.1, help="Adjust the weight to get a new value")
            w4 = st.slider("D2",min_value=0.0,max_value=1.0,value=0.5,step=0.1, help="Adjust the weight to get a new value")
            w5 = st.slider("D3",min_value=0.0,max_value=1.0,value=0.5,step=0.1, help="Adjust the weight to get a new value")
            w6 =st.slider("D4",min_value=0.0,max_value=1.0,value=0.5,step=0.1, help="Adjust the weight to get a new value")
        weights = [w1, w2, w3, w4, w5, w6]


    st.text("-----------------------------------------------------------------------------------")
    if st.button(label="Query!"):
        bool_query = True

    while(bool_query):
        
        with open(os.path.join("tempDir",query_object.name),"wb") as f:
                f.write(query_object.getbuffer())

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(f'tempDir/{query_object.name}')
        ms = preprocess(ms)
        ms.save_current_mesh(f'tempDir/{query_object.name}')
        query_features = extract_features(f'tempDir/{query_object.name}') #returns dict -> df
        query_features = standardized_using_standardizer(query_features)
        query_features = dict_to_feature_vector(query_features)

        if(neighbour_method == "ANN"):
            print("Using ANN")
            a = most_similar_meshes_query_ANN(query_features,feature_matrix)
        if(neighbour_method == "KNN"):
            print("Using KNN")
            a = most_similar_meshes_query(query_features, feature_matrix, weights)
        print(a)
        for dist, id in a:
            streamlit_show_mesh(st, f"m{id}.obj", dist, df[df['ID'] == id])

        import os
        os.remove(fr'tempDir/{query_object.name}')
        return 

if __name__ == '__main__':
    streamlit_query_interface('run_18-11-2022')
    

