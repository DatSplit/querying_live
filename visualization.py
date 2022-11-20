from multiprocessing.sharedctypes import Value
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import pymeshlab
import os
from statistics import median
import numpy as np
from numpy.linalg import eigh
import polyscope as ps
from stats import absolute_angle
from pymeshlab import polyscope_functions
from query import df_to_feature_matrix
import random
import json
from preprocessing import preprocess, resampling

#https://github.com/garrettj403/SciencePlots
plt.style.reload_library()
plt.style.use(['science'])
# def invariant_translation_histogram(distance,dataset):


def generate_plots(folder_name,dataset):
    df = pd.read_csv(f"{folder_name}/tabular_data/statistics_{dataset}")
    # df = df[df["ID"] != "m94"]
    # df = df[df["ID"] != "m1785"]
    # df = df[df["ID"] != "m1784"]
    # selection = set()
    # for dim in ['dim_x','dim_y','dim_z']:
    #     print(df[dim].describe().apply(lambda x: format(x, 'f')))
    #     selection = set.union(set(df[df[dim] > 1.0001]["ID"]), selection)
    # print(selection)
    # print(len(selection))

    # l = []
    # for i in range(len(df)):
    #     l.append(max(df.iloc[i][6], df.iloc[i][7], df.iloc[i][8]))
    # print(min(l))
    # print(max(l))

    # weird = []
    # for i in range(len(df)):
    #     if df.loc[i][5] in selection:
    #         weird.append(i)
    # print(df.loc[weird].describe())

    color = "#FBA27B"
    dataset = dataset + "2"
    os.makedirs(f"plots/{dataset}", exist_ok=True)

    plt.hist(df["n_faces"], edgecolor='black',color=color, bins=30)
    plt.savefig(f"plots/{dataset}/n_faces.png")
    plt.clf()

    plt.hist(df["n_vertexes"], edgecolor='black', color=color, bins=30)
    plt.savefig(f"plots/{dataset}/n_vertexes.png")
    plt.clf()

    plt.hist(df["absolute_angle"], edgecolor='black', color=color, bins=30)
    plt.savefig(f"plots/{dataset}/absolute_angle.png")
    plt.clf()
    
    plt.hist(df["flipped"], edgecolor='black', color=color)
    plt.savefig(f"plots/{dataset}/flipped.png")
    plt.clf()

    counter = defaultdict(int)
    for c in df["class"]:
        counter[c] += 1

    # plt.style.use([ 'science'])
    print(counter.values())
    print(counter.keys())
    plt.pie(counter.values(), labels=counter.keys())
    plt.savefig(f"plots/{dataset}/classes.png")
    plt.clf()


def comparison_plots(preprocessed_dataset, raw_dataset, label_x, label_y, title, comparison,autobinwidth):
    df = pd.read_csv(f"tabular_data/statistics_{preprocessed_dataset}")
    df2 = pd.read_csv(f"tabular_data/statistics_{raw_dataset}")
    os.makedirs(f"plots/{preprocessed_dataset}", exist_ok=True)
    plt.rcParams['figure.figsize'] = (15, 9)
    plt.rcParams['font.size'] = 10
    if(autobinwidth):
        bins2 = np.histogram(np.stack((df[comparison], df2[comparison])), bins=30)[1]
        plt.hist(df2[comparison], alpha=0.5, label="Before refinement",bins=bins2)
        plt.hist(df[comparison], alpha=0.5, label="After refinement",bins=bins2)
    else:
        
        plt.hist(df2[comparison], alpha=0.5, label="Before refinement")
        plt.hist(df[comparison], alpha=0.5, label="After refinement")

    plt.xlabel(label_x, size=20)
    plt.ylabel(label_y, size=20)
    plt.title(title)
    plt.legend(loc='upper right')
    plt.savefig(f"plots/{preprocessed_dataset}/comparison_{comparison}.png")
    plt.clf()


def print_3d_model(path):
    # This PyMeshLab method does not work as intended
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    ms.save_snapshot(imagefilename="test.png")


def average_model(dataset, attribute):
    df = pd.read_csv(f"tabular_data/statistics_{dataset}")
    av = sum(df[attribute]) / len(df)
    print(av)
    window = 100

    average_model = df[(df[attribute] > av - window) & (df[attribute] < av + window)]
    print(average_model)
    average_path = average_model["path"].values[0]
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(average_path)
    ms.show_polyscope()
    ms.clear()


def outlier_model(dataset, attribute):
    df = pd.read_csv(f"tabular_data/statistics_{dataset}")
    window = 100

    outliers = df.sort_values(by=attribute, ascending=False)
    outlier_path = outliers["path"].values[2]
    print(outlier_path)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(outlier_path)
    ms.show_polyscope()
    ms.clear()

def feature_histograms(feature, df, folder):
    
    feature_to_range = {
        "a3": (8,18),
        "d1": (18,28),
        "d2": (28,38),
        "d3": (38,48),
        "d4": (48,58)
    }

    
    range_l, range_r = feature_to_range[feature]
    with open("labels/id_to_label_49.json", "rb") as file:
        ID_TO_LABEL = json.load(file)
    print("Changing classes")
    df['class'] = df['ID'].apply(lambda x: ID_TO_LABEL[str(x)])

    
                            # 9,6 for 49/53 labels
    fig, axs = plt.subplots(9, 6, sharex=True, sharey=True)
    #axes = axs
    axes = []
    for ax_x in axs:
        for ax_y in ax_x:
            axes.append(ax_y)

    ax_index = 0
    for class_name in df['class'].unique():
        print(len(df['class'].unique()))
        print(class_name)
        ax = axes[ax_index]
        ax_index += 1
        small_df = df[df['class'] == class_name]
        vector_matrix = df_to_feature_matrix(small_df)
        ax.set_title(class_name, fontsize= 7)
        for row in vector_matrix:
            ax.plot(row[range_l:range_r])
            ax.set_ylim(top=1)
    fig.set_size_inches(8.3, 11.7)
    plt.tight_layout()
    plt.savefig(f"{folder}/{feature}.png")
        
    #plt.show()

def feature_outliers(feature):
    pass



if __name__ == "__main__":
    #generate_plots("preprocessed")
    
    #generate_plots("raw")

    # df = pd.read_csv("feature_matrices/feature_vectors_2_standardized.csv")


    for feature in ["a3", "d1", "d2", "d3", "d4"]:
        feature_histograms(feature, df, folder)
    #pass
