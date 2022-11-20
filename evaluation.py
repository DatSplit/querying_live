import pickle
import numpy as np
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, top_k_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from query import df_to_feature_matrix
import random
from feature_extraction import standardize_features

def query_all_shapes_KNN(dissim_matrix,index_to_id, k=1500):
    """
    Queries all shapes in the database. Mostly used for calculating metrics.
        
        Parameters:
            

        Returns:
            query_results (dict<ID,list<ID>>) a dictionary with for each id a list of the IDs of the top k results. 
    """
    results = {}
    for query_index in range(len(dissim_matrix)):
        sorted_indexes = np.argsort(dissim_matrix[query_index])
        temp_list = []
        for result_index in sorted_indexes[:k]:
            temp_list.append(index_to_id[result_index])
        results[index_to_id[query_index]] = temp_list
    return results

def query_all_shapes_ANN(feature_matrix,k=1500):
    results = {}
    with open("ANN_index.p", "rb") as file:
        index = pickle.load(file)
    indexes_lists, distance_lists = index.query(feature_matrix, k=k, epsilon=0.5)
    for i in range(len(indexes_lists)):
        temp_list = []
        for j in range(len(indexes_lists[i])):
            result_index = indexes_lists[i][j]
            id = feature_matrix[result_index][0]
            temp_list.append(id)
        query_id =feature_matrix[i][0] 
        results[query_id] = temp_list
    return results

def confusion_matrix_f(query_results, id_to_label):
    
    
    y_true = []
    y_pred = []


    for id in query_results.keys():
        y_true.append(id_to_label[str(int(id))])
        result_id = query_results[id][1]
        y_pred.append(id_to_label[str(int(result_id))])
        
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    for i in range(len(cm)):
        sum = float(np.sum(cm[i]))
        for j in range(len(cm[i])):
            cm[i][j] = float(cm[i][j])/sum*100
    sn.heatmap(cm, cmap="OrRd")
    print(classification_report(y_true, y_pred))
    plt.show()

def confusion_matrix_top_k(query_results, id_to_label, k=5):
    
    
    label_to_number = {}
    all_labels = list(set(id_to_label.values()))
    for i in range(len(all_labels)):
        label_to_number[all_labels[i]] = i
    
    y_true = []
    y_pred = []


    for id in query_results.keys():
        query_label = id_to_label[str(int(id))]
        y_true.append(label_to_number[query_label])
        result = np.zeros((len(all_labels),))

        for result_id in query_results[id][1:k+1]:
            result_label = id_to_label[str(int(result_id))]
            result[label_to_number[result_label]] += 1
        result = result/np.sum(result)
        y_pred.append(result)


    print(f"Top {k} accuracy score: ", top_k_accuracy_score(y_true, y_pred))
    return top_k_accuracy_score(y_true, y_pred)

def topk_precision_recall(df, query_results, id_to_label,k=5):
    y_true = []
    precisions = []

    df['class'] = df['ID'].apply(lambda x: id_to_label[str(x)])
    counts = df['class'].value_counts()
    print(counts.mean())
    recalls = []
    

    for id in query_results.keys():
        query_label = id_to_label[str(int(id))]
        tp = 0
        y_true.append(query_label)
        for result_id in query_results[id][1:k+1]:
            result_label = id_to_label[str(int(result_id))]
            if query_label == result_label:
                tp += 1
        precision =tp/k
        precisions.append(precision) 
        recall = tp/counts[query_label]
        recalls.append(recall)
    print(np.mean(precisions))
    print(np.mean(recalls))
    prec = np.mean(precisions)
    rec = np.mean(recalls)
    return prec, rec
        
            

def ROC_curve(query_results, id_to_label, k=5):
    senses = []
    specifs = []

    df = pd.read_csv("feature_matrices/12-11-2022-test.csv")
    counts = df['class'].value_counts()
    total_n = len(df)

    for id in query_results.keys():
        query_label = id_to_label[str(int(id))]
        true_pos = 0
        false_pos = 0
        for result_id in query_results[id][1:k+1]:
            result_label = id_to_label[str(int(result_id))]
            if query_label == result_label:
                true_pos += 1
            if query_label != result_label:
                false_pos += 1
        senses.append(true_pos/counts[query_label])
        true_neg = total_n - counts[query_label] - false_pos
        specifs.append(true_neg/(total_n - counts[query_label]))
    return np.mean(senses), np.mean(specifs) 


if __name__ == "__main__":
    pass