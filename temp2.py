from sklearn.manifold import TSNE
import pickle
import matplotlib.pyplot as plt
import matplotlib
import json
FOLDER_NAME = "run_16-11-2022"

with open(f"{FOLDER_NAME}/dissim_matrix.p", "rb") as file:
    dissim_matrix = pickle.load(file)
with open(f"{FOLDER_NAME}/index_to_id.p", "rb") as file:
    index_to_id = pickle.load(file)
with open('labels/id_to_label_49.json', 'rb') as file:
    id_to_label = json.load(file)
y  = TSNE(verbose=3, metric="precomputed",method="exact", n_iter=1_000).fit_transform(dissim_matrix)

labels = []
for i in range(len(dissim_matrix)):
    labels.append(id_to_label[str(int(index_to_id[int(i)]))])

colors = []
for name, hex in matplotlib.colors.cnames.items():
    colors.append(name)
label_to_color = {}
c = []
for label in labels:
    if label not in label_to_color.keys():
        label_to_color[label] = colors.pop()
    c.append(label_to_color[label])


labels = ["blue"] *len(dissim_matrix)
labels[0] = "red"
plt.scatter(y[:,0],y[:,1], c=c)
plt.show()
print("done")