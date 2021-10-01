import sympy
from sympy import Matrix
from sympy.solvers.solveset import linsolve
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from math import ceil

from IForest import IForest
from ITree.INode import INode


def IFOR_AS_Simplex_hyperplane_intersection():

    # define symbols (variables of equations)
    string = ''
    for i in range(1, n_dim+1):
        string += 'h' + str(i) + ', '
    string = string[:-2]

    symbols = sympy.symbols(string)

    # IFOR hyperplane
    # MATRIX [H1, H2, ... Hn, KNOWN TERM (on the right part of the equation)]
    ifor_eq = [i for i in range(1, n_dim+1)]
    ifor_eq.append(0)            # [1, 2, 3, 4, ..., n_dim, 0]   ->   h1 + 2*h2 + ... + n_dim*hn = 0

    # simplex
    simplex_eq = [1] * (n_dim+1)          # [1, 1, 1, ..., 1, 1]   ->   h1 + h2 + ... + hn = 1

    print(linsolve(Matrix((ifor_eq, simplex_eq)), symbols))


#def IFOR_AS_normal_projection_on_simplex():


def ifor_wo_correction_factor():
    # upload REAL dataset
    dir_path = os.path.join(os.getcwd(), '../datasets')
    files = os.listdir(dir_path)

    results = pd.DataFrame()

    for dataset_name in files:
        for i in range(5):
            print(dataset_name)
            path = os.path.join(dir_path, dataset_name)
            dataset = pd.read_csv(path)
            labels = np.array(dataset.iloc[:, (-1)])
            dataset = dataset.drop(["id", "y"], axis=1).to_numpy()

            ifor = IForest()
            ifor.fit(dataset)
            depths = ifor.profile(dataset)

            depths_wo_correction = np.clip(depths, 0, ceil(np.log2(256)))

            scores, scores_wo_correction = [], []
            for i in range(len(depths)):
                avg = np.mean(depths[i])        # avg path length
                score = pow(2, -avg / INode.c(256))
                scores.append(score)

                avg = np.mean(depths_wo_correction[i])          # avg path length
                score = pow(2, -avg / INode.c(128))
                scores_wo_correction.append(score)

            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)

            print("roc w/ correction", roc_auc)

            fpr, tpr, _ = roc_curve(labels, scores_wo_correction)
            roc_auc_wo = auc(fpr, tpr)

            print("roc w/o correction", roc_auc_wo)

            df = pd.DataFrame(data={'dataset': dataset_name[:-4],
                                    'ROC AUC con Correzione': roc_auc,
                                    'ROC AUC senza Correzione': roc_auc_wo},
                              index=[0])

            results = results.append(df)

    with pd.ExcelWriter('C:\\Users\\Manuel\\Desktop\\ifor_correction_factor.xlsx', mode='w') as writer:
        results.to_excel(writer, index=False)


# upload REAL dataset
dir_path = os.path.join(os.getcwd(), '../datasets')
files = os.listdir(dir_path)

results = pd.DataFrame()

for dataset_name in files:
    for i in range(5):
        print(dataset_name)
        path = os.path.join(dir_path, dataset_name)
        dataset = pd.read_csv(path)

        labels = np.array(dataset.iloc[:, (-1)])
        dataset = dataset.drop(["id", "y"], axis=1).to_numpy()

        ifor = IForest()
        ifor.fit(dataset)

        depths = ifor.profile(dataset)
        depths_wo_correction = np.clip(depths, 0, ceil(np.log2(256)))

        print("w/ correction", i, depths.max())
        print("w/o correction", i, depths_wo_correction.max())

''' scores, scores_wo_correction = [], []
for i in range(len(depths)):
    avg = np.mean(depths[i])  # avg path length
    score = pow(2, -avg / INode.c(256))
    scores.append(score)

    avg = np.mean(depths_wo_correction[i])  # avg path length
    score = pow(2, -avg / INode.c(128))
    scores_wo_correction.append(score)'''