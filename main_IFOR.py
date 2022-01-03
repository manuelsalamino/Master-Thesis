from Dataset import RealDataset, SyntheticDataset
from ExtendedIForest import ExtendedIForest
from ITree.INode import INode
from utils_functions import evaluate_results
from IForest import IForest

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from math import floor
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


# create SYNTHETIC dataset
#files = []
#files.append(SyntheticDataset(n_samples=300, anomalies_rate=0.1, dataset_type="blobs", show_data=True, n_blobs=1))
#files.append(SyntheticDataset(n_samples=300, anomalies_rate=0.1, dataset_type="blobs", show_data=True, n_blobs=2))
#files.append(SyntheticDataset(n_samples=300, anomalies_rate=0.1, dataset_type="circles", show_data=True, noise=0))
#files.append(SyntheticDataset(n_samples=300, anomalies_rate=0.1, dataset_type="circles", show_data=True, noise=0.05))
#files.append(SyntheticDataset(n_samples=300, anomalies_rate=0.1, dataset_type="moons", show_data=True, noise=0))
#files.append(SyntheticDataset(n_samples=300, anomalies_rate=0.1, dataset_type="moons", show_data=True, noise=0.1))
#files.append(SyntheticDataset(n_samples=300, anomalies_rate=0.1, dataset_type="line", show_data=True))


# upload REAL dataset
dir_path = os.path.join(os.getcwd(), '../datasets')
files = os.listdir(dir_path)
'''files.remove('ForestCover.csv')
files.remove('Http.csv')
files.remove('Mulcross.csv')
files.remove('Smtp.csv')
files.remove('Shuttle.csv')
files.remove('Pendigits.csv')
files.remove('hbk.csv')
files.remove('wood.csv')
'''

results = pd.DataFrame()

#files = ['Mammography.csv']
#files = ['Shuttle.csv', 'Smtp.csv']

for dataset_name in files:
    for j in range(8):
        print(dataset_name)
        path = os.path.join(dir_path, dataset_name)
        dataset = RealDataset(path)

        # TRAIN TEST SPLIT SEMI-SUPERVISED (TRAIN ONLY NORMALS)
        perc_training = 0.8
        indexes = np.argwhere(dataset.labels == 0).reshape(-1, )

        n_training = floor(len(indexes) * perc_training)
        #n_training = dataset.n_samples - dataset.n_anomalies*2

        rnd = np.random.RandomState(None)

        training_indexes = rnd.choice(indexes, size=n_training, replace=False)
        test_indexes = np.setdiff1d(np.arange(dataset.n_samples), training_indexes)

        # DATASET DATA
        training_samples = dataset.data[training_indexes]
        test_samples = dataset.data[test_indexes]

        #y_true = [1 if l == 0 else 0 for l in dataset.labels[test_indexes]]

        ifor_semi = IForest(n_estimators=100, max_samples=256)
        ifor_semi.fit(training_samples)
        depths = ifor_semi.profile(test_samples)

        scores = []
        for i in range(len(depths)):
            avg = np.mean(depths[i])    # avg path length
            score = pow(2, -avg / INode.c(256))
            scores.append(score)

        y_pred_labels = [1 if s >= 0.5 else 0 for s in scores]

        _, roc_auc_semi = evaluate_results(y_true=dataset.labels[test_indexes],
                                           y_pred_score=scores,
                                           y_pred_labels=y_pred_labels)

        # if correction - this make depths with no correction factor
        '''ifor.depths = np.clip(ifor.depths, 0, 8)
        scores = ifor.get_anomaly_scores()
    
        precision, roc_auc = evaluate_results(y_true=ifor.dataset.labels,
                                              y_pred_score=scores,
                                              y_pred_labels=y_pred_labels)'''

        # TRAIN TEST SPLIT NORMAL AND ANOMALIES
        #perc_training = 0.8
        #indexes = np.arange(dataset.n_samples)

        #n_training = floor(len(indexes) * perc_training)
        #n_training = dataset.n_samples - dataset.n_anomalies * 2

        #rnd = np.random.RandomState(1234)

        #training_indexes = rnd.choice(indexes, size=n_training, replace=False)
        #test_indexes = np.setdiff1d(np.arange(dataset.n_samples), training_indexes)

        # DATASET DATA
        training_samples, test_samples, train_labels, test_labels = train_test_split(dataset.data,
                                                                                     dataset.labels,
                                                                                     test_size=0.2,
                                                                                     stratify=dataset.labels,
                                                                                     random_state=None)

        #training_samples = dataset.data[training_indexes]
        #test_samples = dataset.data[test_indexes]

        # y_true = [1 if l == 0 else 0 for l in dataset.labels[test_indexes]]

        ifor_semi = IForest(n_estimators=100, max_samples=256)
        ifor_semi.fit(training_samples)
        depths = ifor_semi.profile(test_samples)

        scores = []
        for i in range(len(depths)):
            avg = np.mean(depths[i])  # avg path length
            score = pow(2, -avg / INode.c(256))
            scores.append(score)

        y_pred_labels = [1 if s >= 0.5 else 0 for s in scores]

        _, roc_auc_unsupervised = evaluate_results(y_true=test_labels,
                                                   y_pred_score=scores,
                                                   y_pred_labels=y_pred_labels)

        # UNSUPERVISED (TRAIN AND TEST ALL DATASET)
        ifor_semi = IForest(n_estimators=100, max_samples=256)
        ifor_semi.fit(dataset.data)
        depths = ifor_semi.profile(dataset.data)

        scores = []
        for i in range(len(depths)):
            avg = np.mean(depths[i])  # avg path length
            score = pow(2, -avg / INode.c(256))
            scores.append(score)

        y_pred_labels = [1 if s >= 0.5 else 0 for s in scores]

        _, roc_auc_full_dataset = evaluate_results(y_true=dataset.labels,
                                                   y_pred_score=scores,
                                                   y_pred_labels=y_pred_labels)

        df = pd.DataFrame([[dataset.dataset_name, roc_auc_semi, roc_auc_unsupervised, roc_auc_full_dataset]],
                          columns=['dataset', 'roc_auc_semi', 'roc_auc_unsupervised', 'roc_auc_full_dataset'])

        results = results.append(df)

with pd.ExcelWriter('output.xlsx', mode='w') as writer:
    results.to_excel(writer, index=False)