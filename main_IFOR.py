from Dataset import RealDataset, SyntheticDataset
from ExtendedIForest import ExtendedIForest
from ITree.INode import INode
from utils_functions import evaluate_results

import os
import numpy as np
import pandas as pd
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

files = ['Mammography.csv']
#files = ['Shuttle.csv', 'Smtp.csv']

for dataset_name in files:
    print(dataset_name)
    path = os.path.join(dir_path, dataset_name)
    dataset = RealDataset(path)

    ifor = ExtendedIForest(N_ESTIMATORS=100, MAX_SAMPLES=256, dataset=dataset)
    ifor.fit_IForest()
    ifor.profile_IForest()
    # ifor.trees_heights_as_histogram()

    scores = ifor.get_anomaly_scores()
    y_pred_labels = [1 if s >= 0.5 else 0 for s in scores]

    precision, roc_auc = evaluate_results(y_true=ifor.dataset.labels,
                                          y_pred_score=scores,
                                          y_pred_labels=y_pred_labels)

    # if correction - this make depths with no correction factor
    '''ifor.depths = np.clip(ifor.depths, 0, 8)
    scores = ifor.get_anomaly_scores()

    precision, roc_auc = evaluate_results(y_true=ifor.dataset.labels,
                                          y_pred_score=scores,
                                          y_pred_labels=y_pred_labels)'''

    ifor_sk = IsolationForest()
    ifor_sk.fit(dataset.data)

    y_pred_labels_sk = ifor_sk.predict(dataset.data)
    y_pred_labels_sk = [1 if l == -1 else 0 for l in y_pred_labels_sk]
    scores_sk = - ifor_sk.score_samples(dataset.data)

    precision_sk, roc_auc_sk = evaluate_results(y_true=dataset.labels,
                                                y_pred_score=scores_sk,
                                                y_pred_labels=y_pred_labels_sk)

    df = pd.DataFrame([[ifor.dataset.dataset_name, precision, roc_auc, precision_sk, roc_auc_sk]],
                      columns=['dataset', 'precision', 'roc_auc', 'precision_sk', 'roc_auc_sk'])

    results = results.append(df)

with pd.ExcelWriter('output.xlsx', mode='w') as writer:
    results.to_excel(writer, index=False)