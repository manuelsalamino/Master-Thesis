from Dataset import RealDataset, SyntheticDataset
from ExtendedIForest import ExtendedIForest
from ITree.INode import INode
from simplex_functions import simplex_hyperplane_points, anomaly_score
from utils_functions import evaluate_results

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil, floor
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from plotly import graph_objects as go
import plotly.express as px
#from skspatial.objects import Point, Points, Line
from scipy.optimize import least_squares
from scipy.odr import RealData, Data, Model, ODR, polynomial
import time
from sklearn.decomposition import PCA


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
files.remove('ForestCover.csv')
files.remove('Http.csv')
files.remove('Mulcross.csv')
files.remove('Smtp.csv')
files.remove('Shuttle.csv')
files.remove('Pendigits.csv')
files.remove('hbk.csv')
files.remove('wood.csv')


results = pd.DataFrame()

#files = ['Breastw.csv', 'Annthyroid.csv']
files = ['ForestCover.csv']

for dataset_name in files:
    print(dataset_name)
    path = os.path.join(dir_path, dataset_name)
    dataset = RealDataset(path)

    for i in range(1):
        ifor = ExtendedIForest(N_ESTIMATORS=100, MAX_SAMPLES=256, dataset=dataset)
        ifor.fit_IForest()
        ifor.profile_IForest()
        ifor.trees_heights_as_histogram()

        # IFOR PREDICTION
        scores = ifor.get_anomaly_scores()
        precision_ifor, roc_auc_ifor = evaluate_results(y_true=ifor.dataset.labels,
                                                        y_pred_score=scores,
                                                        y_pred_labels=[1 if s >= 0.5 else 0 for s in scores],
                                                        print_results=True)

        df = ifor.OC_Svm()
        #df = ifor.LOF()

        results = results.append(df)

with pd.ExcelWriter('output.xlsx', mode='w') as writer:
    results.to_excel(writer, index=False)
