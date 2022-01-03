import random

from Dataset import RealDataset, SyntheticDataset
from ExtendedIForest import ExtendedIForest
from IForest import IForest
from ITree.INode import INode
from simplex_functions import simplex_hyperplane_points, anomaly_score
from utils_functions import get_data_centroid
from utils_functions import evaluate_results, parametric_equation

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC, OneClassSVM
from sklearn.neural_network import MLPClassifier


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

# files = ['Mulcross.csv']
# files = ['Breastw.csv']
files = ['Http.csv', 'ForestCover.csv']

results = pd.DataFrame()

for dataset_name in files:
    print(dataset_name)
    path = os.path.join(dir_path, dataset_name)
    dataset = RealDataset(path)

    for _ in range(3):
        '''  ######## 5-Fold CV IFOR
        kfolds = KFold(n_splits=5, shuffle=True, random_state=1234).split(dataset.data)
        perfs = []
        for train_indexes, test_indexes in kfolds:
            ifor_cv = IForest()
            ifor_cv.fit(dataset.data[train_indexes])
            depths = ifor_cv.profile(dataset.data[test_indexes])

            scores = []
            for i in range(len(depths)):
                depths_i = depths[i]  # output of i-th test instance

                avg = np.mean(depths_i)  # avg path length
                score = pow(2, -avg / INode.c(256))
                scores.append(score)

            precision, roc_auc = evaluate_results(y_true=dataset.labels[test_indexes],
                                                  y_pred_score=scores,
                                                  y_pred_labels=[1 if s >= 0.5 else 0 for s in scores])
            perfs.append(roc_auc)'''


        ifor = ExtendedIForest(N_ESTIMATORS=100, MAX_SAMPLES=256, dataset=dataset)
        ifor.fit_IForest()
        ifor.profile_IForest()

        # IFOR PREDICTION
        scores = ifor.get_anomaly_scores()
        precision_ifor, roc_auc_ifor = evaluate_results(y_true=ifor.dataset.labels,
                                                        y_pred_score=scores,
                                                        y_pred_labels=[1 if s >= 0.5 else 0 for s in scores])

        # EMBEDDING
        ifor.trees_heights_as_histogram()

        '''# TRAIN TEST SPLIT
        perc_training = 0.8
        #indexes = np.argwhere(ifor.dataset.labels == 0).reshape(-1,)
        indexes = np.arange(ifor.dataset.n_samples)

        n_training = floor(len(indexes) * perc_training)

        rnd = np.random.RandomState(1234)

        training_indexes = rnd.choice(indexes, size=n_training, replace=False)
        test_indexes = np.setdiff1d(np.arange(ifor.dataset.n_samples), training_indexes)'''

        for i in [1]:
            # PCA
            # pca = PCA(n_components=0.95, svd_solver='full')
            pca = PCA(n_components=ifor.histogram_data.shape[1] - i)              # remove only the last component
            histogram_data = pca.fit_transform(ifor.histogram_data)
            components = pca.components_
            variance = pca.explained_variance_
            variance_ratio = pca.explained_variance_ratio_
            # print('variance', variance)
            # print('variance_ratio', variance_ratio)

            '''############## LOF
            for k in [5, 10, 20, 30, 60]:
                for alg in ['auto', 'brute']:
                    for leaf_size in [5, 30, 60, 100]:
                        for c in ['auto']:
                            # 5-FOLD CROSS VALIDATION using DECORRELATED DATA
                            lof = LocalOutlierFactor(n_neighbors=k, algorithm=alg, contamination=c, leaf_size=leaf_size)
                            y_pred_labels = lof.fit_predict(histogram_data)
                            y_pred_labels = [1 if l == -1 else 0 for l in y_pred_labels]
                            y_pred_score = -lof.negative_outlier_factor_

                            _, roc_auc_pca = evaluate_results(y_true=ifor.dataset.labels,
                                                                  y_pred_score=y_pred_score,
                                                                  y_pred_labels=y_pred_labels)
                            print('roc_auc_cv DECORRELATED: ', roc_auc_pca)

                            # 5-FOLD CROSS VALIDATION using NON-DECORRELATED DATA
                            lof = LocalOutlierFactor(n_neighbors=k, algorithm=alg, contamination=c, leaf_size=leaf_size)
                            y_pred_labels = lof.fit_predict(ifor.histogram_data)
                            y_pred_labels = [1 if l == -1 else 0 for l in y_pred_labels]
                            y_pred_score = -lof.negative_outlier_factor_

                            _, roc_auc = evaluate_results(y_true=ifor.dataset.labels,
                                                              y_pred_score=y_pred_score,
                                                              y_pred_labels=y_pred_labels)
                            print('roc_auc_cv NON DECORRELATED: ', roc_auc, '\n')

                            df = pd.DataFrame([[ifor.dataset.dataset_name, roc_auc_ifor,
                                                i, k, alg, leaf_size,
                                                roc_auc_pca, roc_auc]],
                                              columns=['dataset', 'roc_auc_ifor',
                                                       'deleted PCA', 'n_neighbors', 'algorithm', 'leaf_size',
                                                       'roc_auc_pca', 'roc_auc'])

                            results = results.append(df)'''

            ############# OC_SVM
            for kernel in ['poly']:
                for gamma in ['auto']:
                    for nu in [0.99]:
                        # 5-FOLD CROSS VALIDATION using DECORRELATED DATA
                        svm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
                        roc_auc_pca = cross_val_score(svm, histogram_data, ifor.dataset.labels,
                                                      cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc')
                        print('roc_auc_cv DECORRELATED: ', roc_auc_pca)

                        '''# 5-FOLD CROSS VALIDATION using NON-DECORRELATED DATA
                        svm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
                        roc_auc = cross_val_score(svm, ifor.histogram_data, ifor.dataset.labels,
                                                  cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc')
                        print('roc_auc_cv NON DECORRELATED: ', roc_auc)'''

                        '''# 5-FOLD CROSS VALIDATION using ORIGINAL DATA
                        svm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
                        roc_auc_orig = cross_val_score(svm, dataset.data, ifor.dataset.labels,
                                                       cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc')
                        print('roc_auc_cv ORIGINAL: ', roc_auc_orig, '\n')'''

                        df = pd.DataFrame([[ifor.dataset.dataset_name, roc_auc_ifor,
                                            i, kernel, gamma, nu,
                                            roc_auc_pca[0], roc_auc_pca[1], roc_auc_pca[2], roc_auc_pca[3],
                                            roc_auc_pca[4], np.mean(roc_auc_pca)]],
                                            #roc_auc[0], roc_auc[1], roc_auc[2], roc_auc[3],
                                            #roc_auc[4], np.mean(roc_auc),
                                            #roc_auc_orig[0], roc_auc_orig[1], roc_auc_orig[2], roc_auc_orig[3],
                                            #roc_auc_orig[4], np.mean(roc_auc_orig)]],
                                          columns=['dataset', 'roc_auc_ifor',
                                                   'deleted PCA', 'kernel', 'gamma', 'nu',
                                                   'roc_auc_pca - 0', 'roc_auc_pca - 1', 'roc_auc_pca - 2',
                                                   'roc_auc_pca - 3', 'roc_auc_pca - 4', 'MEDIA'])
                                                    #'roc_auc_nopca - 0', 'roc_auc_nopca - 1', 'roc_auc_nopca - 2',
                                                    #'roc_auc_nopca - 3', 'roc_auc_nopca - 4', 'MEDIA',
                                                   #'roc_auc_orig - 0', 'roc_auc_orig - 1', 'roc_auc_orig - 2',
                                                   #'roc_auc_orig - 3', 'roc_auc_orig - 4', 'MEDIA'])

                        results = results.append(df)


            '''########### PROVE SUPERVISED
            for solver in ['lsqr']:
                # 5-FOLD CROSS VALIDATION using DECORRELATED DATA
                lda = LinearDiscriminantAnalysis(solver=solver)
                # lda = LinearSVC(random_state=0, tol=1e-5)
                # lda = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(100, 10), random_state=1, max_iter=500)
                # lda = SVC(C=c, kernel=kernel, gamma=gamma)
                roc_auc_pca = cross_val_score(lda, histogram_data, ifor.dataset.labels,
                                              cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc')
                print('roc_auc_cv DECORRELATED: ', roc_auc_pca)

                # 5-FOLD CROSS VALIDATION using NON-DECORRELATED DATA
                lda = LinearDiscriminantAnalysis(solver=solver)
                # lda = LinearSVC(random_state=0, tol=1e-5)
                # lda = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(10, 2), random_state=1, max_iter=500)
                # lda = SVC(C=c, kernel=kernel, gamma=gamma)
                roc_auc = cross_val_score(lda, ifor.histogram_data, ifor.dataset.labels,
                                          cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc')
                print('roc_auc_cv NON DECORRELATED: ', roc_auc)

                # 5-FOLD CROSS VALIDATION using ORIGINAL DATA
                lda = LinearDiscriminantAnalysis(solver=solver)
                # lda = LinearSVC(random_state=0, tol=1e-5)
                # lda = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(100, 10), random_state=1, max_iter=500)
                # lda = SVC(C=c, kernel=kernel, gamma=gamma)
                roc_auc_orig = cross_val_score(lda, dataset.data, ifor.dataset.labels,
                                               cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc')
                print('roc_auc_cv ORIGINAL: ', roc_auc_orig, '\n')

                df = pd.DataFrame([[ifor.dataset.dataset_name, roc_auc_ifor, i, solver,
                                    roc_auc_pca[0], roc_auc_pca[1], roc_auc_pca[2], roc_auc_pca[3], roc_auc_pca[4],
                                    np.mean(roc_auc_pca),
                                    roc_auc[0], roc_auc[1], roc_auc[2], roc_auc[3], roc_auc[4],
                                    np.mean(roc_auc),
                                    roc_auc_orig[0], roc_auc_orig[1], roc_auc_orig[2], roc_auc_orig[3], roc_auc_orig[4],
                                    np.mean(roc_auc_orig)]],
                                  columns=['dataset',  'roc_auc_ifor', 'deleted PCA', 'solver',
                                           'roc_auc_pca - 0', 'roc_auc_pca - 1', 'roc_auc_pca - 2',
                                           'roc_auc_pca - 3', 'roc_auc_pca - 4', 'MEDIA',
                                           'roc_auc_nopca - 0', 'roc_auc_nopca - 1', 'roc_auc_nopca - 2',
                                           'roc_auc_nopca - 3', 'roc_auc_nopca - 4', 'MEDIA',
                                           'roc_auc_orig - 0', 'roc_auc_orig - 1', 'roc_auc_orig - 2',
                                           'roc_auc_orig - 3', 'roc_auc_orig - 4', 'MEDIA'
                                           ])

                results = results.append(df)'''

        '''lda.fit(histogram_data[training_indexes], ifor.dataset.labels[training_indexes])

        par = lda.get_params()

        # TEST LDA
        y_pred_labels = lda.predict(histogram_data[test_indexes])
        y_pred = lda.predict_proba(histogram_data[test_indexes])[:, 1]

        precision, roc_auc = evaluate_results(y_true=ifor.dataset.labels[test_indexes],
                                              y_pred_score=y_pred,
                                              y_pred_labels=y_pred_labels)'''



with pd.ExcelWriter('output.xlsx', mode='w') as writer:
    results.to_excel(writer, index=False)