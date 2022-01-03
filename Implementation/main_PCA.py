from Dataset import RealDataset, SyntheticDataset
from ExtendedIForest import ExtendedIForest
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


def score_on_last_component():
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
    files.remove('wood.csv')'''

    # files = ['Breastw.csv']
    # files = ['Shuttle.csv', 'Smtp.csv']

    results = pd.DataFrame()

    for dataset_name in files:
        print(dataset_name)
        path = os.path.join(dir_path, dataset_name)
        dataset = RealDataset(path)

        for i in range(1):
            ifor = ExtendedIForest(N_ESTIMATORS=100, MAX_SAMPLES=256, dataset=dataset)
            ifor.fit_IForest()
            ifor.profile_IForest()
            ifor.trees_heights_as_histogram()

            '''# TRAIN TEST SPLIT
            perc_training = 0.8
            indexes = np.argwhere(ifor.dataset.labels == 0).reshape(-1,)

            n_training = floor(len(indexes) * perc_training)

            rnd = np.random.RandomState(1234)

            training_indexes = rnd.choice(indexes, size=n_training, replace=False)
            test_indexes = np.setdiff1d(np.arange(ifor.dataset.n_samples), training_indexes)

            # TRAN TEST SAMPLES
            training_samples = ifor.histogram_data[training_indexes]
            test_samples = ifor.histogram_data[test_indexes]'''

            # ENTIRE DATASET
            training_samples = ifor.histogram_data
            test_samples = ifor.histogram_data

            # PCA
            # pca = PCA(n_components=0.95, svd_solver='full')
            pca = PCA(n_components=training_samples.shape[1] - 1)  # remove only the last component
            pca.fit(training_samples)
            components = pca.components_
            variance = pca.explained_variance_
            variance_ratio = pca.explained_variance_ratio_
            print('variance', variance)
            print('variance_ratio', variance_ratio)
            components_transf = pca.transform(components)

            '''# draw data
            anomaly_data = test_samples[ifor.dataset.labels == 1]
            anomaly_data = pca.transform(anomaly_data)
            xdata_anomaly, ydata_anomaly, zdata_anomaly = zip(*anomaly_data)
            fig = go.Figure(data=[go.Scatter3d(x=xdata_anomaly,
                                               y=ydata_anomaly,
                                               z=zdata_anomaly,
                                               mode='markers',
                                               marker=dict(size=3, color='red'),
                                               connectgaps=False,
                                               name="anomaly")])

            normal_data = test_samples[ifor.dataset.labels == 0]
            normal_data = pca.transform(normal_data)
            xdata_normal, ydata_normal, zdata_normal = zip(*normal_data)
            fig.add_scatter3d(x=xdata_normal,
                              y=ydata_normal,
                              z=zdata_normal,
                              mode='markers',
                              marker=dict(size=3, color='green'),
                              connectgaps=False,
                              name="normal")

            colors = ['blue', 'orange', 'yellow']
            for i in range(3):
                centroid = get_data_centroid(pca.transform(test_samples))
                parameters = np.concatenate([centroid, components_transf[i]], axis=0).reshape(2, 3)
                line_points = parametric_equation(parameters)
                xline, yline, zline = zip(*line_points)
                fig.add_scatter3d(x=xline,
                                  y=yline,
                                  z=zline,
                                  mode='lines',
                                  marker=dict(size=3, color=colors[i]),
                                  connectgaps=False,
                                  name="normal")'''

            test_samples_pca = pca.transform(test_samples)

            data_label = zip(test_samples_pca, ifor.dataset.labels)
            distances_dict = []

            centroid = get_data_centroid(test_samples_pca)
            parameters = np.concatenate([centroid, components_transf[centroid.shape[0] - 1]], axis=0).reshape(2,
                                                                                                              centroid.shape[
                                                                                                                  0])

            t1 = -20
            t2 = 20

            # extract two points on the Line
            A = parameters[0] + parameters[1] * t1
            B = parameters[0] + parameters[1] * t2

            ba = B - A  # distance vector that define the direction of the Line

            for dat, l in data_label:
                P = dat
                pa = P - A  # distance vector from the point A on the Line to the point P

                t = np.dot(pa, ba) / np.dot(ba, ba)
                d = np.linalg.norm(pa - t * ba, ord=2)

                distances_dict.append({'t_opt': t, 'p_opt': t * ba + A, 'dist_opt': d, 'label': l})

                '''x_opt, y_opt, z_opt = t * ba + A
                fig.add_scatter3d(x=[x_opt, P[0]],
                                  y=[y_opt, P[1]],
                                  z=[z_opt, P[2]],
                                  mode='lines',
                                  marker=dict(size=3, color='purple'),
                                  connectgaps=False,
                                  name="dist")

            fig.show()'''

            distances = [v['dist_opt'] for v in distances_dict]

            test_labels = ifor.dataset.labels  # [test_indexes]

            precision, roc_auc = evaluate_results(y_true=test_labels,
                                                  y_pred_score=distances,
                                                  y_pred_labels=[1 if e > 0.04 else 0 for e in distances])

            '''# score as reconstruction error
            diff = test_samples - pca.inverse_transform(pca.transform(test_samples))

            errors = [np.linalg.norm(el, ord=2) for el in diff]
            errors = np.asarray(errors)

            test_labels = ifor.dataset.labels  # [test_indexes]

            precision, roc_auc = evaluate_results(y_true=test_labels,
                                                  y_pred_score=errors,
                                                  y_pred_labels=[1 if e > 0.04 else 0 for e in errors])'''

            '''normal_indexes = np.argwhere(test_labels == 0).reshape(-1,)
            anomaly_indexes = np.setdiff1d(np.arange(len(test_labels)), normal_indexes)

            plt.plot([min(errors), max(errors)], [0, 0], 'b-')
            plt.plot([min(errors), max(errors)], [1, 1], 'b-')

            plt.plot(errors[normal_indexes], [1]*len(normal_indexes), 'go')
            plt.plot(errors[anomaly_indexes], [0] * len(anomaly_indexes), 'ro')

            plt.title(dataset_name)

            plt.show()'''

            df = pd.DataFrame([[ifor.dataset.dataset_name, roc_auc]],
                              columns=['dataset', 'roc_auc'])

            '''df = pd.DataFrame([[ifor.dataset.dataset_name, variance[-4], variance[-3], variance[-2], variance[-1],
                                variance_ratio[-4], variance_ratio[-3], variance_ratio[-2], variance_ratio[-1]]],
                              columns=['dataset', 'variance -4', 'variance -3', 'variance -2', 'variance -1',
                                       'variance_ratio -4', 'variance_ratio -3', 'variance_ratio -2', 'variance_ratio -1'])'''

            results = results.append(df)

    with pd.ExcelWriter('output.xlsx', mode='w') as writer:
        results.to_excel(writer, index=False)


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
files.remove('wood.csv')'''

#files = ['Breastw.csv']
#files = ['Shuttle.csv', 'Smtp.csv']

results = pd.DataFrame()

for dataset_name in files:
    print(dataset_name)
    path = os.path.join(dir_path, dataset_name)
    dataset = RealDataset(path)

    for i in range(1):
        ifor = ExtendedIForest(N_ESTIMATORS=100, MAX_SAMPLES=256, dataset=dataset)
        ifor.fit_IForest()
        ifor.profile_IForest()
        ifor.trees_heights_as_histogram()

        '''# TRAIN TEST SPLIT
        perc_training = 0.8
        indexes = np.argwhere(ifor.dataset.labels == 0).reshape(-1,)

        n_training = floor(len(indexes) * perc_training)

        rnd = np.random.RandomState(1234)

        training_indexes = rnd.choice(indexes, size=n_training, replace=False)
        test_indexes = np.setdiff1d(np.arange(ifor.dataset.n_samples), training_indexes)
        
        # TRAN TEST SAMPLES
        training_samples = ifor.histogram_data[training_indexes]
        test_samples = ifor.histogram_data[test_indexes]'''

        # ENTIRE DATASET
        training_samples = ifor.histogram_data
        test_samples = ifor.histogram_data

        # PCA
        #pca = PCA(n_components=0.95, svd_solver='full')
        pca = PCA(n_components=training_samples.shape[1])            # remove only the last component
        pca.fit(training_samples)
        components = pca.components_
        variance = pca.explained_variance_
        variance_ratio = pca.explained_variance_ratio_
        print('variance', variance)
        print('variance_ratio', variance_ratio)
        components_transf = pca.transform(components)

        '''# draw data
        anomaly_data = test_samples[ifor.dataset.labels == 1]
        anomaly_data = pca.transform(anomaly_data)
        xdata_anomaly, ydata_anomaly, zdata_anomaly = zip(*anomaly_data)
        fig = go.Figure(data=[go.Scatter3d(x=xdata_anomaly,
                                           y=ydata_anomaly,
                                           z=zdata_anomaly,
                                           mode='markers',
                                           marker=dict(size=3, color='red'),
                                           connectgaps=False,
                                           name="anomaly")])

        normal_data = test_samples[ifor.dataset.labels == 0]
        normal_data = pca.transform(normal_data)
        xdata_normal, ydata_normal, zdata_normal = zip(*normal_data)
        fig.add_scatter3d(x=xdata_normal,
                          y=ydata_normal,
                          z=zdata_normal,
                          mode='markers',
                          marker=dict(size=3, color='green'),
                          connectgaps=False,
                          name="normal")

        colors = ['blue', 'orange', 'yellow']
        for i in range(3):
            centroid = get_data_centroid(pca.transform(test_samples))
            parameters = np.concatenate([centroid, components_transf[i]], axis=0).reshape(2, 3)
            line_points = parametric_equation(parameters)
            xline, yline, zline = zip(*line_points)
            fig.add_scatter3d(x=xline,
                              y=yline,
                              z=zline,
                              mode='lines',
                              marker=dict(size=3, color=colors[i]),
                              connectgaps=False,
                              name="normal")'''

        ############################
        '''
        test_samples_pca = pca.transform(test_samples)

        # score as reconstruction error
        diff = test_samples - pca.inverse_transform(pca.transform(test_samples))

        errors = [np.linalg.norm(el, ord=2) for el in diff]
        errors = np.asarray(errors)

        test_labels = ifor.dataset.labels  # [test_indexes]

        precision, roc_auc = evaluate_results(y_true=test_labels,
                                              y_pred_score=errors,
                                              y_pred_labels=[1 if e > 0.04 else 0 for e in errors])'''
        #############################

        '''normal_indexes = np.argwhere(test_labels == 0).reshape(-1,)
        anomaly_indexes = np.setdiff1d(np.arange(len(test_labels)), normal_indexes)

        plt.plot([min(errors), max(errors)], [0, 0], 'b-')
        plt.plot([min(errors), max(errors)], [1, 1], 'b-')

        plt.plot(errors[normal_indexes], [1]*len(normal_indexes), 'go')
        plt.plot(errors[anomaly_indexes], [0] * len(anomaly_indexes), 'ro')

        plt.title(dataset_name)

        plt.show()'''

        #############################
        '''df = pd.DataFrame([[ifor.dataset.dataset_name, roc_auc]],
                          columns=['dataset', 'roc_auc'])'''
        #############################

        df = pd.DataFrame([[ifor.dataset.dataset_name,
                            variance[-5], variance[-4], variance[-3], variance[-2], variance[-1],
                            variance_ratio[-5], variance_ratio[-4], variance_ratio[-3],
                            variance_ratio[-2], variance_ratio[-1]]],
                          columns=['dataset', 'variance[-5]', 'variance[-4]', 'variance[-3]',
                                   'variance[-2]', 'variance[-1]',
                                   'variance_ratio[-5]', 'variance_ratio[-4]', 'variance_ratio[-3]',
                                   'variance_ratio[-2]', 'variance_ratio[-1]'])

        results = results.append(df)

with pd.ExcelWriter('output.xlsx', mode='w') as writer:
    results.to_excel(writer, index=False)
