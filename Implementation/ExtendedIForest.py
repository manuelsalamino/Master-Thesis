from IForest import IForest
from Dataset import AbstractDataset, SyntheticDataset
from ITree.INode import INode
from utils_functions import parametric_equation, distance_matrix, get_data_centroid,\
    evaluate_results, line_point_distance
from simplex_functions import simplex_hyperplane_points, perfect_anomalies_hyperplane_points

import numpy as np
import pandas as pd
from math import ceil, floor
from scipy.optimize import least_squares
from sklearn.metrics import roc_curve, auc, precision_score
from plotly import graph_objects as go
import matplotlib.pyplot as plt
import time
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
import warnings

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)


class ExtendedIForest:
    N_ESTIMATORS: int
    MAX_SAMPLES: int
    dataset: AbstractDataset
    ifor: IForest
    depths: np.ndarray
    histogram_data: np.ndarray
    parameters: dict

    def __init__(self, N_ESTIMATORS, MAX_SAMPLES, dataset):
        self.N_ESTIMATORS = N_ESTIMATORS
        self.MAX_SAMPLES = MAX_SAMPLES
        self.dataset = dataset

        self.ifor = IForest(n_estimators=self.N_ESTIMATORS, max_samples=self.MAX_SAMPLES)
        self.depths = None

    def fit_IForest(self):
        data = self.dataset.data
        self.ifor.fit(data)

    def profile_IForest(self):
        data = self.dataset.data
        self.depths = self.ifor.profile(data)

    def get_anomaly_scores(self):
        if self.depths is None:
            self.profile_IForest()

        scores = []
        for i in range(len(self.depths)):
            depths_i = self.depths[i]         # output of i-th test instance

            avg = np.mean(depths_i)        # avg path length
            score = pow(2, -avg / INode.c(self.MAX_SAMPLES))
            scores.append(score)

        return np.asarray(scores)

    def trees_heights_as_histogram(self, show_plot=False):                 # create histogram with the tree height
        if self.depths is None:
            self.profile_IForest()

        max_depth = ceil(np.max(self.depths))
        histogram = []

        if show_plot:                # if show plots, compute scores in order to print them
            scores = self.get_anomaly_scores()

        for i in range(len(self.depths)):
            depths_i = self.depths[i]                # output of i-th test instance

            # "max_depths + 2" to have the last interval [max_depth, max_depth+1]
            # if max_depth = 8 last interval is [8, 9] , otherwise [7, 8] (two significant depths in the same interval)
            h = np.histogram(depths_i, range(1, max_depth + 2))

            if show_plot and (i < 5 or i > self.dataset.n_samples-6):     # plot histogram of some normals and anomalies
                title = self.dataset.dataset_name + ' - anomaly score: ' + str(scores[i])[:6] + ' --> Real label: '
                if self.dataset.labels[i] == 1:
                    plt.hist(depths_i, range(1, max_depth + 1), color='#ff7f0e')
                    title += 'ANOMALY'
                else:
                    plt.hist(depths_i, range(1, max_depth + 1), color='#1f77b4')
                    title += 'NORMAL'
                plt.title(title)
                plt.xticks(ticks=np.arange(0, max_depth+1, step=2), labels=np.arange(0, max_depth+1, step=2))
                plt.yticks(ticks=np.arange(0, max(h[0]) + 10, step=10),
                           labels=[str(n)[:3] for n in np.arange(0, max(h[0])/100 + 0.1, step=0.1)])
                plt.show()

            h = h[0] / self.N_ESTIMATORS

            histogram.append(h)

        histogram = np.asarray(histogram)
        self.histogram_data = histogram

        return histogram

    def fit_histogram_points(self, degree=1):

        def point_approximation(points, coeff):
            t_line = np.arange(-1, 1, 0.1)
            estimated_points = []

            for point in points:
                dist_opt = np.inf
                p_opt = 0
                best_t = {0: np.inf, 1: np.inf, 2: np.inf}
                for t in t_line:
                    p = [0] * n_variables
                    for d in range(degree + 1):
                        p += coeff[d] * (t ** d)
                    dist = np.linalg.norm(point - p)
                    if dist < max(best_t.values()):
                        max_key = max(best_t, key=best_t.get)
                        del best_t[max_key]
                        best_t[t] = dist

                t_min = min(best_t.keys())
                t_max = max(best_t.keys())
                t_line_restrict = np.arange(t_min, t_max, 0.01)

                for t in t_line_restrict:
                    p = [0] * n_variables
                    for d in range(degree + 1):
                        p += coeff[d] * (t ** d)
                    dist = np.linalg.norm(point - p)
                    if dist < dist_opt:
                        dist_opt = dist
                        p_opt = p
                # p_opt is my best fitting for point
                estimated_points.append(p_opt)

            return estimated_points

        def estimate_least_squares(x, data, centroid):
            n_variables = data.shape[1]

            if n_variables == 1:
                data = [d[0] for d in data]
                data = np.asarray(data)

            degree = int(len(x) / n_variables)

            coeff = np.concatenate([centroid, x]).reshape(degree + 1, n_variables)

            estimate_data = point_approximation(data, coeff)

            '''# Test all samples through each estimator using multithreading
            N_THREADS = 1
            BATCH_SIZE = 1
            n_samples = len(data)
            start = time.perf_counter()

           with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
                futures = []
                for iter in range(ceil(n_samples/BATCH_SIZE)):
                    futures.append(executor.submit(point_approximation,
                                                   points=data[iter*BATCH_SIZE:iter*BATCH_SIZE + BATCH_SIZE],
                                                   coeff=coeff))
                    n_samples -= BATCH_SIZE
                print("before wait: ", time.perf_counter() - start)
                wait(futures)
                print("after wait: ", time.perf_counter() - start)
                estimate_data = np.concatenate([future.result() for future in futures])'''

            distances = [np.linalg.norm(np.subtract(estimate_data[i], data[i]), ord=2) for i in range(len(data))]
            error = np.linalg.norm(distances, 2) ** 2
            print(f'error: {error}')
            end = time.perf_counter()
            #print("time: ", end-start)
            return distances

        n_variables = self.histogram_data.shape[1]

        # fit points
        n_restarts = 1
        best_error = np.inf

        centroid = get_data_centroid(self.histogram_data)

        # func = Model(estimate)
        # func = polynomial(order=2)
        # xdata1, xdata2, ydata = zip(*data)
        # xdata = np.asarray([xdata1, xdata2])
        # ydata = np.asarray([xdata2, ydata])
        # mydata = RealData(xdata, ydata)

        for _ in range(n_restarts):
            # fit using Ordinary Least Square
            if 'parameters' not in dir(self):
                x0 = np.random.rand(degree * n_variables)
            else:
                x0 = np.random.rand(degree * n_variables)
                par = self.parameters.reshape(-1)
                x0[:len(par)] = par

            print(x0)
            opt = least_squares(estimate_least_squares, x0=x0, ftol=1e-05,
                                args=([self.histogram_data, centroid]))
            error = np.linalg.norm(opt.fun, ord=2)   # / data.shape[0]
            popt = np.concatenate([centroid, opt.x]).reshape(degree + 1, n_variables)
            print(popt)

            # visualize plots
            fig = self.plot(parameters=popt, error=error)
            if fig is not None:
                if isinstance(fig, list):
                    for f in fig:
                        f.show()
                else:
                    fig.show()

            # fit using ODR
            # odr = ODR(mydata, func, beta0=np.random.rand((degree+1)*n_variables))
            # out = odr.run()
            # error_odr = out.sum_square
            # popt_odr = np.asarray(out.beta).reshape(degree + 1, n_variables)
            # print(popt_odr)

            # visualize plots (if possible: 2D or 3D)
            # fig = plot(data=data, labels=labels, parameters=popt_odr, error=error_odr)
            # fig.show()

            # fit data using ODR --> initial parameters are the ones found with ordinary least square
            # myodr = ODR(mydata, func, beta0=opt.x)
            # myoutput = myodr.run()
            # popt = np.asarray(myoutput.beta).reshape(degree + 1, n_variables)
            # error = myoutput.sum_square
            # print(popt)

            # visualize plots (if possible: 2D or 3D)
            # fig = plot(data=data, labels=labels, parameters=popt, error=error)
            # fig.show()

            if error < best_error:
                best_popt = popt
                best_error = error

        best_popt = np.asarray(best_popt).reshape(degree + 1, n_variables)
        self.parameters = best_popt

        return best_popt

    def OC_Svm(self):
        perc_training = 0.8
        #indexes = np.argwhere(self.dataset.labels == 0).reshape(-1,)
        indexes = np.arange(self.dataset.n_samples)

        n_training = floor(len(indexes) * perc_training)

        rnd = np.random.RandomState(1234)

        training_indexes = rnd.choice(indexes, size=n_training, replace=False)
        test_indexes = np.setdiff1d(np.arange(self.dataset.n_samples), training_indexes)

        # HISTOGRAM DATA
        training_samples = self.histogram_data[training_indexes]
        test_samples = self.histogram_data[test_indexes]

        # DATASET DATA
        #training_samples = self.dataset.data[training_indexes]
        #test_samples = self.dataset.data[test_indexes]

        y_true = [1 if l == 0 else 0 for l in self.dataset.labels[test_indexes]]

        '''# HISTOGRAM DATA FULL UNSUPERVISED
        training_samples = self.histogram_data
        test_samples = self.histogram_data

        # DATASET DATA FULL UNSUPERVISED
        #training_samples = self.dataset.data
        #test_samples = self.dataset.data

        y_true = [1 if l==0 else 0 for l in self.dataset.labels]'''

        df = pd.DataFrame()

        for kernel in ['poly', 'linear', 'rbf']:
            for gamma in ['auto']:
                for nu in [0.1, 0.5, 0.9]:

                    svm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu).fit(training_samples)

                    y_pred_labels = svm.predict(test_samples)
                    y_pred_labels = [0 if l == -1 else 1 for l in y_pred_labels]

                    y_pred_score = svm.score_samples(test_samples)

                    print(f'########### {kernel}-{gamma}-{nu} ###########')

                    precision, roc_auc = evaluate_results(y_true=y_true,
                                                          y_pred_score=y_pred_score,
                                                          y_pred_labels=y_pred_labels)

                    df = df.append(pd.DataFrame([[self.dataset.dataset_name, kernel, gamma, nu, roc_auc]],
                                                columns=['dataset', 'kernel', 'gamma', 'nu', 'roc_auc']))
                    print('dataset: ', self.dataset.dataset_name, '    roc_auc: ', roc_auc)

                    '''
                    #draw dataset (if synthetic dataset and if data=dataset.data
                    dataset = copy.deepcopy(self.dataset)
                    dataset.data = test_samples
                    dataset.n_samples = len(test_samples)
                    dataset.labels = np.asarray([1 if l == 0 else 0 for l in y_true])
                    dataset.show_data()
                    dataset.labels = np.asarray([1 if l == 0 else 0 for l in y_pred_labels])
                    dataset.show_data()'''

        return df

    def LOF(self):

        lof = LocalOutlierFactor(n_neighbors=10)         # n_neighbors = 10 from IFOR paper
        y_pred_labels = lof.fit_predict(self.histogram_data)
        y_pred_labels = [1 if l == -1 else 0 for l in y_pred_labels]
        y_pred_score = -lof.negative_outlier_factor_

        df = pd.DataFrame()

        precision, roc_auc = evaluate_results(y_true=self.dataset.labels,
                                              y_pred_score=y_pred_score,
                                              y_pred_labels=y_pred_labels)

        df = df.append(pd.DataFrame([[self.dataset.dataset_name, precision, roc_auc]],
                                    columns=['dataset', 'precision', 'roc_auc']))

        return df

    def IFOR_prediction(self):
        perc_training = 0.8
        indexes = np.argwhere(self.dataset.labels == 0).reshape(-1, )

        n_training = floor(len(indexes) * perc_training)

        rnd = np.random.RandomState(1234)

        training_indexes = rnd.choice(indexes, size=n_training, replace=False)
        test_indexes = np.setdiff1d(np.arange(self.dataset.n_samples), training_indexes)

        training_samples = self.histogram_data[training_indexes]
        test_samples = self.histogram_data[test_indexes]

        ifor = IForest(n_estimators=100, max_samples=256)
        ifor.fit(training_samples)
        depths = ifor.profile(test_samples)

        y_pred_score = []
        for i in range(len(depths)):
            depths_i = depths[i]  # output of i-th test instance

            avg = np.mean(depths_i)  # avg path length
            score = pow(2, -avg / INode.c(self.MAX_SAMPLES))
            y_pred_score.append(score)

        y_pred_labels = [1 if y >= 0.5 else 0 for y in y_pred_score]

        precision, roc_auc = evaluate_results(y_true=self.dataset.labels[test_indexes],
                                              y_pred_score=y_pred_score,
                                              y_pred_labels=y_pred_labels)

        df = pd.DataFrame([[self.dataset.dataset_name, precision, roc_auc]],
                          columns=['dataset', 'precision', 'roc_auc'])

        return df

    def KMeans_prediction(self):

        kmeans = KMeans(n_clusters=3, init='random', n_init=100)
        kmeans.fit(X=self.histogram_data)

        for i in range(3):
            print(i, (i+1)%3)
            y_pred_labels = np.asarray([0 if l == i or l == (i+1)%3 else 1 for l in kmeans.labels_])

            print('#anom:', len(y_pred_labels[y_pred_labels == 1]), '#norm:', len(y_pred_labels[y_pred_labels == 0]))
            #if len(y_pred_labels[y_pred_labels == 1]) > len(y_pred_labels[y_pred_labels == 0]):  # if #anom > #normal, swap
            #    print('swap')
             #   y_pred_labels = [1 if l == 0 else 0 for l in kmeans.labels_]

            precision, _ = evaluate_results(y_true=self.dataset.labels,
                                            y_pred_score=None,
                                            y_pred_labels=y_pred_labels)

            y_pred_labels = [1 if l == 0 else 0 for l in y_pred_labels]

            precision, _ = evaluate_results(y_true=self.dataset.labels,
                                            y_pred_score=None,
                                            y_pred_labels=y_pred_labels)

        df = pd.DataFrame([[self.dataset.dataset_name, precision]],
                          columns=['dataset', 'precision'])

        return df

    def plot(self, parameters, error=None):
        degree, n_variables = parameters.shape
        degree -= 1

        X, Y, Z = simplex_hyperplane_points()

        # real label
        anomaly_data = self.histogram_data[self.dataset.labels == 1]
        normal_data = self.histogram_data[self.dataset.labels == 0]

        # IFOR label
        #scores = self.get_anomaly_scores()
        #anomaly_data = self.histogram_data[scores >= 0.5]
        #normal_data = self.histogram_data[scores < 0.5]

        if n_variables == 2:  # 2D plot
            # draw approximation
            xdata_anomaly, ydata_anomaly = zip(*anomaly_data)
            plt.scatter(xdata_anomaly, ydata_anomaly)

            xdata_normal, ydata_normal = zip(*normal_data)
            plt.scatter(xdata_normal, ydata_normal)

            fit_points = parametric_equation(parameters)
            x_test, y_test = zip(*fit_points)

            plt.plot(x_test, y_test, 'r-')

            plt.title(self.dataset.dataset_name)
            if error is not None:
                plt.legend(['fit error: %f' % error])

            return plt

        elif n_variables == 3:  # 3D plot

            H1, H2, H3 = perfect_anomalies_hyperplane_points(parameters[1])

            # draw the simplex surface
            fig = go.Figure(data=[go.Mesh3d(x=X,
                                            y=Y,
                                            z=Z,
                                            opacity=0.5,
                                            color='rgba(244,22,100,0.6)',
                                            name='h₁ + h₂ + h₃ = 1',
                                            showlegend=True
                                            )])

            # draw the perfect anomaly surface
            fig.add_mesh3d(x=H1,
                           y=H2,
                           z=H3,
                           opacity=0.5,
                           color='rgba(23,31,255,0.6)',
                           name='h₁ + 2h₂ + 3h₃ = 0',
                           showlegend=True
                           )

            # draw data
            xdata_anomaly, ydata_anomaly, zdata_anomaly = zip(*anomaly_data)
            fig.add_scatter3d(x=xdata_anomaly,
                              y=ydata_anomaly,
                              z=zdata_anomaly,
                              mode='markers',
                              marker=dict(size=3, color='red'),
                              connectgaps=False,
                              name="anomaly")

            xdata_normal, ydata_normal, zdata_normal = zip(*normal_data)
            fig.add_scatter3d(x=xdata_normal,
                              y=ydata_normal,
                              z=zdata_normal,
                              mode='markers',
                              marker=dict(size=3, color='green'),
                              connectgaps=False,
                              name="normal")

            fit_points = parametric_equation(parameters)
            x_test, y_test, z_test = zip(*fit_points)

            fig.add_scatter3d(x=x_test, y=y_test, z=z_test, mode='lines', name='plane normal')

            if error is not None:
                fig.add_annotation(text='Error = %f' % error,
                                   font=dict(family="Arial"),
                                   xref="paper", yref="paper",
                                   x=1., y=0.97, showarrow=False,
                                   borderwidth=2,
                                   borderpad=4,
                                   bgcolor="#ff7f0e",
                                   opacity=0.8
                                   )

            fig.update_layout(
                scene=dict(
                    xaxis=dict(nticks=4),  # , range=[0, 1], ),
                    xaxis_title="h1",
                    yaxis=dict(nticks=4),  # , range=[0, 1], ),
                    yaxis_title="h2",
                    zaxis=dict(nticks=4),  # , range=[0, 1], ),
                    zaxis_title="h3",
                    annotations=[
                        dict(
                            x=1,
                            y=0,
                            z=0,
                            ax=50,
                            ay=20,
                            text="most anomalous point",
                            arrowhead=1,
                            xanchor="left",
                            yanchor="bottom"
                        ),
                        dict(
                            x=0,
                            y=0,
                            z=1,
                            ax=50,
                            ay=-10,
                            text="most normal point",
                            arrowhead=1,
                            xanchor="left",
                            yanchor="bottom"
                        )
                    ]
                ),
                margin=dict(r=20, l=10, b=10, t=10),
                title={
                    'text': self.dataset.dataset_name,
                    'x': 0.5,
                    'y': 0.95,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                legend={
                    'x': 0.82,
                    'y': 0.87,
                    'xanchor': 'right',
                    'yanchor': 'top'}
            )

            return fig

        else:   # n_variables > 3
            figures = []
            STEP_DIMENSION = 2  # 5

            #for j in range(4, 9):  # n_variables-STEP_DIMENSION):
            for j in range(2, n_variables-STEP_DIMENSION):

                x_normal, y_normal, z_normal = [], [], []
                x_anomaly, y_anomaly, z_anomaly = [], [], []

                for h in anomaly_data:
                    a, b, c = np.split(h, [j, j + STEP_DIMENSION])

                    x_anomaly.append(np.sum(a))
                    y_anomaly.append(np.sum(b))
                    z_anomaly.append(np.sum(c))

                for h in normal_data:
                    a, b, c = np.split(h, [j, j + STEP_DIMENSION])

                    x_normal.append(np.sum(a))
                    y_normal.append(np.sum(b))
                    z_normal.append(np.sum(c))


                # draw the simplex surface
                fig = go.Figure(data=[go.Mesh3d(x=X,
                                                y=Y,
                                                z=Z,
                                                opacity=0.5,
                                                color='rgba(244,22,100,0.6)',
                                                name='h₁ + h₂ + h₃ = 1',
                                                showlegend=True
                                                )])

                a, b, c = np.split(parameters[1], [j, j + STEP_DIMENSION])
                par = [np.sum(a), np.sum(b), np.sum(c)]

                H1, H2, H3 = perfect_anomalies_hyperplane_points(par)
                
                # draw the perfect anomaly surface
                fig.add_mesh3d(x=H1,
                               y=H2,
                               z=H3,
                               opacity=0.5,
                               color='rgba(23,31,255,0.6)',
                               name=str(par[0])[:5] + '*h₁ + ' + str(par[1])[:5] + '*h₂ + ' + str(par[2])[:5] + '*h₃ = 0',
                               showlegend=True
                               )


                # draw data
                fig.add_scatter3d(x=x_anomaly,
                                  y=y_anomaly,
                                  z=z_anomaly,
                                  mode='markers',
                                  marker=dict(size=3, color='red'),
                                  connectgaps=False,
                                  name="anomaly")

                fig.add_scatter3d(x=x_normal,
                                  y=y_normal,
                                  z=z_normal,
                                  mode='markers',
                                  marker=dict(size=3, color='green'),
                                  connectgaps=False,
                                  name="normal")

                fit_points = parametric_equation(parameters)

                x_test, y_test, z_test = [], [], []

                for p in fit_points:
                    a, b, c = np.split(p, [j, j + STEP_DIMENSION])
                    a = np.sum(a)
                    b = np.sum(b)
                    c = np.sum(c)

                    if -2 <= a <= 2 and -2 <= b <= 2 and -2 <= c <= 2:
                        x_test.append(a)
                        y_test.append(b)
                        z_test.append(c)

                fig.add_scatter3d(x=x_test, y=y_test, z=z_test, mode='lines', name='hyperplane normal')
                '''
                fpr, tpr, _ = roc_curve(self.dataset.labels, self.get_anomaly_scores())
                roc_auc = auc(fpr, tpr)

                fig.add_annotation(text='ROC curve (AUC = %0.2f)' % roc_auc,
                                   font=dict(family="Arial"),
                                   xref="paper", yref="paper",
                                   x=1., y=0.97, showarrow=False,
                                   borderwidth=2,
                                   borderpad=4,
                                   bgcolor="#ff7f0e",
                                   opacity=0.8
                                   )
                '''

                fig.update_layout(
                    scene=dict(
                        xaxis=dict(nticks=4),  # range=[-2, 2], ),
                        xaxis_title="h1=[0..." + str(j-1) + "]",
                        yaxis=dict(nticks=4),  # range=[-2, 2], ),
                        yaxis_title="h2=[" + str(j) + "..." + str(j+STEP_DIMENSION-1) + "]",
                        zaxis=dict(nticks=4),  # range=[-2, 2], ),
                        zaxis_title="h3=[" + str(j+STEP_DIMENSION) + "..." + str(n_variables) + "]",
                        annotations=[
                            dict(
                                x=1,
                                y=0,
                                z=0,
                                ax=50,
                                ay=20,
                                text="most anomalous point",
                                arrowhead=1,
                                xanchor="left",
                                yanchor="bottom"
                            ),
                            dict(
                                x=0,
                                y=0,
                                z=1,
                                ax=50,
                                ay=-10,
                                text="most normal point",
                                arrowhead=1,
                                xanchor="left",
                                yanchor="bottom"
                            )
                        ]
                    ),
                    margin=dict(r=20, l=10, b=10, t=10),
                    title={
                        'text': self.dataset.dataset_name,
                        'x': 0.5,
                        'y': 0.95,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    legend={
                        'x': 0.82,
                        'y': 0.87,
                        'xanchor': 'right',
                        'yanchor': 'top'}
                )

                figures.append(fig)

            return figures

    def roc_auc_along_hyperplane(self, indexes, parameters=None, show_distance_plot=False):    # indexes: indexes of test set

        if parameters is not None:
            self.parameters = parameters
        degree, n_variables = self.parameters.shape
        degree -= 1

        data_label = zip(self.histogram_data[indexes], self.dataset.labels[indexes], self.get_anomaly_scores()[indexes])

        distances_dict = []

        t1 = -1
        t2 = 1

        # extract two points on the Line
        A = self.parameters[0] + self.parameters[1] * t1
        B = self.parameters[0] + self.parameters[1] * t2

        ba = B - A  # distance vector that define the direction of the Line

        for dat, l, score in data_label:
            P = dat
            pa = P - A  # distance vector from the point A on the Line to the point P

            t = np.dot(pa, ba) / np.dot(ba, ba)
            d = np.linalg.norm(pa - t * ba, ord=2)

            distances_dict.append({'t_opt': t, 'p_opt': t * ba, 'dist_opt': d,
                                   'versor_opt': [], 'label': l, 'score': score})

        '''else:         # else of if degree==1 (before t1=-1)

            t_line = np.arange(-2, 2, 0.05)

            for dat, l, score in data_label:
                dist_opt = np.inf
                p_opt = 0
                best_t = {0: np.inf, 1: np.inf, 2: np.inf}
                for t in t_line:
                    p = [0] * n_variables
                    for d in range(degree + 1):
                        p += self.parameters[d] * (t ** d)
                    dist = np.linalg.norm(dat - p)
                    if dist < max(best_t.values()):
                        max_key = max(best_t, key=best_t.get)
                        del best_t[max_key]
                        best_t[t] = dist

                t_min = min(best_t.keys())
                t_max = max(best_t.keys())
                t_line_restrict = np.arange(t_min, t_max, low_interval/500)

                for t in t_line_restrict:
                    p = [0] * n_variables
                    for d in range(degree + 1):
                        p += self.parameters[d] * (t ** d)
                    dist = np.linalg.norm(dat - p)
                    if dist < dist_opt:
                        dist_opt = dist
                        p_opt = p
                        t_opt = t
                # p_opt is my fitting for dat

                dist_versor = dat - p_opt
                distances_dict[tuple(dat)] = {'t_opt': t_opt, 'p_opt': p_opt, 'dist_opt': dist_opt,
                                              'versor_opt': dist_versor, 'label': l, 'score': score}'''

        colors = ['red' if v['label'] == 1 else 'green' for v in distances_dict]

        distances = [v['dist_opt'] for v in distances_dict]
        labels_distance_order = [v['label'] for v in distances_dict]
        #labels_distance_order = self.dataset.labels[indexes]
        ts = [v['t_opt'] for v in distances_dict]

        # COMPUTE ROC AUC of IFOR
        fpr_IFOR, tpr_IFOR, _ = roc_curve(self.dataset.labels[indexes], self.get_anomaly_scores()[indexes])
        roc_auc_IFOR = auc(fpr_IFOR, tpr_IFOR)

        # COMPUTE ROC AUC of the EMBEDDING
        fpr_embedding, tpr_embedding, _ = roc_curve(labels_distance_order, ts)  # (ts - min(ts)) / (max(ts) - min(ts)))
        roc_auc_embedding = auc(fpr_embedding, tpr_embedding)
        if roc_auc_embedding < 0.5:
            fpr_embedding, tpr_embedding, _ = roc_curve(labels_distance_order, [-t for t in ts])  # 1 - ((ts - min(ts)) / (max(ts) - min(ts))))
            roc_auc_embedding = auc(fpr_embedding, tpr_embedding)

        if show_distance_plot:

            low_interval = 1e-3

            unique, unique_counts = np.unique(ts, return_counts=True)
            doubles = unique[unique_counts > 1]
            for t in doubles:
                index = np.where(ts == t)[0]
                for i in range(1, len(index)):
                    ts[index[i]] += (i * (low_interval / len(index)))

            #avg_width = (max(ts) - min(ts)) / 700
            '''if degree == 1:
                avg_width *= 3
            else:
                avg_width /= 2'''

            avg_width = 7e-4

            fig = go.Figure([go.Bar(x=(ts-min(ts)) / (max(ts)-min(ts)),
                                    y=distances,
                                    marker_color=colors,
                                    width=[avg_width] * len(ts))
                             ])

            fig.add_annotation(text=f'ROC AUC IFOR= {roc_auc_IFOR:.4f}',
                               font=dict(family="Arial"),
                               xref="paper", yref="paper",
                               x=1., y=0.97, showarrow=False,
                               borderwidth=2,
                               borderpad=4,
                               bgcolor="#ff7f0e",
                               opacity=0.8
                               )
            fig.add_annotation(text=f'ROC AUC embedding= {roc_auc_embedding:.4f}',
                               font=dict(family="Arial"),
                               xref="paper", yref="paper",
                               x=1., y=0.91, showarrow=False,
                               borderwidth=2,
                               borderpad=4,
                               bgcolor="#ff5f0e",
                               opacity=0.8
                               )

            fig.update_layout(title={
                'text': self.dataset.dataset_name + " - fit_polynomial of degree " + str(degree),
                'x': 0.435,
                'y': 0.98,
                'xanchor': 'center',
                'yanchor': 'top'}
            )

            fig.show()

        '''# PLOT DISTANCES USING ANOMALY SCORE ON X-AXIS

        scores = [v['score'] for v in distances_dict.values()]

        fig = go.Figure([go.Bar(x=(scores-min(scores)) / (max(scores)-min(scores)),
                                y=distances,
                                marker_color=colors,
                                width=[avg_width] * len(ts))
                         ])

        fpr_IFOR, tpr_IFOR, _ = roc_curve(self.dataset.labels, self.get_anomaly_scores())
        roc_auc_IFOR = auc(fpr_IFOR, tpr_IFOR)

        fpr_embedding, tpr_embedding, _ = roc_curve(labels_distance_order, (ts - min(ts)) / (max(ts) - min(ts)))
        roc_auc_embedding = auc(fpr_embedding, tpr_embedding)
        if roc_auc_embedding < 0.5:
            fpr_embedding, tpr_embedding, _ = roc_curve(labels_distance_order,
                                                        1 - ((ts - min(ts)) / (max(ts) - min(ts))))
            roc_auc_embedding = auc(fpr_embedding, tpr_embedding)

        fig.add_annotation(text=f'ROC AUC IFOR= {roc_auc_IFOR:.4f}',
                           font=dict(family="Arial"),
                           xref="paper", yref="paper",
                           x=1., y=0.97, showarrow=False,
                           borderwidth=2,
                           borderpad=4,
                           bgcolor="#ff7f0e",
                           opacity=0.8
                           )
        fig.add_annotation(text=f'ROC AUC embedding= {roc_auc_embedding:.4f}',
                           font=dict(family="Arial"),
                           xref="paper", yref="paper",
                           x=1., y=0.91, showarrow=False,
                           borderwidth=2,
                           borderpad=4,
                           bgcolor="#ff5f0e",
                           opacity=0.8
                           )

        fig.update_layout(title={
            'text': self.dataset.dataset_name + " - fit_polynomial of degree " + str(degree),
            'x': 0.435,
            'y': 0.98,
            'xanchor': 'center',
            'yanchor': 'top'}
        )

        fig.show()'''

        '''fig = self.plot(parameters=self.parameters)

        if n_variables == 3:
            for k, v in distances_dict.items():
                x_line, y_line, z_line = zip(*[k, v['p_opt']])
                fig.add_trace(
                    go.Scatter3d(x=x_line, y=y_line, z=z_line, mode='lines', marker=dict(size=2, color='red'),
                                 name='t = ' + str(round(v['t_opt'], ndigits=3))))

        if fig is not None:
            fig.show()'''

        return roc_auc_IFOR, roc_auc_embedding

    def distance_matrices_analysis(self):

        # compute distance matrix of ORIGINAL data
        original_data_distance_matrix = distance_matrix(self.dataset.data)
        original_data_distance_matrix /= original_data_distance_matrix.max()

        # compute distance matrix of point on the SIMPLEX
        simplex_distance_matrix = distance_matrix(self.histogram_data)
        simplex_distance_matrix /= simplex_distance_matrix.max()

        # compute (MANHATTAN) distance between depth arrays
        depth_arrays_distance_matrix = distance_matrix(self.depths, norm_ord=1)
        depth_arrays_distance_matrix /= depth_arrays_distance_matrix.max()

        # compute division (original data / simplex data) element-wise
        division1 = np.divide(original_data_distance_matrix, simplex_distance_matrix)
        division1 = np.nan_to_num(division1)
        division1[division1 > 1] = 1

        # compute division (depth arrays / simplex data) element-wise
        division2 = np.divide(depth_arrays_distance_matrix, simplex_distance_matrix)
        division2 = np.nan_to_num(division2)
        division2[division2 > 3] = 3

        fig = plt.figure(figsize=(20, 14))
        ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
        ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
        ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
        ax4 = plt.subplot2grid((2, 6), (1, 1), colspan=2)
        ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=2)
        fig.suptitle('Distance Matrices ' + self.dataset.dataset_name, fontsize=14)

        # Set titles for the figure and the subplot respectively
        ax1.set_title('Original Data')
        sp1 = ax1.imshow(original_data_distance_matrix, cmap='viridis')
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(sp1, cax=cax1)

        ax2.set_title('Depths Arrays')
        sp2 = ax2.imshow(depth_arrays_distance_matrix, cmap='viridis')
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(sp2, cax=cax2)

        ax3.set_title('Simplex')
        sp3 = ax3.imshow(simplex_distance_matrix, cmap='viridis')
        divider = make_axes_locatable(ax3)
        cax3 = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(sp3, cax=cax3)

        ax4.set_title('Original/Simplex Division')
        sp4 = ax4.imshow(division1, cmap='viridis')
        divider = make_axes_locatable(ax4)
        cax4 = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(sp4, cax=cax4)

        ax5.set_title('Depths/Simplex Division')
        sp5 = ax5.imshow(division2, cmap='viridis')
        divider = make_axes_locatable(ax5)
        cax5 = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(sp5, cax=cax5)

        plt.show()

    def depths_array_analysis(self):
        anomaly_score = self.get_anomaly_scores()

        if self.histogram_data is None:
            self.trees_heights_as_histogram()
        min_distances_norm = {10: (0, 0), 11: (0, 0), 12: (0, 0)}
        min_distances_anom = {10: (0, 0), 11: (0, 0), 12: (0, 0)}
        min_distances_x = {10: (0, 0), 11: (0, 0), 12: (0, 0)}

        for i in range(1, self.dataset.n_samples):
            for j in range(i):
                # dist = np.linalg.norm(self.histogram_data[i] - self.histogram_data[j], ord=2)    # euclidean distance
                dist = abs(anomaly_score[i] - anomaly_score[j])        # anomaly score distance
                if self.dataset.labels[i] == 0 and self.dataset.labels[j] == 0:
                    if dist < max(min_distances_norm.keys()) and dist not in min_distances_norm:
                        max_key = max(min_distances_norm.keys())
                        del min_distances_norm[max_key]
                        min_distances_norm[dist] = (i, j)
                elif self.dataset.labels[i] == 1 and self.dataset.labels[j] == 1:
                    if dist < max(min_distances_anom.keys()) and dist not in min_distances_anom:
                        max_key = max(min_distances_anom.keys())
                        del min_distances_anom[max_key]
                        min_distances_anom[dist] = (i, j)
                else:
                    if dist < max(min_distances_x.keys()) and dist not in min_distances_x:
                        max_key = max(min_distances_x.keys())
                        del min_distances_x[max_key]
                        min_distances_x[dist] = (i, j)

        min_distances = {}
        min_distances.update(min_distances_norm)
        min_distances.update(min_distances_anom)
        min_distances.update(min_distances_x)

        max_depth = ceil(np.max(self.depths))

        for k, v in min_distances.items():
            if isinstance(self.dataset, SyntheticDataset):
                self.dataset.show_data(v)

            sub = abs(self.depths[v[0]]-self.depths[v[1]])
            print(v, k)
            print("#zeros: ", sum(sub == 0))
            #print(sub[sub != 0])
            print("non-zeros mean: ", np.mean(sub[sub != 0]))
            print("L1 distance: ", np.linalg.norm(sub, ord=1), "\n")

            vmin = min(min(self.depths[v[0]]), min(self.depths[v[1]]), 4)
            vmax = max(max(self.depths[v[0]]), max(self.depths[v[1]]), 16)

            fig = plt.figure(figsize=(12, 8))
            gs = fig.add_gridspec(nrows=20, ncols=3)
            ax1 = fig.add_subplot(gs[1:13, 0])
            ax2 = fig.add_subplot(gs[1:13, 1])
            ax3 = fig.add_subplot(gs[1:13, 2])
            ax4 = fig.add_subplot(gs[14:, 0])
            ax5 = fig.add_subplot(gs[14:, 1])
            fig.suptitle('Distance Matrices ' + self.dataset.dataset_name, fontsize=14)

            # --------------- COLUMN 1 ------------------

            # Set titles for the figure and the subplot respectively
            title = 'Real label: '
            title += 'NORMAL\n\n' if self.dataset.labels[v[0]] == 0 else 'ANOMALY\n\n'
            title += 'anomaly score: ' + str(anomaly_score[v[0]])[:8] + '\n'
            ax1.set_title(title)
            depths = np.repeat([self.depths[v[0]]], 10, axis=0)
            sp1 = ax1.imshow(depths.T, cmap='viridis')
            divider = make_axes_locatable(ax1)
            cax1 = divider.append_axes("right", size="20%", pad=0.2)
            sp1.set_clim(vmin, vmax)
            ax1.axes.xaxis.set_visible(False)
            ax1.axes.yaxis.set_visible(False)
            plt.colorbar(sp1, cax=cax1)

            # histogram
            if self.dataset.labels[v[0]] == 1:
                ax4.hist(self.depths[v[0]], range(1, max_depth + 1), color='#ff7f0e')
            else:
                ax4.hist(self.depths[v[0]], range(1, max_depth + 1), color='#1f77b4')
            #ax4.set_ylim((0, max(self.depths[v[0]].max(), self.depths[v[1]].max())))

            # --------------- COLUMN 2 ------------------

            title = 'Real label: '
            title += 'NORMAL\n\n' if self.dataset.labels[v[1]] == 0 else 'ANOMALY\n\n'
            title += 'anomaly score: ' + str(anomaly_score[v[1]])[:8] + '\n'
            ax2.set_title(title)
            depths = np.repeat([self.depths[v[1]]], 10, axis=0)
            sp2 = ax2.imshow(depths.T, cmap='viridis')
            divider = make_axes_locatable(ax2)
            cax2 = divider.append_axes("right", size="20%", pad=0.2)
            sp2.set_clim(vmin, vmax)
            ax2.axes.xaxis.set_visible(False)
            ax2.axes.yaxis.set_visible(False)
            plt.colorbar(sp2, cax=cax2)

            # histogram
            if self.dataset.labels[v[1]] == 1:
                ax5.hist(self.depths[v[1]], range(1, max_depth + 1), color='#ff7f0e')
            else:
                ax5.hist(self.depths[v[1]], range(1, max_depth + 1), color='#1f77b4')
            #ax5.set_ylim((0, max(self.depths[v[0]].max(), self.depths[v[1]].max())))
            ax4.get_shared_y_axes().join(ax4, ax5)

            # --------------- COLUMN 3 ------------------

            ax3.set_title('|Instance 1 - Instance 2|\n\n')
            depths = np.repeat([sub], 10, axis=0)
            sp3 = ax3.imshow(depths.T, cmap='viridis')
            divider = make_axes_locatable(ax3)
            cax3 = divider.append_axes("right", size="20%", pad=0.2)
            #sp3.set_clim(vmin, vmax)
            ax3.axes.xaxis.set_visible(False)
            ax3.axes.yaxis.set_visible(False)
            plt.colorbar(sp3, cax=cax3)

            #fig.suptitle('Depths Array ' + self.dataset.dataset_name, fontsize=14)
            plt.show()
