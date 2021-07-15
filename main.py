from Dataset import RealDataset, SyntheticDataset
from ExtendedIForest import ExtendedIForest
from ITree.INode import INode
from simplex_functions import simplex_hyperplane_points, anomaly_score
from utils_functions import get_data_centroid

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


'''def OLD_fit_points(depths, labels, show_plot=False):
    points, scores = [], []

    max_depth = ceil(np.max(depths))
    for i in range(len(depths)):
        depths_i = depths[i]
        h = np.histogram(depths_i, range(1, max_depth + 1))           # compute histogram values
        h = h[0] / len(depths_i)

        points.append(h)

        avg = np.mean(depths_i)  # avg path length
        score = anomaly_score(avg, n_samples=MAX_SAMPLES)
        scores.append(score)

    points = Points(points)

    # fit points
    line_fit = Line.best_fit(points)

    # order points in the line
    ordered_points = {}
    for i in range(len(points)):
        p = tuple(points[i])
        ordered_points[p] = (line_fit.transform_points(p), labels[i])
    ordered_points = {k: v for k, v in sorted(ordered_points.items(), key=lambda item: item[1][0])}

    _, labs = zip(*ordered_points.values())

    ordered_distance_points = {}
    for k, v in ordered_points.items():
        dist = line_fit.distance_point(k)
        projected_point = line_fit.project_point(k)
        distance_vector = Line.from_points(projected_point, k)

        ordered_distance_points[k] = (dist, distance_vector)

    colors = ['red' if l==1 else 'green' for l in labs]

    distances, _ = zip(*ordered_distance_points.values())

    fig = go.Figure([go.Bar(#x=np.arange(len(ordered_distance_points.keys())),
                            x=ordered_distance_points.keys(),
                            y=distances,
                            marker_color=colors)
                    ])

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig.add_annotation(text='ROC curve AUC = %0.2f' % roc_auc,
                       font=dict(family="Arial"),
                       xref="paper", yref="paper",
                       x=1., y=0.97, showarrow=False,
                       borderwidth=2,
                       borderpad=4,
                       bgcolor="#ff7f0e",
                       opacity=0.8
                       )

    fig.update_layout(title={
            'text': dataset_name,
            'x': 0.435,
            'y': 0.98,
            'xanchor': 'center',
            'yanchor': 'top'}
    )

    fig.show()

    if show_plot:
        X, Y, Z = simplex_hyperplane_points()

        # draw the simplex surface
        fig = go.Figure(data=[go.Mesh3d(x=X,
                                        y=Y,
                                        z=Z,
                                        opacity=0.5,
                                        color='rgba(244,22,100,0.6)'
                                        )])

        x_points, y_points, z_points = zip(*points)

        # draw points on simplex
        fig.add_trace(
            go.Scatter3d(x=x_points, y=y_points, z=z_points, mode='markers', marker=dict(size=3), name='points'))

        # compute fit_line
        t_line = np.arange(-1, 1, 0.001)          # set of values of t in order to draw the line

        x_line, y_line, z_line = [], [], []
        for t in t_line:
            point = line_fit.to_point(t=t)
            if point[0] > 0 and point[1] > 0 and point[2] > 0:       # take only points of fit_line inside the simplex
                x_line.append(point[0])
                y_line.append(point[1])
                z_line.append(point[2])

        # draw fit_line
        fig.add_trace(go.Scatter3d(x=x_line, y=y_line, z=z_line, mode='lines', marker=dict(size=3), name='fit'))

        # draw distance lines
        for k, v in ordered_distance_points.items():
            x_line, y_line, z_line = [], [], []
            line = v[1]
            for t in np.arange(0, 1, 0.1):
                point = line.to_point(t=t)
                x_line.append(point[0])
                y_line.append(point[1])
                z_line.append(point[2])
            fig.add_trace(go.Scatter3d(x=x_line, y=y_line, z=z_line, mode='lines', marker=dict(size=2, color='green'), name='distance vector'))

        fig.update_layout(
            scene=dict(
                xaxis=dict(nticks=4, range=[0, 1], ),
                xaxis_title="h1",
                yaxis=dict(nticks=4, range=[0, 1], ),
                yaxis_title="h2",
                zaxis=dict(nticks=4, range=[0, 1], ),
                zaxis_title="h3",
            ),
            #width=900,
            margin=dict(r=20, l=10, b=10, t=10),
            title={
                'text': dataset_name,
                'x': 0.435,
                'y': 0.98,
                'xanchor': 'center',
                'yanchor': 'top'}
        )

        fig.show()

    return line_fit'''


def extract_histogram(depths):
    data, scores = [], []

    max_depth = ceil(np.max(depths))
    for i in range(len(depths)):
        depths_i = depths[i]
        h = np.histogram(depths_i, range(1, max_depth + 1))  # compute histogram values
        h = h[0] / len(depths_i)

        data.append(h)

        avg = np.mean(depths_i)  # avg path length
        score = anomaly_score(avg, n_samples=MAX_SAMPLES)
        scores.append(score)

    return np.asarray(data), np.asarray(scores)


def plot_distances(data, labels, scores, param):
    degree, n_variables = param.shape
    degree -= 1

    data_label = zip(data, labels)

    low_interval = 1e-3

    t_line = np.arange(-3, 3, 0.1)
    distances_dict = {}

    for dat, l in data_label:
        dist_opt = np.inf
        p_opt = 0
        best_t = {0: np.inf, 1: np.inf, 2: np.inf}
        for t in t_line:
            p = [0] * n_variables
            for d in range(degree + 1):
                p += param[d] * (t ** d)
            dist = np.linalg.norm(dat - p)
            if dist < max(best_t.values()):
                max_key = max(best_t, key=best_t.get)
                del best_t[max_key]
                best_t[t] = dist

        t_min = min(best_t.keys())
        t_max = max(best_t.keys())
        t_line_restrict = np.arange(t_min, t_max, low_interval)

        for t in t_line_restrict:
            p = [0] * n_variables
            for d in range(degree + 1):
                p += param[d] * (t ** d)
            dist = np.linalg.norm(dat - p)
            if dist < dist_opt:
                dist_opt = dist
                p_opt = p
                t_opt = t
        # p_opt is my fitting for dat

        dist_versor = dat - p_opt
        distances_dict[tuple(dat)] = {'t_opt': t_opt, 'p_opt': p_opt, 'dist_opt': dist_opt, 'versor_opt': dist_versor, 'label': l}

    colors = ['red' if v['label'] == 1 else 'green' for v in distances_dict.values()]

    distances = [v['dist_opt'] for v in distances_dict.values()]
    ts = [v['t_opt'] for v in distances_dict.values()]

    unique, unique_counts = np.unique(ts, return_counts=True)
    doubles = unique[unique_counts > 1]
    for t in doubles:
        index = np.where(ts == t)[0]
        for i in range(1, len(index)):
            ts[index[i]] += (i*(low_interval/len(index)))

    avg_width = (max(ts) - min(ts)) / len(ts)

    fig = go.Figure([go.Bar(x=ts,
                            y=distances,
                            marker_color=colors,
                            width=[avg_width/3]*len(ts))
                     ])

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig.add_annotation(text='ROC curve AUC = %0.2f' % roc_auc,
                       font=dict(family="Arial"),
                       xref="paper", yref="paper",
                       x=1., y=0.97, showarrow=False,
                       borderwidth=2,
                       borderpad=4,
                       bgcolor="#ff7f0e",
                       opacity=0.8
                       )

    fig.update_layout(title={
        'text': dataset_name,
        'x': 0.435,
        'y': 0.98,
        'xanchor': 'center',
        'yanchor': 'top'}
    )

    fig.show()

    fig = plot(data, labels=labels, parameters=param)

    if n_variables == 3:
        for k, v in distances_dict.items():
            x_line, y_line, z_line = zip(*[k, v['p_opt']])
            fig.add_trace(go.Scatter3d(x=x_line, y=y_line, z=z_line, mode='lines', marker=dict(size=2, color='red'),
                                   name='t = ' + str(round(v['t_opt'], ndigits=3))))

    if fig is not None:
        fig.show()


def plot(data, labels, parameters, error=None):
    degree, n_variables = parameters.shape
    degree -= 1

    anomaly_data = data[[True if l==1 else False for l in labels]]
    normal_data = data[np.logical_not(labels)]

    if n_variables == 2:  # 2D plot
        # draw approximation
        xdata_anomaly, ydata_anomaly = zip(*anomaly_data)
        plt.scatter(xdata_anomaly, ydata_anomaly)

        xdata_normal, ydata_normal = zip(*normal_data)
        plt.scatter(xdata_normal, ydata_normal)

        fit_point = parametric_equation(parameters)
        x_test, y_test = zip(*fit_point)

        plt.plot(x_test, y_test, 'r-')

        plt.title(dataset_name)
        if error is not None:
            plt.legend(['fit error: %f' % error])

        return plt

    elif n_variables == 3:  # 3D plot
        # draw approximation
        xdata_anomaly, ydata_anomaly, zdata_anomaly = zip(*anomaly_data)
        fig = go.Figure(data=[go.Scatter3d(x=xdata_anomaly,
                                           y=ydata_anomaly,
                                           z=zdata_anomaly,
                                           mode='markers',
                                           marker=dict(size=3, color='red'),
                                           connectgaps=False,
                                           name="anomaly")])

        xdata_normal, ydata_normal, zdata_normal = zip(*normal_data)
        fig.add_scatter3d(x=xdata_normal,
                          y=ydata_normal,
                          z=zdata_normal,
                          mode='markers',
                          marker=dict(size=3, color='green'),
                          connectgaps=False,
                          name="normal")

        fit_point = parametric_equation(parameters)
        x_test, y_test, z_test = zip(*fit_point)

        fig.add_scatter3d(x=x_test, y=y_test, z=z_test, mode='lines', name='fit')

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
                xaxis=dict(nticks=4, range=[0, 1], ),
                xaxis_title="h1",
                yaxis=dict(nticks=4, range=[0, 1], ),
                yaxis_title="h2",
                zaxis=dict(nticks=4, range=[0, 1], ),
                zaxis_title="h3",
            ),
            margin=dict(r=20, l=10, b=10, t=10),
            width=900,
            title={
                'text': dataset_name,
                'x': 0.435,
                'y': 0.98,
                'xanchor': 'center',
                'yanchor': 'top'}
        )

        return fig


def parametric_equation(param):
    degree, n_variables = param.shape
    degree -= 1

    t_line = np.arange(-5, 5, 0.01)
    points = []
    points_backup = []

    for t in t_line:
        p = [0] * (n_variables)
        for d in range(degree + 1):
            p += param[d] * (t ** d)
        points_backup.append(p)
        if all(0 <= i <= 1 for i in p):
            points.append(p)

    if len(points) == 0:
        return points_backup  # if approximation is outside the hypercube 1x1x...x1 return all the points
    return points


def fit_points_ODR(data, degree, show_plot=False):

    def estimate_least_squares(x, data):
        n_variables = data.shape[1]

        if n_variables == 1:
            data = [d[0] for d in data]
            data = np.asarray(data)

        degree = int(len(x) / (n_variables)) - 1

        coeff = np.asarray(x).copy().reshape(degree + 1, n_variables)

        t_line = np.arange(-1, 1, 0.1)
        estimate_data = []
        #start = time.perf_counter()

        for dat in data:
            dist_opt = np.inf
            p_opt = 0
            best_t = {0: np.inf, 1: np.inf, 2: np.inf}
            for t in t_line:
                p = [0] * n_variables
                for d in range(degree+1):
                    p += coeff[d] * (t**d)
                dist = np.linalg.norm(dat - p)
                if dist < max(best_t.values()):
                    max_key = max(best_t, key=best_t.get)
                    del best_t[max_key]
                    best_t[t] = dist

            t_min = min(best_t.keys())
            t_max = max(best_t.keys())
            t_line_restrict = np.arange(t_min, t_max, 0.01)

            for t in t_line_restrict:
                p = [0] * (n_variables)
                for d in range(degree+1):
                    p += coeff[d] * (t**d)
                dist = np.linalg.norm(dat - p)
                if dist < dist_opt:
                    dist_opt = dist
                    p_opt = p
            # p_opt is my fitting for dat

            estimate_data.append(p_opt)

        #print(f'data: {dat}, p_opt: {p_opt},\ttarget: {ydata[-1]}')
        #points = [np.append(xdata[i], ydata[i]) for i in range(len(xdata))]
        #print("time: ", time.perf_counter() - start)
        distances = [np.linalg.norm(np.subtract(estimate_data[i], data[i]), ord=2) for i in range(len(data))]
        error = np.linalg.norm(distances, 2) ** 2
        print(f'error: {error}')
        return distances

    def estimate(x, xdata):
        xdata = xdata.T
        n_variables = np.asarray(xdata).shape[1] + 1
        degree = int(len(x) / n_variables) - 1

        coeff = np.asarray(x).copy().reshape(degree + 1, n_variables)

        t_line = np.arange(-5, 5, 0.1)
        estimate_data = []

        for dat in xdata:
            dist_opt = np.inf
            p_opt = 0
            best_t = {0: np.inf, 1: np.inf, 2: np.inf, 3: np.inf, 4: np.inf}
            for t in t_line:
                p = [0] * n_variables
                for d in range(degree+1):
                    p += coeff[d] * (t**d)
                dist = np.linalg.norm(dat - p[:-1])
                if dist < max(best_t.values()):
                    max_key = max(best_t, key=best_t.get)
                    del best_t[max_key]
                    best_t[t] = dist

            t_min = min(best_t.keys())
            t_max = max(best_t.keys())
            t_line_restrict = np.arange(t_min, t_max, 0.01)

            for t in t_line_restrict:
                p = [0] * n_variables
                for d in range(degree+1):
                    p += coeff[d] * (t**d)
                dist = np.linalg.norm(dat - p[:-1])
                if dist < dist_opt:
                    dist_opt = dist
                    p_opt = p
            # p_opt is my fitting for dat

            estimate_data.append(p_opt[-2:])

        #print(f'data: {dat}, p_opt: {p_opt},\ttarget: {ydata[-1]}')
        #points = [np.append(xdata[i], ydata[i]) for i in range(len(xdata))]
        #distances = [np.linalg.norm(np.subtract(estimate_data[i], points[i]), ord=2) for i in range(len(points))]
        #error = np.linalg.norm(distances, 2) ** 2
        #print(f'error: {error}')
        est_y1, est_y2 = zip(*estimate_data)
        return np.asarray([est_y1, est_y2])

    n_variables = data.shape[1]

    # fit points
    n_restarts = 1
    best_error = np.inf

    func = Model(estimate)
    #func = polynomial(order=2)
    #xdata1, xdata2, ydata = zip(*data)
    #xdata = np.asarray([xdata1, xdata2])
    #ydata = np.asarray([xdata2, ydata])
    #mydata = RealData(xdata, ydata)

    for _ in range(n_restarts):
        # fit using Ordinary Least Square
        opt = least_squares(estimate_least_squares, x0=np.random.rand((degree + 1) * n_variables), ftol=1e-03,
                            args=([data]))
        error = np.linalg.norm(opt.fun, ord=2)  # / data.shape[0]
        popt = np.asarray(opt.x).reshape(degree + 1, n_variables)
        print(popt)

        # visualize plots (if possible: 2D or 3D)
        fig = plot(data=data, labels=labels, parameters=popt, error=error)
        if fig is not None:
            fig.show()

        # fit using ODR
        #odr = ODR(mydata, func, beta0=np.random.rand((degree+1)*n_variables))
        #out = odr.run()
        #error_odr = out.sum_square
        #popt_odr = np.asarray(out.beta).reshape(degree + 1, n_variables)
        #print(popt_odr)

        # visualize plots (if possible: 2D or 3D)
        #fig = plot(data=data, labels=labels, parameters=popt_odr, error=error_odr)
        #fig.show()

        # fit data using ODR --> initial parameters are the ones found with ordinary least square
        #myodr = ODR(mydata, func, beta0=opt.x)
        #myoutput = myodr.run()
        #popt = np.asarray(myoutput.beta).reshape(degree + 1, n_variables)
        #error = myoutput.sum_square
        #print(popt)

        # visualize plots (if possible: 2D or 3D)
        #fig = plot(data=data, labels=labels, parameters=popt, error=error)
        #fig.show()

        if error < best_error:
            best_popt = popt
            best_error = error

    return np.asarray(best_popt).reshape(degree + 1, n_variables)


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
files.remove('Http.csv')
files.remove('ForestCover.csv')
files.remove('Mulcross.csv')
files.remove('Shuttle.csv')
files.remove('Smtp.csv')

#results = pd.DataFrame()

#files = ['Breastw.csv']
files = ['Shuttle.csv', 'Smtp.csv']

for dataset_name in files:
    print(dataset_name)
    path = os.path.join(dir_path, dataset_name)
    dataset = RealDataset(path)

    ifor = ExtendedIForest(N_ESTIMATORS=100, MAX_SAMPLES=256, dataset=dataset)
    ifor.fit_IForest()
    ifor.profile_IForest()
    ifor.trees_heights_as_histogram()

    # parameters = np.array([[0] * ifor.histogram_data.shape[1], [i for i in range(1, ifor.histogram_data.shape[1] + 1)]])

    centroid = get_data_centroid(ifor.histogram_data)
    pca = PCA(n_components=1)
    pca.fit(ifor.histogram_data)
    parameters = np.concatenate([centroid, pca.components_[0]]).reshape(2, centroid.shape[0])

    '''# visualize plots
    fig = ifor.plot(parameters)
    if fig is not None:
        if isinstance(fig, list):
            for f in fig:
                f.show()
        else:
            fig.show()'''

    ifor.plot_distances(parameters=parameters)

    '''
    #for d in [1, 2, 3, 5]:
     #   polynomial_parameters = ifor.fit_histogram_points(degree=d)
      #  ifor.plot_distances(d)

    par = np.zeros(shape=(2, ifor.histogram_data.shape[1]))
    for i in range(par.shape[1]):
        if i == 0:
            #par[1][0] = INode.c(n=ifor.MAX_SAMPLES)
            par[1][0] = 0
        else:
            par[1][i] = i

    ifor.parameters = par
    ifor.plot_distances()
    '''
    #df = ifor.OC_Svm()
    #df = ifor.IFOR_prediction()
    #df = ifor.KMeans_prediction()

    #results = results.append(df)

#with pd.ExcelWriter('output.xlsx', mode='w') as writer:
  #  results.to_excel(writer, index=False)