from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_score
from sklearn.datasets import make_blobs, make_circles, make_moons

from ITree.INode import INode


def anomaly_score(avg, n_samples=256):
    score = pow(2, -avg / INode.c(n_samples))
    return score


def draw_roc_curve(depths, labels):
    scores = []
    for d in depths:
        avg = sum(d) / len(d)                    # avg path length
        scores.append(anomaly_score(avg))            # compute anomaly score for each sample of the test set

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def draw_histogram(depths):
    for i in range(10):     # compute and show the histogram only of the first 10 test sample
        depths_i = np.asarray(depths[i])

        plt.hist(depths_i, range(1, ceil(max(depths_i))+1))     # compute and draw histogram
        # plt.plot([0, max(depths_int)], [sum(depths[i])/len(depths[i]), sum(depths[i])/len(depths[i])], color='navy', linestyle='--')
        plt.xlabel('Depth')
        plt.ylabel('# of trees')

        avg = sum(depths_i) / len(depths_i)                 # avg path length
        score = anomaly_score(avg)
        label = 'ANOMALY' if score > 0.5 else 'NORMAL'
        title = 'histogram -> ' + label + ' (score: ' + str(score) + ')'
        plt.title(title)
        plt.show()


def OLD_create_synthetic_dataset(n_samples=1000, anomalies_rate=0.1, show_data=True):

    def generate(median, err, outlier_err, size, outlier_size):
        errs = err * np.random.rand(size) * np.random.choice((-1, 1), size)
        data = list(zip(median + errs, np.zeros(shape=errs.shape)))

        lower_errs = outlier_err * np.random.rand(outlier_size)
        lower_outliers = list(zip(median - err - lower_errs, np.ones(shape=errs.shape)))

        upper_errs = outlier_err * np.random.rand(outlier_size)
        upper_outliers = list(zip(median + err + upper_errs, np.ones(shape=errs.shape)))

        data = np.concatenate((data, lower_outliers, upper_outliers))

        return zip(*data)

    np.random.seed(1234)

    n_anomalies = floor(n_samples * anomalies_rate)

    x1, y1 = generate(median=10, err=4, outlier_err=4, size=n_samples-n_anomalies, outlier_size=n_anomalies)
    x2, y2 = generate(median=5, err=2, outlier_err=5, size=n_samples-n_anomalies, outlier_size=n_anomalies)

    x1_anom = np.asarray(x1[-n_anomalies*2:])
    np.random.shuffle(x1_anom)
    x1 = list(x1[:-n_anomalies*2]) + list(x1_anom)

    df = pd.DataFrame(data={'x1': x1, 'x2': x2, 'y': y1})

    if show_data:
        plt.figure(figsize=(14, 7))
        plt.scatter(x=df[df['y'] == 0]['x1'], y=df[df['y'] == 0]['x2'], label='y = 0')
        plt.scatter(x=df[df['y'] == 1]['x1'], y=df[df['y'] == 1]['x2'], label='y = 1')
        plt.title('Synthetic Dataset', fontsize=20)
        plt.legend()

        plt.show()

    # extract the attributes name (remove the label 'y')
    attributes = df.columns.tolist()
    attributes.remove("y")

    return df, attributes


def create_synthetic_dataset(n_samples=300, anomalies_rate=0.1, dataset_type="blobs", show_data=True, **kwargs):
    n_outliers = floor(n_samples * anomalies_rate)
    n_inliers = n_samples - n_outliers

    x, y = generate_dataset(dataset_type=dataset_type, n_samples=n_inliers, **kwargs)
    inliers = list(zip(x, y, np.zeros(shape=(len(x),))))          # x, y and label (0=normal)

    rng = np.random.RandomState(128)

    x, y = zip(*rng.uniform(low=-4, high=4, size=(n_outliers, 2)))
    outliers = list(zip(x, y, np.ones(shape=(len(x),))))            # x, y and label (1=anomaly)

    data = np.concatenate([inliers, outliers])

    data = pd.DataFrame(data, columns=['x1', 'x2', 'y'])
    attributes = ['x1', 'x2']

    if show_data:
        plt.scatter(data[data['y'] == 0]['x1'], data[data['y'] == 0]['x2'])
        plt.scatter(data[data['y'] == 1]['x1'], data[data['y'] == 1]['x2'])
        plt.axis('equal')
        plt.show()

    return data, attributes


def generate_dataset(dataset_type, n_samples, **kwargs):
    if dataset_type == "blobs":
        centers = [[0, 0], [-3, -1]]
        x, y = zip(*make_blobs(n_samples=n_samples, centers=centers[:kwargs['n_blobs']],
                               cluster_std=0.5,  n_features=2, random_state=22)[0])
    elif dataset_type == "circles":
        x, y = zip(*make_circles(n_samples=n_samples, noise=kwargs['noise'], factor=0.5, random_state=0)[0])
    elif dataset_type == "moons":
        x, y = zip(*make_moons(n_samples=n_samples, noise=kwargs['noise'], random_state=0)[0])
    elif dataset_type == "line":
        x, y = make_line(n_samples=n_samples, x_inf=-4, x_sup=4, random_state=20)
    return x, y


def make_line(n_samples=100, x_inf=0, x_sup=1, random_state=None):
    x_line = np.arange(x_inf, x_sup, (x_sup-x_inf)/n_samples)

    rnd = np.random.RandomState(random_state)
    m, q = rnd.randint(-2, 2, size=2)

    y_line = m*x_line + q

    return x_line, y_line


def real_dataset(csv_path):
    data = pd.read_csv(csv_path)

    # extract the attributes name (remove the 'id' and the label 'y')
    attributes = data.columns.tolist()
    attributes.remove("y")
    attributes.remove("id")

    return data, attributes


def parametric_equation(param):
    degree, n_variables = param.shape
    degree -= 1

    t_line = np.arange(-20, 20, 0.01)
    points = []
    points_backup = []

    for t in t_line:
        p = [0] * (n_variables)
        for d in range(degree + 1):
            p += param[d] * (t ** d)
        points_backup.append(p)
        if all(-2 <= i <= 2 for i in p):
            points.append(p)

    if len(points) == 0:
        return points_backup  # if approximation is outside the hypercube 1x1x...x1 return all the points
    return points


def distance_matrix(data, norm_ord=2):
    data = np.asarray(data)
    n_instances = data.shape[0]
    distance_matrix = np.zeros(shape=(n_instances, n_instances))

    for i in range(n_instances):
        for j in range(i+1):
            sub = data[i] - data[j]
            dist = np.linalg.norm(sub, ord=norm_ord)
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist

    return distance_matrix


def get_data_centroid(data):
    data = np.asarray(data)
    centroid = np.zeros(shape=(data.shape[1],))

    for i in range(data.shape[1]):
        centroid[i] = np.mean(data[:, i])

    return centroid


def evaluate_results(y_true, y_pred_score=None, y_pred_labels=None):
    y_true = np.asarray(y_true)
    y_pred_labels = np.asarray(y_pred_labels)
    print('n_normal true:', len(y_true[y_true == 0]), 'n_normal pred:', len(y_pred_labels[y_pred_labels == 0]))
    print('n_anomaly true:', len(y_true[y_true == 1]), 'n_anomaly pred:', len(y_pred_labels[y_pred_labels == 1]))

    precision = precision_score(y_true, y_pred_labels)
    print('precision: ', precision)

    if y_pred_score is None:
        print('roc: None\n')
        return precision, None

    fpr, tpr, _ = roc_curve(y_true, y_pred_score)
    roc_auc = auc(fpr, tpr)
    print('roc: ', roc_auc, '\n')

    return precision, roc_auc