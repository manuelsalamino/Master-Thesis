import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import floor
from sklearn.datasets import make_blobs, make_circles, make_moons
from abc import ABC

from utils_functions import make_line


class AbstractDataset(ABC):
    dataset_name: str
    labels: np.ndarray
    data: np.ndarray
    attributes: list
    n_samples: int
    n_anomalies: int

    def get_anomaly_instances(self):
        return self.data[self.labels == 1]

    def get_normal_instances(self):
        return self.data[self.labels == 0]

    def order_dataset(self):
        self.data = np.concatenate([self.get_normal_instances(), self.get_anomaly_instances()])
        self.labels.sort()


class RealDataset(AbstractDataset):

    def __init__(self, csv_path, train_test_split=None):
        # save dataset_name
        dataset_name = os.path.split(csv_path)[1]         # extract tail (filename) of the path
        if dataset_name[-4:] == '.csv':               # remove extension of the file if present
            dataset_name = dataset_name[:-4]

        self.dataset_name = dataset_name

        # save data and attributes
        data = pd.read_csv(csv_path)

        # extract the attributes name (remove the 'id' and the label 'y')
        attributes = data.columns.tolist()
        attributes.remove("y")
        attributes.remove("id")
        self.attributes = attributes

        self.labels = np.asarray(data['y'], dtype=int)
        self.data = data.loc[:, self.attributes].to_numpy()

        super().order_dataset()

        self.n_samples = self.data.shape[0]
        self.n_anomalies = len(self.labels[self.labels == 1])


class SyntheticDataset(AbstractDataset):
    dataset_type: str

    def __init__(self, n_samples=300, anomalies_rate=0.1, dataset_type="blobs", show_data=False, **kwargs):

        self.n_samples = n_samples
        self.n_anomalies = floor(n_samples * anomalies_rate)

        self.dataset_type = dataset_type

        # generate normal instances
        x, y = self.generate_dataset(**kwargs)
        inliers = list(zip(x, y, np.zeros(shape=(len(x),))))          # x, y and label (0=normal)

        # generate anomaly instances
        rng = np.random.RandomState(128)
        x, y = zip(*rng.uniform(low=-4, high=4, size=(self.n_anomalies, 2)))
        outliers = list(zip(x, y, np.ones(shape=(len(x),))))          # x, y and label (1=anomaly)

        data = np.concatenate([inliers, outliers])
        data = pd.DataFrame(data, columns=['x1', 'x2', 'y'])

        self.labels = np.asarray(data['y'], dtype=int)
        self.attributes = ['x1', 'x2']
        self.data = data.loc[:, self.attributes].to_numpy()

        super().order_dataset()

        if show_data:
            self.show_data()

    def show_data(self, indexes=None):
        colors = np.array(['#1f77b4' if self.labels[i] == 0 else '#ff7f0e' for i in range(self.n_samples)])
        if indexes is not None:
            colors[indexes[0]] = 'red'
            colors[indexes[1]] = 'red'

        normal_points_x1, normal_points_x2 = zip(*self.data[self.labels == 0])
        anomaly_points_x1, anomaly_points_x2 = zip(*self.data[self.labels == 1])
        #color_normal = ['#1f77b4'] * (self.n_samples - self.n_anomalies)
        #color_anomaly = ['#ff7f0e'] * self.n_anomalies
        color_normal = colors[self.labels == 0]
        color_anomaly = colors[self.labels == 1]
        plt.scatter(normal_points_x1, normal_points_x2, c=color_normal)
        plt.scatter(anomaly_points_x1, anomaly_points_x2, c=color_anomaly)
        plt.axis('equal')
        plt.title(self.dataset_name)
        plt.xlabel('x1')
        plt.ylabel('x2')

        if indexes is not None:
            plt.annotate('instance 1', xy=self.data[indexes[0]], xycoords='data',
                         xytext=(150 if self.data[indexes[0]][0] < 0 else -150, 20 if self.data[indexes[0]][1] < 0 else -20), textcoords='offset points',
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                         horizontalalignment='right', verticalalignment='top',
                         )
            plt.annotate('instance 2', xy=self.data[indexes[1]], xycoords='data',
                         xytext=(150 if self.data[indexes[1]][0] < 0 else -150, -20 if self.data[indexes[1]][1] < 0 else 20), textcoords='offset points',
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                         horizontalalignment='right', verticalalignment='top',
                         )
        plt.show()

    def generate_dataset(self, **kwargs):
        if self.dataset_type == "blobs":
            self.dataset_name = f"Synthetic - {kwargs['n_blobs']} blob{'s' if kwargs['n_blobs']>1 else ''}"
            centers = [[0, 0], [-3, -1]]
            x, y = zip(*make_blobs(n_samples=self.n_samples-self.n_anomalies, centers=centers[:kwargs['n_blobs']],
                                   cluster_std=0.5, n_features=2, random_state=22)[0])

        elif self.dataset_type == "circles":
            self.dataset_name = f"Synthetic - {'Noisy ' if kwargs['noise']>0 else ''}Circles"
            x, y = zip(*make_circles(n_samples=self.n_samples-self.n_anomalies, noise=kwargs['noise'], factor=0.5,
                                     random_state=0)[0])

        elif self.dataset_type == "moons":
            self.dataset_name = f"Synthetic - {'Noisy ' if kwargs['noise']>0 else ''}Moons"
            x, y = zip(*make_moons(n_samples=self.n_samples-self.n_anomalies, noise=kwargs['noise'], random_state=0)[0])

        elif self.dataset_type == "line":
            self.dataset_name = "Synthetic - Line"
            x, y = make_line(n_samples=self.n_samples-self.n_anomalies, x_inf=-4, x_sup=4, random_state=20)

        return x, y
