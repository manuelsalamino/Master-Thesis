from Dataset import RealDataset
from ExtendedIForest import ExtendedIForest
from utils_functions import parametric_equation
from simplex_functions import hyperplane_points, simplex_hyperplane_points, perfect_anomalies_hyperplane_points

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go


def dim2():
    fig, ax = plt.subplots()

    # simplex points (sum hi = 1)
    h1 = np.arange(0, 1.1, 0.1)
    h2 = 1 - h1

    ax.plot(h1, h2)

    # hyperplane points (E(x) = 0)           1*h1 + 2*h2 = 0
    h1 = np.arange(-2, 2.1, 0.1)
    h2 = -1/2 * h1

    ax.plot(h1, h2)

    # plot details
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

    ax.legend(['h₁ + h₂ = 1', 'h₁ + 2h₂ = 0'])

    ax.annotate(text='most anomalous point', xy=(1, 0), xycoords='data', xytext=(20, 40),
                 bbox=dict(boxstyle="round", fc="none", ec="gray"), textcoords='offset points',
                 ha='center', arrowprops=dict(arrowstyle="->"))
    ax.annotate(text='most normal point', xy=(0, 1), xycoords='data', xytext=(-10, 40),
                 bbox=dict(boxstyle="round", fc="none", ec="gray"), textcoords='offset points',
                 ha='center', arrowprops=dict(arrowstyle="->"))

    # Create 'x' and 'y' labels placed at the end of the axes
    ax.set_xlabel('h₂')
    ax.xaxis.set_label_position('top')

    ax.set_ylabel('      h₁', rotation='horizontal')
    ax.yaxis.set_label_position('right')


    # Draw arrows
    arrow_fmt = dict(markersize=4, color='black', clip_on=False)
    ax.plot((2), (0), marker='>', **arrow_fmt)
    ax.plot((0), (2), marker='^', **arrow_fmt)

    plt.show()


def parallelE():
    fig, ax = plt.subplots()

    # simplex points (sum hi = 1)
    h1 = np.arange(0, 1.1, 0.1)
    h2 = 1 - h1

    ax.plot(h1, h2)

    # hyperplane points (E(x) = 0)           h1 + 2h2 = 0
    h1 = np.arange(-2, 2.1, 0.1)
    h2 = -1 / 2 * h1

    ax.plot(h1, h2)

    # hyperplane points (E(x) = i)           h1 + 2h2 = i  ->  h2 = 1/2 * ( i - h1 )
    h1 = np.arange(-2, 2.1, 0.1)

    legend = ['h₁ + h₂ = 1', 'h₁ + 2h₂ = 0']
    colors = ['', 'r', 'g']
    for i in range(1, 3):
        h2 = 1/2 * (i - h1)

        legend.append('h₁ + 2h₂ = ' + str(i))
        ax.plot(h1, h2, color=colors[i], alpha=0.2)

    # plot details
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

    ax.legend(legend)

    ax.annotate(text='most anomalous point', xy=(1, 0), xycoords='data', xytext=(20, 40),
                bbox=dict(boxstyle="round", fc="none", ec="gray"), textcoords='offset points',
                ha='center', arrowprops=dict(arrowstyle="->"))
    ax.annotate(text='most normal point', xy=(0, 1), xycoords='data', xytext=(-10, 40),
                bbox=dict(boxstyle="round", fc="none", ec="gray"), textcoords='offset points',
                ha='center', arrowprops=dict(arrowstyle="->"))

    # Create 'x' and 'y' labels placed at the end of the axes
    ax.set_xlabel('h₂')
    ax.xaxis.set_label_position('top')

    ax.set_ylabel('      h₁', rotation='horizontal')
    ax.yaxis.set_label_position('right')

    # Draw arrows
    arrow_fmt = dict(markersize=4, color='black', clip_on=False)
    ax.plot((2), (0), marker='>', **arrow_fmt)
    ax.plot((0), (2), marker='^', **arrow_fmt)

    plt.show()


# MAX_SAMPLES = 5
def dim3():
    X, Y, Z = simplex_hyperplane_points()
    H1, H2, H3 = perfect_anomalies_hyperplane_points([1, 2, 3])

    # upload REAL dataset
    dir_path = os.path.join(os.getcwd(), '../datasets')
    #files = os.listdir(dir_path)
    # files.remove('Http.csv')
    # files.remove('ForestCover.csv')
    # files.remove('Mulcross.csv')
    # files.remove('Shuttle.csv')
    # files.remove('Smtp.csv')

    files = ['Breastw.csv']

    for dataset_name in files:
        print(dataset_name)
        path = os.path.join(dir_path, dataset_name)
        dataset = RealDataset(path)

        ifor = ExtendedIForest(N_ESTIMATORS=100, MAX_SAMPLES=5, dataset=dataset)
        ifor.fit_IForest()
        ifor.profile_IForest()
        ifor.trees_heights_as_histogram()

        # real label
        #anomaly_data = ifor.histogram_data[ifor.dataset.labels == 1]
        #normal_data = ifor.histogram_data[ifor.dataset.labels == 0]

        # IFOR label
        scores = ifor.get_anomaly_scores()
        anomaly_data = ifor.histogram_data[scores >= 0.5]
        normal_data = ifor.histogram_data[scores < 0.5]

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
                'text': dataset_name,
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

        fig.show()


# MAX_SAMPLES = 256
def dim3_aggregate():
    X, Y, Z = simplex_hyperplane_points()
    H1, H2, H3 = perfect_anomalies_hyperplane_points()

    # upload REAL dataset
    dir_path = os.path.join(os.getcwd(), '../datasets')
    # files = os.listdir(dir_path)
    # files.remove('Http.csv')
    # files.remove('ForestCover.csv')
    # files.remove('Mulcross.csv')
    # files.remove('Shuttle.csv')
    # files.remove('Smtp.csv')

    files = ['Breastw.csv']

    for dataset_name in files:
        print(dataset_name)
        path = os.path.join(dir_path, dataset_name)
        dataset = RealDataset(path)

        ifor = ExtendedIForest(N_ESTIMATORS=100, MAX_SAMPLES=256, dataset=dataset)
        ifor.fit_IForest()
        ifor.profile_IForest()
        ifor.trees_heights_as_histogram()
        # real label
        # anomaly_data = ifor.histogram_data[ifor.dataset.labels == 1]
        # normal_data = ifor.histogram_data[ifor.dataset.labels == 0]

        # IFOR label
        scores = ifor.get_anomaly_scores()
        anomaly_data = ifor.histogram_data[scores >= 0.5]
        normal_data = ifor.histogram_data[scores < 0.5]


        STEP_DIMENSION = 5

        for j in range(4, 9):  # n_variables-STEP_DIMENSION):

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

            parameters = np.array([[0]*anomaly_data.shape[1], [i for i in range(1, anomaly_data.shape[1]+1)]])
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

            fig.add_scatter3d(x=x_test, y=y_test, z=z_test, mode='lines', name='fit')
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
                    xaxis_title="h1",
                    yaxis=dict(nticks=4),  # range=[-2, 2], ),
                    yaxis_title="h2",
                    zaxis=dict(nticks=4),  # range=[-2, 2], ),
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
                    'text': dataset_name,
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

            fig.show()


def simplex():
    X, Y, Z = hyperplane_points()

    '''# draw the hyperplane surface
    fig = go.Figure(data=[go.Mesh3d(x=X,
                                    y=Y,
                                    z=Z,
                                    #opacity=0.5,
                                    #color='rgba(244,22,100,0.1)',
                                    color='rgba(112,172,255,0.3)',
                                    name='h₁ + h₂ + h₃ = 1',
                                    showlegend=True
                                    )])'''

    # draw simplex contours
    fig = go.Figure(data=[go.Scatter3d(x=[1, 0, 0, 1],
                                       y=[0, 1, 0, 0],
                                       z=[0, 0, 1, 0],
                                       mode='lines',
                                       #line=dict(dash='dash'),
                                       marker=dict(size=5, color='rgba(78,131,255,0.7)'),
                                       name="simplex contours",
                                       showlegend=False)])

    # draw simplex surface
    fig.add_mesh3d(x=[1, 0, 0, 1],
                   y=[0, 1, 0, 0],
                   z=[0, 0, 1, 0],
                   # opacity=0.5,
                   # color='rgba(244,22,100,0.1)',
                   color='rgba(78,131,255,0.7)',
                   name='simplex',
                   showlegend=True
                   )

    fig.update_layout(
        #scene_aspectmode='manual',
        #scene_aspectratio=dict(x=2, y=2, z=1.85),
        scene=dict(
            xaxis=dict(nticks=4),  # , range=[0, 1.5], ),
            xaxis_title="h1",
            yaxis=dict(nticks=4),  # range=[0, 1.5], ),
            yaxis_title="h2",
            zaxis=dict(nticks=4),  # range=[0, 1.5], ),
            zaxis_title="h3",
            annotations=[
                dict(
                    showarrow=False,
                    x=1,
                    y=0,
                    z=0,
                    text="(1,0,0)",
                    xanchor="left",
                    yanchor="bottom",
                    xshift=-30,
                    font=dict(
                        color="black",
                        size=18
                    )
                ),
                dict(
                    showarrow=False,
                    x=0,
                    y=1,
                    z=0,
                    text="(0,1,0)",
                    xanchor="left",
                    yanchor="bottom",
                    xshift=-30,
                    font=dict(
                        color="black",
                        size=18
                    )
                ),
                dict(
                    showarrow=False,
                    x=0,
                    y=0,
                    z=1,
                    text="(0,0,1)",
                    xanchor="left",
                    yanchor="bottom",
                    xshift=-30,
                    font=dict(
                        color="black",
                        size=18
                    )
                )
            ]
        ),
        margin=dict(r=20, l=10, b=10, t=10),
        title={
            'text': 'Simplex Representation',
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

    fig.show()


def plot_histogram():
    # upload REAL dataset
    dir_path = os.path.join(os.getcwd(), '../datasets')
    files = os.listdir(dir_path)

    files = ['Breastw.csv']

    for dataset_name in files:
        print(dataset_name)
        path = os.path.join(dir_path, dataset_name)
        dataset = RealDataset(path)

        ifor = ExtendedIForest(N_ESTIMATORS=100, MAX_SAMPLES=256, dataset=dataset)
        ifor.fit_IForest()
        ifor.profile_IForest()
        ifor.trees_heights_as_histogram(show_plot=True)


simplex()