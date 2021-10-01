import numpy as np
from math import ceil, floor
import constraint
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

from ITree.INode import INode
from utils_functions import anomaly_score


def simplex_hyperplane_points():
    problem = constraint.Problem()

    problem.addVariable('x', np.linspace(0, 1, 10))
    problem.addVariable('y', np.linspace(0, 1, 10))
    problem.addVariable('z', np.linspace(0, 1, 10))

    def our_constraint(x, y, z):
        if x + y + z == 1:
            return True

    problem.addConstraint(our_constraint, ['x', 'y', 'z'])

    solutions = problem.getSolutions()

    X = np.asarray([list(sol.values())[0] for sol in solutions])
    Y = np.asarray([list(sol.values())[1] for sol in solutions])
    Z = np.asarray([list(sol.values())[2] for sol in solutions])

    return X, Y, Z


def perfect_anomalies_hyperplane_points(parameters):
    problem = constraint.Problem()

    problem.addVariable('h1', np.arange(-2, 2, 0.05))
    problem.addVariable('h2', np.arange(-2, 2, 0.05))
    problem.addVariable('h3', np.arange(-2, 2, 0.05))

    def our_constraint(x, y, z):
        a = parameters[0]
        b = parameters[1]
        c = parameters[2]

        if -0.05 <= a*x + b*y + c*z <= 0.05: #  and -1.5 <= x <= 1.5 and -1.5 <= y <= 1.5 and -1.5 <= z <= 1.5:
            return True

    problem.addConstraint(our_constraint, ['h1', 'h2', 'h3'])

    solutions = problem.getSolutions()

    X = np.asarray([list(sol.values())[0] for sol in solutions])
    Y = np.asarray([list(sol.values())[1] for sol in solutions])
    Z = np.asarray([list(sol.values())[2] for sol in solutions])

    return X, Y, Z


def anomaly_score_hyperplane_points(b, c):
    problem = constraint.Problem()

    problem.addVariable('x', np.linspace(0, 1, 50))
    problem.addVariable('y', np.linspace(0, 1, 50))
    problem.addVariable('z', np.linspace(0, 1, 50))

    def our_constraint(x, y, z):
        epsilon = 0.02
        if c-epsilon <= b.dot([x, y, z]) <= c+epsilon:
            return True

    problem.addConstraint(our_constraint, ['x', 'y', 'z'])

    solutions = problem.getSolutions()

    X = np.asarray([list(sol.values())[0] for sol in solutions])
    Y = np.asarray([list(sol.values())[1] for sol in solutions])
    Z = np.asarray([list(sol.values())[2] for sol in solutions])

    return X, Y, Z


def anomaly_score_hyperplane(c):
    X, Y, Z = anomaly_score_hyperplane_points(b=np.arange(1, 4), c=c)

    #simplex_points = np.asarray(list(zip(X, Y, Z)))
    #avg = np.asarray([1, 2, 3]).dot(simplex_points.T)
    #score = anomaly_score(avg)

    # draw the simplex surface
    fig = go.Figure(data=[go.Mesh3d(x=X,
                                    y=Y,
                                    z=Z,
                                    opacity=0.5
                                    )])

    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[0, 1], ),
            xaxis_title="h1",
            yaxis=dict(nticks=4, range=[0, 1], ),
            yaxis_title="h2",
            zaxis=dict(nticks=4, range=[0, 1], ),
            zaxis_title="h3",
        ),
        width=800,
        margin=dict(r=20, l=10, b=10, t=10))

    fig.show()


def simplex_and_anomaly_score_hyperplanes():
    X, Y, Z = simplex_hyperplane_points()

    simplex_points = np.asarray(list(zip(X, Y, Z)))
    avg = np.asarray([1, 2, 3]).dot(simplex_points.T)
    score = anomaly_score(avg)

    # draw the simplex surface
    fig = go.Figure(data=[go.Mesh3d(x=X,
                                    y=Y,
                                    z=Z,
                                    opacity=0.5
                                    )])

    for i in range(3, 7):
        avg = np.asarray([1, 2, 3]).dot(simplex_points.T)
        PSI = i
        score = anomaly_score(avg)

        X_b, Y_b, Z_b = anomaly_score_hyperplane_points(b=np.arange(1, 4), c=INode.c(PSI))

        fig.add_trace(go.Mesh3d(x=X_b,
                                      y=Y_b,
                                      z=Z_b,
                                      opacity=0.5
                                      ))

        threshold_indexes = np.intersect1d(np.where(score > 0.498), np.where(score < 0.502))

        X_th, Y_th, Z_th = zip(*np.asarray(list(zip(X, Y, Z)))[threshold_indexes.astype(int)])

        fig.add_trace(go.Scatter3d(x=X_th, y=Y_th, z=Z_th, mode='lines', marker=dict(size=2), name=('PSI=' + i + ')')))

    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[0, 1], ),
            xaxis_title="h1",
            yaxis=dict(nticks=4, range=[0, 1], ),
            yaxis_title="h2",
            zaxis=dict(nticks=4, range=[0, 1], ),
            zaxis_title="h3",
        ),
        width=800,
        margin=dict(r=20, l=10, b=10, t=10))

    fig.show()


def draw_point_on_simplex(depths, labels, dataset_name):
    X, Y, Z = simplex_hyperplane_points()

    STEP_DIMENSION = 5

    for j in range(4, 9): # ceil(np.amax(depths))-STEP_DIMENSION):
        # draw the simplex surface
        fig = go.Figure(data=[go.Mesh3d(x=X,
                                        y=Y,
                                        z=Z,
                                        opacity=0.5,
                                        color='rgba(244,22,100,0.6)'
                                        )])

        x_normal, y_normal, z_normal = [], [], []
        x_anomaly, y_anomaly, z_anomaly = [], [], []
        scores = []

        for i in range(len(depths)):
            #depths_int = np.asarray([int(round(d, 0)) for d in depths[i]])
            depths_i = np.asarray(depths[i])
            #depths_int[np.where(depths_int == 4)[0]] = 3     # depth=4 -> depth=3

            h = np.histogram(depths_i, range(1, ceil(max(depths_i))+1))         # compute histogram values

            h = h[0] / len(depths_i)
            # print(depths_int)
            # print(h)

            avg = np.mean(depths_i)               # avg path length
            score = anomaly_score(avg)
            scores.append(score)

            a, b, c = np.split(h, [j, j+STEP_DIMENSION])

            #if score >= 0.5:              # split in anomaly and normal using the computed score
            if labels[i]:                    # split in anomaly and normal using label
                x_anomaly.append(np.sum(a))
                y_anomaly.append(np.sum(b))
                z_anomaly.append(np.sum(c))
            else:
                x_normal.append(np.sum(a))
                y_normal.append(np.sum(b))
                z_normal.append(np.sum(c))

        fpr, tpr, _ = roc_curve(labels, scores)
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

        # draw normal and anomaly points
        fig.add_scatter3d(x=x_anomaly, y=y_anomaly, z=z_anomaly, mode='markers', connectgaps=False, name="anomaly")
        fig.add_scatter3d(x=x_normal, y=y_normal, z=z_normal, mode='markers', connectgaps=False, name="normal")

        fig.update_layout(
            scene=dict(
                xaxis=dict(nticks=4, range=[0, 1], ),
                xaxis_title="h1=[0,...," + str(j-1) + "]",
                yaxis=dict(nticks=4, range=[0, 1], ),
                yaxis_title="h2=[" + str(j) + ",...," + str(j+STEP_DIMENSION-1) + "]",
                zaxis=dict(nticks=4, range=[0, 1], ),
                zaxis_title="h3=[" + str(j+STEP_DIMENSION) + ",...," + str(floor(np.amax(depths))) + "]",
            ),
            width=900,
            margin=dict(r=20, l=10, b=10, t=10),
            title={
                'text': dataset_name,
                'x': 0.435,
                'y': 0.98,
                'xanchor': 'center',
                'yanchor': 'top'}
        )

        fig.show()


def heatmap_on_simplex(threshold=False):
    X, Y, Z = simplex_hyperplane_points()

    simplex_points = np.asarray(list(zip(X, Y, Z)))
    avg = np.asarray([1, 2, 3]).dot(simplex_points.T)
    score = pow(2, -avg / INode.c(5))

    # draw the simplex surface
    fig = go.Figure(data=[go.Mesh3d(x=X,
                                    y=Y,
                                    z=Z,
                                    opacity=0.5,
                                    intensity=score,
                                    colorscale='Rainbow',
                                    cmin=0.3,      # if change this, change value in point_color
                                    cmid=0.5,
                                    cmax=0.8      # if change this, change value in point_color
                                    )])

    if threshold:
        threshold_indexes = np.intersect1d(np.where(score > 0.498), np.where(score < 0.502))

        X_th, Y_th, Z_th = zip(*np.asarray(list(zip(X, Y, Z)))[threshold_indexes.astype(int)])

        fig.add_trace(go.Scatter3d(x=X_th, y=Y_th, z=Z_th, mode='lines', marker=dict(size=2)))

    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[0, 1], ),
            xaxis_title="h1",
            yaxis=dict(nticks=4, range=[0, 1], ),
            yaxis_title="h2",
            zaxis=dict(nticks=4, range=[0, 1], ),
            zaxis_title="h3",
        ),
        width=800,
        margin=dict(r=20, l=10, b=10, t=10))

    return fig