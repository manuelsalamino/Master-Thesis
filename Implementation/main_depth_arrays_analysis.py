from Dataset import RealDataset, SyntheticDataset
from ExtendedIForest import ExtendedIForest
from utils_functions import *
from simplex_functions import *

import numpy as np
import os
from math import ceil, sqrt
import matplotlib.pyplot as plt
from plotly.colors import find_intermediate_color, convert_colors_to_same_type
#from skspatial.objects import Points, Plane

np.seterr(divide='ignore', invalid='ignore')


def point_color(fig, point_score):
    intermed = (point_score-0.3) / (0.8-0.3)         # 0.2 and 0.9 are cmin and cmax of the colormap
    colorscale = fig['data'][0]['colorscale']
    th, rgb = zip(*colorscale)
    index = np.argmax(th > intermed)

    lowcolor, _ = convert_colors_to_same_type(rgb[index-1], colortype="rgb")
    lowcolor = lowcolor[0]
    highcolor, _ = convert_colors_to_same_type(rgb[index], colortype="rgb")
    highcolor = highcolor[0]

    relative_intermed = (intermed-th[index-1]) / (th[index]-th[index-1])

    color = find_intermediate_color(lowcolor, highcolor, relative_intermed, colortype='rgb')
    color = color[4:-1].split(',')
    color = [float(c)/255. for c in color]
    color = tuple(color)

    return color


def colored_original_space(depths, testing_data):
    fig_heatmap = heatmap_on_simplex()
    fig_heatmap.show()

    scores = []
    colors = []
    for instance in depths:
        avg = np.mean(np.asarray(instance))
        score = anomaly_score(avg)
        colors.append(point_color(fig_heatmap, score))
        scores.append(score)

    #testing_data = pd.DataFrame(data=data.iloc[testing_indices])
    testing_data['color'] = colors

    for i in range(len(colors)):
        for j in range(len(colors)):
            if i != j and abs(colors[i][0] - colors[j][0]) < 0.001 and abs(colors[i][1] - colors[j][1]) < 0.001 and \
                    abs(colors[i][2] - colors[j][2]) < 0.001 and abs(scores[i] - scores[j]) < 0.001:
                print(i, j)

    # plot dataset with the new color
    #plt.figure(figsize=(16, 7))
    plt.scatter(x=testing_data['x1'], y=testing_data['x2'], c=testing_data['color'])
    #plt.axis([-6, 10, -6, 10])
    plt.axis('equal')

    plt.title('Synthetic with Score Color', fontsize=20)
    plt.legend()

    plt.show()

    # plot original dataset
    #plt.figure(figsize=(16, 7))
    plt.scatter(x=testing_data[testing_data['y'] == 0]['x1'], y=testing_data[testing_data['y'] == 0]['x2'],
                label='y = 0')
    plt.scatter(x=testing_data[testing_data['y'] == 1]['x1'], y=testing_data[testing_data['y'] == 1]['x2'],
                label='y = 1')
    #plt.axis([-6, 10, -6, 10])
    plt.axis('equal')
    plt.title('Synthetic Dataset', fontsize=20)
    plt.legend()

    plt.show()


def project_on_simplex(points_3d):
    # FONTE: https://math.stackexchange.com/questions/236540/2d-coordinates-of-projection-of-3d-vector-onto-2d-plane

    projection_matrix = [[0, -1 / sqrt(2), 1 / sqrt(2)],
                         [-1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)]]
    projection_matrix = np.asarray(projection_matrix)

    points_2d = np.matmul(points_3d, projection_matrix.T)

    return points_2d


def depth_arrays_distance(depths):
    depths = np.asarray(depths)
    n_instances = depths.shape[0]
    distance_matrix = np.zeros(shape=(n_instances, n_instances))

    for i in range(n_instances):
        for j in range(i + 1):
            sub = depths[i] - depths[j]
            dist = np.linalg.norm(sub, ord=1)
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist

    return distance_matrix


def OLD_point_distance(depths, labels, show_plot=False):
    scores = []
    points_3d = {'normal': [], 'anomaly': []}

    for i in range(len(depths)):
        depths_i = np.asarray(depths[i])

        h = np.histogram(depths_i, range(1, ceil(max(depths_i)) + 1))  # compute histogram values
        h = h[0] / len(depths_i)

        avg = np.mean(depths_i)  # avg path length
        score = anomaly_score(avg, n_samples=MAX_SAMPLES)
        scores.append(score)

        # if score >= 0.5:              # split in anomaly and normal using the computed score
        if labels[i]:  # split in anomaly and normal using label
            points_3d['anomaly'].append(h)
        else:
            points_3d['normal'].append(h)

    points_3d_anomaly = np.asarray(points_3d['anomaly'])
    points_3d_normal = np.asarray(points_3d['normal'])

    plane = Plane(point=(0, 0, 0), normal=(1, 1, 1))    # plane parallel to simplex but passing throw the origin
    triangle_points = Points([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])

    # project points in the plane (the plane passing throw the origin)
    #projections_anomaly = [list(map(float, plane.project_point(pa))) for pa in points_3d_anomaly]               # convert Point3D in list of float
    #projections_normal = [list(map(float, plane.projection(pn))) for pn in points_3d_normal]                # convert Point3D in list of float
    #projection_triangle = [list(map(float, plane.projection(pt))) for pt in triangle_points]                # convert Point3D in list of float

    projection_anomalies = [plane.project_point(pa) for pa in points_3d_anomaly]
    projection_normals = [plane.project_point(pn) for pn in points_3d_normal]
    projection_triangle = [plane.project_point(pt) for pt in triangle_points]

    projection_anomalies = np.asarray(projection_anomalies)
    projection_normals = np.asarray(projection_normals)
    projection_triangle = np.asarray(projection_triangle)

    points_2d_anomaly = project_on_simplex(projection_anomalies)
    points_2d_normal = project_on_simplex(projection_normals)
    triangle_2d = project_on_simplex(projection_triangle)

    points2d = list(points_2d_normal)
    points2d.extend(points_2d_anomaly)
    normal_indices = range(len(points_2d_normal))

    n_points = len(points2d)
    distance_matrix = np.zeros(shape=(n_points, n_points))

    for i in range(n_points):
        for j in range(n_points):
            sub = points2d[i] - points2d[j]
            dist = np.linalg.norm(sub, ord=2)
            distance_matrix[i][j] = dist

    plt.imshow(distance_matrix, cmap='viridis')
    plt.colorbar()
    plt.show()

    x_anomaly, y_anomaly = zip(*points_2d_anomaly)
    x_normal, y_normal = zip(*points_2d_normal)
    x_triangle, y_triangle = zip(*triangle_2d)

    plt.scatter(x_anomaly, y_anomaly, s=7)
    plt.scatter(x_normal, y_normal, s=7)
    plt.plot(x_triangle, y_triangle)
    plt.show()

    for i in range(len(projection_normals)):
        for j in range(i+1, len(projection_normals)):
            a = projection_normals[i] - projection_normals[j]
            dista = np.linalg.norm(a, ord=2)
            b = points_2d_normal[i] - points_2d_normal[j]
            distb = np.linalg.norm(b, ord=2)

            #if abs(dista - distb) != 0:
                #print(f"i: {i}, j: {j}\ni-3d: {projections_normal[i]}, j-3d: {projections_normal[j]}\ni-2d: {points_2d_normal[i]}, j-2d: {points_2d_normal[j]}")
            print(f"|dist3D - dist2D| = {abs(dista-distb)}")

    if show_plot:
        X, Y, Z = simplex_hyperplane_points()

        fig = go.Figure(data=[go.Mesh3d(x=X,
                                        y=Y,
                                        z=Z,
                                        opacity=0.5,
                                        color='rgba(244,22,100,0.6)'
                                        )])

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
        x_anomaly, y_anomaly, z_anomaly = zip(*points_3d['anomaly'])
        x_normal, y_normal, z_normal = zip(*points_3d['normal'])

        fig.add_scatter3d(x=x_anomaly, y=y_anomaly, z=z_anomaly, mode='markers', marker=dict(size=4), connectgaps=False,
                          name="anomaly")
        fig.add_scatter3d(x=x_normal, y=y_normal, z=z_normal, mode='markers', marker=dict(size=4), connectgaps=False,
                          name="normal")

        fig.update_layout(
            scene=dict(
                xaxis=dict(nticks=4, range=[0, 1], ),
                xaxis_title="h1",
                yaxis=dict(nticks=4, range=[0, 1], ),
                yaxis_title="h2",
                zaxis=dict(nticks=4, range=[0, 1], ),
                zaxis_title="h3",
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


'''# create SYNTHETIC dataset
datasets = ["Synthetic - One Blob", "Synthetic - Two Blobs", "Synthetic - Circles", "Synthetic - Noisy Circles",
            "Synthetic - Moons", "Synthetic - Noisy Moons", "Synthetic - Line"]

for dataset_name in datasets:
    if dataset_name == "Synthetic - One Blob":
        dataset, attributes = create_synthetic_dataset(n_samples=200, anomalies_rate=0.1, dataset_type='blobs',
                                                       n_blobs=1, show_data=True)
    elif dataset_name == "Synthetic - Two Blobs":
        dataset, attributes = create_synthetic_dataset(n_samples=200, anomalies_rate=0.1, dataset_type='blobs',
                                                       n_blobs=2, show_data=True)
    elif dataset_name == "Synthetic - Circles":
        dataset, attributes = create_synthetic_dataset(n_samples=200, anomalies_rate=0.1, dataset_type='circles',
                                                       noise=0, show_data=True)
    elif dataset_name == "Synthetic - Noisy Circles":
        dataset, attributes = create_synthetic_dataset(n_samples=200, anomalies_rate=0.1, dataset_type='circles',
                                                       noise=0.05, show_data=True)
    elif dataset_name == "Synthetic - Moons":
        dataset, attributes = create_synthetic_dataset(n_samples=200, anomalies_rate=0.1, dataset_type='moons',
                                                       noise=0, show_data=True)
    elif dataset_name == "Synthetic - Noisy Moons":
        dataset, attributes = create_synthetic_dataset(n_samples=200, anomalies_rate=0.1, dataset_type='moons',
                                                       noise=0.1, show_data=True)
    elif dataset_name == "Synthetic - Line":
        dataset, attributes = create_synthetic_dataset(n_samples=200, anomalies_rate=0.1, dataset_type='line',
                                                       show_data=True)
'''

# create SYNTHETIC dataset
datasets = []
datasets.append(SyntheticDataset(n_samples=300, anomalies_rate=0.1, dataset_type="blobs", n_blobs=1))
#datasets.append(SyntheticDataset(n_samples=300, anomalies_rate=0.1, dataset_type="blobs" n_blobs=2))
datasets.append(SyntheticDataset(n_samples=300, anomalies_rate=0.1, dataset_type="circles", noise=0))
#datasets.append(SyntheticDataset(n_samples=300, anomalies_rate=0.1, dataset_type="circles" noise=0.05))
#datasets.append(SyntheticDataset(n_samples=300, anomalies_rate=0.1, dataset_type="moons" noise=0))
#datasets.append(SyntheticDataset(n_samples=300, anomalies_rate=0.1, dataset_type="moons" noise=0.1))
#datasets.append(SyntheticDataset(n_samples=300, anomalies_rate=0.1, dataset_type="line"))

for dataset in datasets:

    ifor = ExtendedIForest(N_ESTIMATORS=100, MAX_SAMPLES=256, dataset=dataset)
    ifor.fit_IForest()
    ifor.profile_IForest()
    ifor.trees_heights_as_histogram()
    #for d in [1, 2, 3, 5, 10, 20, 30]:
        #polynomial_parameters = ifor.fit_histogram_points(degree=d)
        #ifor.plot_distances(d)
    #ifor.distance_matrices_analysis()
    ifor.depths_array_analysis()
