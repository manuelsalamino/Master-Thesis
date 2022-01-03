import random
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
import sympy
from sympy import Matrix
from sympy.solvers.solveset import linsolve
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from math import ceil, floor
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM

from IForest import IForest
from ITree.INode import INode
from Dataset import RealDataset
from ExtendedIForest import ExtendedIForest
from utils_functions import evaluate_results


def IFOR_AS_Simplex_hyperplane_intersection():

    # define symbols (variables of equations)
    string = ''
    for i in range(1, n_dim+1):
        string += 'h' + str(i) + ', '
    string = string[:-2]

    symbols = sympy.symbols(string)

    # IFOR hyperplane
    # MATRIX [H1, H2, ... Hn, KNOWN TERM (on the right part of the equation)]
    ifor_eq = [i for i in range(1, n_dim+1)]
    ifor_eq.append(0)            # [1, 2, 3, 4, ..., n_dim, 0]   ->   h1 + 2*h2 + ... + n_dim*hn = 0

    # simplex
    simplex_eq = [1] * (n_dim+1)          # [1, 1, 1, ..., 1, 1]   ->   h1 + h2 + ... + hn = 1

    print(linsolve(Matrix((ifor_eq, simplex_eq)), symbols))


#def IFOR_AS_normal_projection_on_simplex():


def ifor_wo_correction_factor():
    # upload REAL dataset
    dir_path = os.path.join(os.getcwd(), '../datasets')
    files = os.listdir(dir_path)
    files.remove('Pendigits.csv')
    files.remove('hbk.csv')
    files.remove('wood.csv')

    results = pd.DataFrame()

    for dataset_name in files:
        for i in range(10):
            print(dataset_name)
            path = os.path.join(dir_path, dataset_name)
            dataset = pd.read_csv(path)
            labels = np.array(dataset.iloc[:, (-1)])
            dataset = dataset.drop(["id", "y"], axis=1).to_numpy()

            ifor = IForest()
            ifor.fit(dataset)
            depths = ifor.profile(dataset)

            depths_wo_correction = np.clip(depths, 0, ceil(np.log2(256)))

            scores, scores_wo_correction = [], []
            for i in range(len(depths)):
                avg = np.mean(depths[i])        # avg path length
                score = pow(2, -avg / INode.c(256))
                scores.append(score)

                avg = np.mean(depths_wo_correction[i])          # avg path length
                score = pow(2, -avg / INode.c(128))
                scores_wo_correction.append(score)

            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)

            print("roc w/ correction", roc_auc)

            fpr, tpr, _ = roc_curve(labels, scores_wo_correction)
            roc_auc_wo = auc(fpr, tpr)

            print("roc w/o correction", roc_auc_wo)

            df = pd.DataFrame(data={'dataset': dataset_name[:-4],
                                    'ROC AUC con Correzione': roc_auc,
                                    'ROC AUC senza Correzione': roc_auc_wo},
                              index=[0])

            results = results.append(df)

    with pd.ExcelWriter('C:\\Users\\Manuel\\Desktop\\Manuel\\PoliMi\\Tesi\\ifor_correction_factor.xlsx', mode='w') as writer:
        results.to_excel(writer, index=False)


def paired_T_test():
    # upload REAL dataset
    #path = os.path.join(os.getcwd(), '../../ifor_correction_factor.xlsx')
    #path = os.path.join(os.getcwd(), 'OC_SVM embedding vs  original - STRATIFIED - FINAL.xlsx')
    path = os.path.join(os.getcwd(), 'LDA embedding vs LDA original - STRATIFIED - FINAL - no correction.xlsx')

    results = pd.DataFrame()

    data = pd.read_excel(path)
    #data = (data.iloc[:, 1:4]).to_numpy()
    data = (data.iloc[:, [0, 1, 15]]).to_numpy()
    mask = np.invert(data[:, 0] == 'MEDIA')
    mask[0] = False
    data = data[mask]
    data = data[:, 1:]

    for j in range(1):
        #data_tmp = data[10*j:(10*j)+10]
        #data_tmp = data[[i*10+j for i in range(12)]]          # j-th element
        #data_tmp = np.asarray([np.mean(data[i*10:(i+1)*10], axis=0) for i in range(12)])    # mean

        '''data_tmp = []
        for i in range(12):
            tmp = data[i*10:(i+1)*10]
            #np.random.shuffle(tmp)
            #data_tmp = tmp[:j]
            diff = [tmp[i, 1] - tmp[i, 0] for i in range(len(tmp))]
            val, idx = min((val, idx) for (idx, val) in enumerate(diff))
            data_tmp.append(tmp[idx])
        data_tmp = np.asarray(data_tmp)'''
        #data_tmp = np.asarray([random.choice(data[10*i:(10*i)+10]) for i in range(12)])      # one per dataset random
        data_tmp = data

        for alt in ['two-sided', 'less', 'greater']:
            a = stats.ttest_rel(data_tmp[:, 0], data_tmp[:, 1], alternative=alt)

            print("i: ", j, "alternative: ", alt)
            print(a)

            if alt == 'two-sided':
                if a[1] > 0.1:
                    outcome = "accept"
                else:
                    outcome = "reject"

            elif alt == 'greater':
                if (a[0] > 0) and (a[1] / 2 < 0.1):
                    outcome = "reject"
                else:
                    outcome = "accept"

            elif alt == 'less':
                if (a[0] < 0) and (a[1] / 2 < 0.1):
                    outcome = "reject"
                else:
                    outcome = "accept"

            df = pd.DataFrame([[j, alt, a[0], a.pvalue, a.pvalue/2, outcome]],
                              columns=['j', 'alt', 'tstat', 'pvalue', 'pvalue/2', 'outcome'])

            results = results.append(df)

    with pd.ExcelWriter('paired_T_test.xlsx', mode='w') as writer:
        results.to_excel(writer, index=False)


def roc_curve_plot():
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    from scipy import interp
    from sklearn.metrics import roc_auc_score

    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Binarize the output
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(
        svm.SVC(kernel="linear", probability=True, random_state=random_state)
    )
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(6, 5))
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="ROC Curve",
        color="navy",
        #linestyle=":",
        linewidth=2,
    )

    plt.plot([0, 1], [0, 1], color='gray', linestyle='dashed', lw=2, label="random guesser")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()


    # HIGHLIGTHS AUC
    plt.figure(figsize=(6, 5))
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="ROC Curve",
        color="navy",
        #linestyle=":",
        linewidth=2,
    )

    plt.plot([0, 1], [0, 1], color='gray', linestyle='dashed', lw=2, label="random guesser")
    plt.fill_between(fpr["macro"], tpr["macro"], alpha=0.3, label=f"ROC AUC: {roc_auc['macro']:.2f}")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()


def LDA_test():
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

    # files = ['Breastw.csv']
    # files = ['Shuttle.csv', 'Smtp.csv']

    results = pd.DataFrame()

    for dataset_name in files:
        print(dataset_name)
        path = os.path.join(dir_path, dataset_name)
        dataset = RealDataset(path)

        ifor = ExtendedIForest(N_ESTIMATORS=100, MAX_SAMPLES=256, dataset=dataset)
        ifor.fit_IForest()
        ifor.profile_IForest()

        # IFOR PREDICTION
        scores = ifor.get_anomaly_scores()
        precision_ifor, roc_auc_ifor = evaluate_results(y_true=ifor.dataset.labels,
                                                        y_pred_score=scores,
                                                        y_pred_labels=[1 if s >= 0.5 else 0 for s in scores])
        print("IFOR roc auc: ", roc_auc_ifor)

        '''# ROC AUC USING AVG PATH LENGTH
        avg_depths = []
        for i in range(len(ifor.depths)):
            depths_i = ifor.depths[i]   # output of i-th test instance

            avg = np.mean(depths_i)     # avg path length
            avg_depths.append(-avg)       # -avg to have an inverse order for roc computation

        precision_ifor, roc_auc_ifor_avg = evaluate_results(y_true=ifor.dataset.labels,
                                                            y_pred_score=avg_depths,
                                                            y_pred_labels=[1 if s <= 10.18 else 0 for s in avg_depths])
        print("AVG path length roc auc: ", roc_auc_ifor_avg)'''

        # EMBEDDING
        ifor.trees_heights_as_histogram()

        # LDA CV 5-FOLD
        lda = LinearDiscriminantAnalysis(solver='lsqr')
        roc_auc_cv_LDA = cross_val_score(lda, ifor.histogram_data, ifor.dataset.labels,
                                         cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc')
        print(f'roc_auc_cv LDA: {roc_auc_cv_LDA}, mean: {np.mean(roc_auc_cv_LDA)}')

        # LDA
        training_indexes, test_indexes = train_test_split(np.arange(ifor.dataset.n_samples), train_size=0.8)

        lda = LinearDiscriminantAnalysis(solver='lsqr')
        lda.fit(ifor.histogram_data[training_indexes], ifor.dataset.labels[training_indexes])

        # TEST LDA
        y_pred_labels = lda.predict(ifor.histogram_data[test_indexes])
        y_pred = lda.predict_proba(ifor.histogram_data[test_indexes])[:, 1]

        precision, roc_auc_LDA = evaluate_results(y_true=ifor.dataset.labels[test_indexes],
                                              y_pred_score=y_pred,
                                              y_pred_labels=y_pred_labels)
        print('roc_auc LDA: ', roc_auc_LDA)

        #rint("coeff:", lda.coef_)

        '''# ROC AUC USING LINEAR COMBINATION
        scores_lda = []
        for h in ifor.histogram_data[test_indexes]:
            scores_lda.append(np.dot(h, lda.coef_[0]))

        precision_ifor, roc_auc_score_lda = evaluate_results(y_true=ifor.dataset.labels[test_indexes],
                                                            y_pred_score=scores_lda,
                                                            y_pred_labels=[1 if s <= 10.18 else 0 for s in scores_lda])
        print("AVG path length roc auc: ", roc_auc_score_lda)'''


        #################### EMBEDDING WITH DECORRELATION + LDA

        pca = PCA(n_components=ifor.histogram_data.shape[1] - 7)        # remove components
        histogram_data = pca.fit_transform(ifor.histogram_data)
        variance = pca.explained_variance_
        variance_ratio = pca.explained_variance_ratio_

        print(variance)
        print(variance_ratio)

        # LDA PCA CV 5-FOLD
        lda = LinearDiscriminantAnalysis(solver='lsqr')
        roc_auc_pca_LDA = cross_val_score(lda, histogram_data, ifor.dataset.labels,
                                         cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc')
        print(f'roc_auc_pca LDA: {roc_auc_pca_LDA}, mean: {np.mean(roc_auc_pca_LDA)}')

        ##################### DECORRELATION + LDA representation
        #training_indexes, test_indexes = train_test_split(np.arange(ifor.dataset.n_samples), train_size=0.8)

        lda = LinearDiscriminantAnalysis(solver='lsqr')
        lda.fit(histogram_data[training_indexes], ifor.dataset.labels[training_indexes])

        # TEST LDA
        #y_pred_labels = lda.predict(histogram_data[test_indexes])
        #y_pred = lda.predict_proba(histogram_data[test_indexes])[:, 1]

        scores_lda = []
        for h in histogram_data[test_indexes]:
            scores_lda.append(np.dot(h, lda.coef_[0]))

        scores_lda = np.asarray(scores_lda)

        precision_ifor, roc_auc_score_lda = evaluate_results(y_true=ifor.dataset.labels[test_indexes],
                                                             y_pred_score=scores_lda,
                                                             y_pred_labels=[1 if s <= 10.18 else 0 for s in scores_lda])
        print("LINEAR COMBINATION roc auc: ", roc_auc_score_lda)

        normal_indexes = np.argwhere(ifor.dataset.labels[test_indexes] == 0).reshape(-1, )
        anomaly_indexes = np.argwhere(ifor.dataset.labels[test_indexes] == 1).reshape(-1, )

        plt.plot([min(scores_lda), max(scores_lda)], [0, 0], 'b-')
        plt.plot([min(scores_lda), max(scores_lda)], [1, 1], 'b-')

        plt.plot(scores_lda[normal_indexes], [1]*len(normal_indexes), 'go')
        plt.plot(scores_lda[anomaly_indexes], [0] * len(anomaly_indexes), 'ro')

        plt.title(f'{dataset_name} - roc auc: {roc_auc_score_lda}')

        plt.show()


def PCA_test():
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

    # files = ['Breastw.csv']
    # files = ['Shuttle.csv', 'Smtp.csv']

    results = pd.DataFrame()

    for dataset_name in files:
        print(dataset_name)
        path = os.path.join(dir_path, dataset_name)
        dataset = RealDataset(path)

        ifor = ExtendedIForest(N_ESTIMATORS=100, MAX_SAMPLES=256, dataset=dataset)
        ifor.fit_IForest()
        ifor.profile_IForest()

        # IFOR PREDICTION
        scores = ifor.get_anomaly_scores()
        precision_ifor, roc_auc_ifor = evaluate_results(y_true=ifor.dataset.labels,
                                                        y_pred_score=scores,
                                                        y_pred_labels=[1 if s >= 0.5 else 0 for s in scores])
        print("IFOR roc auc: ", roc_auc_ifor)

        # EMBEDDING
        ifor.trees_heights_as_histogram()

        pca = PCA(n_components=ifor.histogram_data.shape[1] - 7)        # remove components
        histogram_data = pca.fit_transform(ifor.histogram_data)
        variance = pca.explained_variance_
        variance_ratio = pca.explained_variance_ratio_

        print(variance)
        print(variance_ratio)

        normal_indexes = np.argwhere(ifor.dataset.labels == 0).reshape(-1, )
        normal_indexes = random.choices(normal_indexes, k=100)  # floor(len(normal_indexes)*0.3))    # data sampling

        anomaly_indexes = np.argwhere(ifor.dataset.labels == 1).reshape(-1, )
        anomaly_indexes = random.choices(anomaly_indexes, k=100)  # floor(len(anomaly_indexes) * 0.3))    # data sampling

        plt.plot([min(histogram_data), max(histogram_data)], [0, 0], 'b-')
        plt.plot([min(histogram_data), max(histogram_data)], [1, 1], 'b-')

        plt.plot(histogram_data[normal_indexes], [1]*len(normal_indexes), 'go')
        plt.plot(histogram_data[anomaly_indexes], [0] * len(anomaly_indexes), 'ro')

        #plt.title(f'{dataset_name} - roc auc: {roc_auc_score_lda}')
        plt.title(dataset_name)

        plt.show()


def OC_SVM_test():
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

    # files = ['Breastw.csv']
    # files = ['Shuttle.csv', 'Smtp.csv']

    results = pd.DataFrame()

    for dataset_name in files:
        print(dataset_name)
        path = os.path.join(dir_path, dataset_name)
        dataset = RealDataset(path)

        ifor = ExtendedIForest(N_ESTIMATORS=100, MAX_SAMPLES=256, dataset=dataset)
        ifor.fit_IForest()
        ifor.profile_IForest()
        ifor.trees_heights_as_histogram()

        ############# OC_SVM
        for kernel in ['poly']:
            for gamma in ['auto']:
                for nu in [0.99]:
                    '''# 5-FOLD CROSS VALIDATION using DECORRELATED DATA
                    svm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
                    roc_auc_pca = cross_val_score(svm, histogram_data, ifor.dataset.labels,
                                                  cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc')
                    print('roc_auc_cv DECORRELATED: ', roc_auc_pca)'''

                    # 5-FOLD CROSS VALIDATION using NON-DECORRELATED DATA
                    svm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
                    roc_auc = cross_val_score(svm, ifor.histogram_data, ifor.dataset.labels,
                                              cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc')
                    print('roc_auc_cv NON DECORRELATED: ', roc_auc)

                    # 5-FOLD CROSS VALIDATION using ORIGINAL DATA
                    svm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
                    roc_auc_orig = cross_val_score(svm, dataset.data, ifor.dataset.labels,
                                                   cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc')
                    print('roc_auc_cv ORIGINAL: ', roc_auc_orig, '\n')

                    df = pd.DataFrame([[ifor.dataset.dataset_name, roc_auc_ifor,
                                        i, kernel, gamma, nu,
                                        roc_auc_pca[0], roc_auc_pca[1], roc_auc_pca[2], roc_auc_pca[3],
                                        roc_auc_pca[4], np.mean(roc_auc_pca)]],
                                      # roc_auc[0], roc_auc[1], roc_auc[2], roc_auc[3],
                                      # roc_auc[4], np.mean(roc_auc),
                                      # roc_auc_orig[0], roc_auc_orig[1], roc_auc_orig[2], roc_auc_orig[3],
                                      # roc_auc_orig[4], np.mean(roc_auc_orig)]],
                                      columns=['dataset', 'roc_auc_ifor',
                                               'deleted PCA', 'kernel', 'gamma', 'nu',
                                               'roc_auc_pca - 0', 'roc_auc_pca - 1', 'roc_auc_pca - 2',
                                               'roc_auc_pca - 3', 'roc_auc_pca - 4', 'MEDIA'])
                    # 'roc_auc_nopca - 0', 'roc_auc_nopca - 1', 'roc_auc_nopca - 2',
                    # 'roc_auc_nopca - 3', 'roc_auc_nopca - 4', 'MEDIA',
                    # 'roc_auc_orig - 0', 'roc_auc_orig - 1', 'roc_auc_orig - 2',
                    # 'roc_auc_orig - 3', 'roc_auc_orig - 4', 'MEDIA'])

                    results = results.append(df)

    with pd.ExcelWriter('output.xlsx', mode='w') as writer:
        results.to_excel(writer, index=False)


paired_T_test()