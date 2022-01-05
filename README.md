# IFOR_Embedding

## Introduction
Anomaly Detection (AD) is a Data Mining process and consists finding unusual patterns or rare observations in a set of data. Usually anomalies represent negative events, in fact anomaly detection is used in many different fields, from medicine to industry. We faced the problem by taking as starting point a milestone AD algorithm: Isolation Forest (iForest). This thesis propose to use the **intermediate output of iForest** to create an **embedding**, hence a new data representation on which known classification or anomaly detection techniques can be applied. Our empirical evaluation shows that our approach performs just as well, and sometimes better, than iForest on the same data. But our most important result is the creation of a new framework to enable other techniques to improve the anomaly detection performance.

## Isolation Forest

iForest\textsuperscript{\cite{ifor}} is an unsupervised model-based method for anomaly detection. This method represent a breakthrough, before iForest the usual approach to AD problems was: construct a \textit{normal data profile}, then test unseen data instances and identify as anomalies the instances that do not conform to the normal profile. iForest differs from all the previous ones since it is based on the idea of directly isolates anomalies, instead of recognized them as far from the normal data profile.\\
This approach works because anomalies are more susceptible to isolation than normal instances. In fact, Figure \ref{fig:isolation} shows that a normal instance requires much more partitions than an anomaly to be isolated. iForest assigns an anomaly score to each instance based on the number of splits required to isolate them.

\begin{figure}[t]
    \centering
    \subfigure[Isolating normal.]{
        \includegraphics[width=0.46\linewidth]{isolating_an.png}
        }
    \subfigure[Isolating anomaly.]{
        \includegraphics[width=0.46\linewidth]{isolating_norm.png}
        }
    \captionsetup{width=0.95\linewidth}
    \caption{(a) Normal point $x_i$ requires twelve partitions to be isolated;
    (b) Anomaly $x_0$ requires only four partitions to be isolated\textsuperscript{\cite{ifor}}.}
    \label{fig:isolation}
\end{figure}

The model is based on a trees ensemble, each tree is called \textit{Isolation Tree} (iTree). Each iTree is built on a different sub-sample of the dataset. This allows to obtain a set of different \textit{experts}, each one able to recognized different type of anomalies (depending on the sub-sample on which it is trained). Then, the prediction of a test instance is made by computing the average of the paths covered in each iTree before reaching a leaf node. Shortest paths (few splits) identify anomalies, while longest ones (more splits) predict normal instances.

\subsection{Isolation Tree} \label{sec:itree}
An iTree has the same structure of a Proper Binary Tree, a tree in which each node belongs exactly to one of the following categories:

\vspace{5px}

\begin{itemize}[itemsep=4px]
    \item \textit{external node}, a leaf node, no child;
    \item \textit{internal node}, a node with exactly two children nodes (right and left).
\end{itemize}

\vspace{10px}

Each iTree is built starting from a subset $X$ of the dataset (because of sub-sampling):
\[ |X| = \psi \qquad \qquad X \subset \mathbb{R}^d, \]
with $\psi$ sub-sample size and $d$ number of features.\\
To build an iTree, as for a Binary Tree, we recursively divide X using as \textit{splitting criterion} to divide data into right and left child. The \textit{splitting criterion} consists in randomly selecting an attribute $q$ (among the $d$ possible) and a split value $v$ in order to test if $q < v$. All the instances for which the test returns True are put in the left child, whereas all the others in the right one. The \textit{split value} of $v$ is randomly selected in the range of values of $q$ in $X$.\\
The splits are repeated until one of the following conditions is reached:

\vspace{5px}

\begin{enumerate}[label=(\roman*), itemsep=4px]
    \item the tree reaches a height limit $l$, or
    \item $|X| = 1$, or
    \item all data in X have the same value.
\end{enumerate}

\vspace{6px}

\subsection{Anomaly Score}
To understand the anomaly score computation, we have to introduce the \textit{Path Length} $p(x)$ of an instance $x$, which is defined as the number of edges traversed by $x$ from the root node to the external node reached following the iTree rules defined in the training phase. The expected behaviour of normal instances is to exit the iTrees from the deepest leaves (longer path length), while anomalies are expected to exit from the shallowest ones (shorter path length).\\
Before defining the \textit{anomaly score}, we define a function that is used both in anomaly score and in the evaluation phase: $c(n)$, which estimate the height of a Binary Search Tree\textsuperscript{\cite{ifor}} with $n$ elements,
\begin{equation}
    \label{eqn:normfactor}
    c(n) = 2H(n - 1) - \frac{2(n - 1)}{n},
\end{equation}

where $H(i)$ is the harmonic number and it can be estimated
by $ln(i)$ + $0.5772156649$ (Euler’s constant).
Now we define the \textit{anomaly score $s$} of an instance $x$ as:
\vspace{4px}
\begin{equation}
    \label{eqn:anomalyscore}
    s(x,\psi) = 2^{-\frac{E(p(x))}{c(\psi)}},
\end{equation}

\vspace{6px}

where $E(p(x))$ is the average $p(x)$ of the iTrees of the Forest and $c(\psi)$ is used to normalize.\\
The anomaly score reflects the relation between $E(p(x))$ and predicted label defined before: if shorter path length, $s$ close to 1, the test instance is an anomaly, if longer path length, $s$ close to 0, the test instance is normal.

\subsection{c(n) as Correction Factor} \label{sec:correct_factor}
During \textit{evaluation} of an instance $x$, in path length computation, there is key passage on which we discuss later: when a node, during training, is defined as \textit{terminal} because the tree height limit $l$ is reached, the iTree is truncated, so, during evaluation, we add to $p(x)$ an adjustment $c(n)$, where n is the number of not split elements in that leaf node. $c(n)$ estimates the height of the truncated subtree using (\ref{eqn:normfactor}).


\section{Proposed Solution} \label{sec:proposed_solution}
\subsection{Problem Formulation}
Let’s look at the problem of AD in a more formal way\textsuperscript{\cite{problem_formulation}}. The anomaly detection problem consists in mo\-ni\-to\-ri\-ng a set of $n$ data:
\[ \mathcal{X} = \{x_1, \dots, x_n \quad | \quad x_i \in \mathbb{R}^d\}, \]
where each element $x_i$ is realization of a random variable having pdf $\phi_0$, with the aim of detect outliers, i.e. points that do not conform with $\phi_0$.

\begin{equation*}
    x_i \sim 
    \begin{cases}
        \phi_0 & \textit{normal data}\\
        \phi_1 & \textit{anomalies}
    \end{cases}
\end{equation*}

where $\phi_0 \neq \phi_1$ and $\phi_0$ and $\phi_1$ are unknown. Since $X$ has no labels we are in an unsupervised setting, furthermore there is no clear distinction between training and test data. Each dataset contains both normal and anomalous samples. The only mild assumption we can make is that normal data far outnumber anomalies.

\subsection{Embedding} \label{sec:embedding}
We introduce a new embedding that gives to input data a new representation, but first of all introduce some definitions:

\vspace{5px}

\begin{itemize}[itemsep=4px]
    \item \textbf{depths vector $y$}: intermediate output of iForest, \( y \in \mathbb{R}^t \). $y_i$ is the returned depth of the \textit{i-th} iTree;
    
    \item \textbf{histogram $h$}: histogram of \textit{depths vector} $y$. Then it is normalized: $\lVert h \rVert_1 = 1$, $h \in \mathbb{Q}^n$, with \textit{n=ceil(max(y))}, thus the first integer bigger than the $max$ of $y$.\\
    More precisely each $h_i \in [0, 1]$ because represents the fraction of the $t$ elements that are into a specific bin:
    \begin{equation}
        \textit{h = histogram(y)}, \hspace{20px} \lVert h \rVert_1 = 1
        \label{eqn:histogram}
    \end{equation}
\end{itemize}

\vspace{5px}

\begin{comment}
    \vspace{-30px}
    \begin{adjustwidth}{0.05cm}{}
        \begin{equation}
            \centering
            \renewcommand{\arraystretch}{2.5}
            \begin{tabular}
            \hline
                \textit{h = histogram(y)}, \hspace{20px} $\lVert h \rVert_1 = 1$ \\
                $h_i=\frac{1}{t}$
                \begin{cases}
                    \#([i, \textit{i}+1)) & \text{if i = 1, \dots, n-1}\\
                    \#([i, \textit{i}+1]) & \text{if i = n}
                \end{cases}
            \end{tabular}
            \label{eqn:histogram}
        \end{equation}
    \end{adjustwidth}
        
    \begin{adjustwidth}{0.8cm}{}
        $\frac{1}{t}$ is used to obtain $\lVert h \rVert_1 = 1$ and $\#(\cdot)$ counts the elements into the given bin, all the bins are half-open, except for the last one which is closed.
    \end{adjustwidth}
    
    \vspace{5px}
\end{comment}


Let's summarize how to obtain the histogram $h$ from input instance $x$:
\begin{equation*}
    x \in \mathbb{R}^d \xrightarrow{\hspace{3px}iForest\hspace{3px}} y \in \mathbb{R}^t \xrightarrow{\hspace{3px}histogram\hspace{3px}} h \in \mathbb{Q}^n
\end{equation*}

\vspace{4px}

Our \textit{embedding} is a new dimensional space in $\mathbb{Q}^n$, where input instances $x$, transformed in the corresponding histogram $h$, lie in a $n$-dimensional \textit{simplex} (a triangle in $n$ \mbox{dimensions}), because the constraint $\lVert h \rVert_1 = 1$ positions all of them in an hyperplane, while $h_i \in [0, 1]$, for $i, \dots, n$, constraints them in a simplex on that hyperplane.\\
Using the embedding, and so using this new representation of input data, we expect normal and anomalous instances to yield different histograms, i.e. anomalous instances have high frequencies for bins representing low depths, while normal instances have high frequencies for bins representing high depths. See the histograms in Figure \ref{fig:histograms}.
\begin{figure}[t]
    \centering
    \subfigure[Anomaly instance]{
        \includegraphics[width=0.46\linewidth]{Breastw anomaly.png}
        }
    \subfigure[Normal instance]{
        \includegraphics[width=0.46\linewidth]{Breastw normal.png}
        }
    \captionsetup{width=0.95\linewidth}
    \caption{Data coming from Breastw dataset. (a) Anomaly shows that most iTrees return a depth between 5 and 12. (b) Normal shows highest frequencies between 13 and 18.}
    \label{fig:histograms}
\end{figure}

\subsection{Remove the Correction Factor} \label{sec:remove_correction_factor}
In Section \ref{sec:correct_factor} we introduce that, when an input instance $x$ reaches a terminal node, to $p(x)$ is added the correction factor $c(Size)$, which estimates the height of the truncated subtree.\\
The correction factor leads to two problems in the embedding:

\vspace{5px}

\begin{enumerate}[itemsep=4px]
    \item \textit{increases the embedding dimension} (increases the $n$ of $\mathbb{Q}^n$), and
    
    \item \textit{the depth becomes a real number}, $y \in \mathbb{R}^t$, instead of integer.
\end{enumerate}

\vspace{10px}

We want to start the improvements by redefining the anomaly score (\ref{eqn:anomalyscore}) in the embedding, to do that is necessary to have $y \in \mathbb{N}^t$, so from this point on, the embedding is defined using depths vector $y \in \mathbb{N}^t$ computed using iForest \textit{without correction}. Later, a formulation of $E(p(x))$ in the embedding if there is no correction factor.


\subsection{Average Path Length in Embedding} \label{sec:avg_path_length_in_embedding}

We redefine the average path length, so the $E(p(x))$ of iForest, in the embedding, is computed with the following expression:
\vspace{-2px}
\begin{equation}
    E(p(x)) = \sum_{i=1}^n w_i \cdot h_i,
    \label{eqn:avg_path_length_embedding}
\end{equation}

where $w_i=i$, $h_i$ is the \textit{i-th} element of $h$ and $h = histogram(y)$, as defined in (\ref{eqn:histogram}).\\
Since we are considering iForest \textit{without correction}, the depths vector $y$ is composed only by integer values ($y \in \mathbb{N}^t$), thus $h_i$ is the fraction of elements of the depths vector $y$ that are equal to $i$. Hence by multipling the fraction of how many times a value is present in $y$ ($h_i$) times the value itself ($w_i$) for $w_i = 1, \dots, n$ we obtain the average path length.


\subsection{Linear Discriminant Analysis} \label{sec:lda}

Looking at (\ref{eqn:avg_path_length_embedding}), we note that the $E(p(x))$ in the embedding is simply a linear combination of the embedding feature $h_1, \dots, h_n$ with predefined weights. The intuition is about finding a new weight vector $\hat{w}$, which is dataset-specific and can increase the AD performance. To this purpose we use Linear Discriminant Analysis (LDA) to find the new weights. LDA is a supervised method used in statistics and pattern recognition to find a linear combination of features that separates two or more classes. Hence we use LDA to find a linear combination of $h_1, \dots, h_n$ which better separates the two classes.\\
The new formulation is:
\begin{equation}
    L(y) = \sum_{i=1}^n \hat{w_i} \cdot h_i,
    \label{eqn:lda_lcombination}
\end{equation}
$\hat{w_1}, \dots, \hat{w_n}$ weights computed using LDA.\\
Using LDA we are using a supervised technique, so we are computing something like an upper bound of the performance of iForest, since the linear combination is computed using optimal weights $\hat{w}$.


\subsection{One-Class SVM}  \label{sec:ocsvm}

Another idea comes from the fact that the embedding is a totally new representation of the data. We apply in this new framework a well known AD technique: One-Class Support Vector Machine (OC SVM). It is able to identify the best boundary, in feature space, which separates normal instances from anomalies. We want to know if in this new data representation makes easier perform anomaly detection.


\section{Experiments} \label{sec:experiments}
\subsection{Datasets} \label{sec:datasets}
The datasets used for testing are the same used to evaluate iForest in \cite{ifor}. Twelve datasets: eleven of them are natural datasets, plus a synthetic one. These are the same datasets used to test iForest, since iForest is the benchmark for our performance and we want to evaluate our modification proposals using the same data on which it was originally tested.

\subsection{Remove the Correction Factor} \label{sec:exp_remove_correction_factor}
\begin{adjustwidth}{0.2cm}{}
    \textit{Goal.} Check whether iForest \textit{with correction factor} and iForest \textit{without correction factor} have the same performance.\\
\end{adjustwidth}

\textit{Paired-T-test} is used to determine whether the mean difference between two sets of observations is zero. The two sets of observation are the ROC AUCs obtained executing 10 times iForest \textit{with correction} and \textit{without correction}.\\
We use this to check if the performance are identical or not and if there is a statistical relevance on that. We perform two tests:

\vspace{5px}

\begin{itemize}[itemsep=4px]
    \item \textit{two-sided paired-T-test}, to check if the mean difference of the two variants is zero across different runs and datasets. It shows a \textit{p-value=}0.00313, and this means that the \textbf{null hypothesis of identical averages is rejected} and the two variants have different perfomances;
    
    \item \textit{one-sided} to verify if \textit{with correction < without correction}, it returns a positive \textit{t-statistic}, but a \textit{p-value=}0.0015, that halved become \textit{p-value=}0.0007, hence \textbf{is rejected}.
\end{itemize}

\vspace{10px}

We can state that traditional iForest performs better than the one \textit{without correction}. We have to take this into account, but to get en embedding with low dimensional space and a depths vector $y$ with integer values, we populate the embedding with iForest outputs generated using \textit{without correction} version.


\subsection{Embedding + LDA} \label{sec:emb+lda}

\begin{adjustwidth}{0.2cm}{}
    \textit{Goal.} Use Linear Discriminant Analysis (LDA) to find the best linear combination of embedding features that separates the two classes.
\end{adjustwidth}
\vspace{5px}

The idea of use LDA to find the weigths $\hat{w_1}, \dots, \hat{w_n}$ comes from the intuition described in Section \ref{sec:lda}. The resulting optimal $\hat{w}$ is used in Formula (\ref{eqn:lda_lcombination}), in order to obtain the scores and compute the ROC AUC. For every dataset, iForest is executed 10 times, and after each run, are computed: a) anomaly scores using average path length (standard iForest, no embedding) and b) anomaly scores computed using linear combination defined in (\ref{eqn:lda_lcombination}). LDA performance are measured by 5-fold cross validation. Results are reported in Table \ref{table:ldaresults}.

Among the 12 tested datasets, in 9 of them the version Embedding+LDA has performance better than the standard iForest. The other good thing is that in the three datasets in which standard iForest performs better, the difference is very small, while the proposed version, in some datasets (i.e. Mulcross, Mammography), has better results with a significant gap.\\
An important fact to remark is that the optimal weights $\hat{w}$ are \textit{dataset-specific}: we couldn't find a vector $\hat{w}$ that reaches very good performance in all the dataset. So, for each dataset is necessary a LDA training phase to find the best weights for that domain. The fact that some datasets show results that are better in traditional iForest than in the embedding+LDA, is surprising because we expect that use a supervised techniques, trained on each single dataset, returns better performances.

%%%%%%%%%%%%%%%% IFOR vs EMBEDDING + LDA - no correction
\begin{table}
    \centering
    \renewcommand{\arraystretch}{1.3}
    \begin{tabular}{|ccc|}
        \hline
        \rowcolor{bluePoli!50}
        Dataset & iForest & Embedding+LDA \T \B \\\hline \hline
        Http & \textbf{1.000} & 0.999 \\%\hline
        ForestCover & 0.938 & \textbf{0.969} \\%\hline
        Mulcross & 0.899 & \textbf{0.957} \\%\hline
        Smtp & 0.850 & \textbf{0.853} \\%\hline
        Shuttle & 0.995 & \textbf{0.997} \\%\hline
        Mammography & 0.764 & \textbf{0.823} \\%\hline
        Annthyroid & 0.816 & \textbf{0.818} \\%\hline
        Satellite & 0.707 & \textbf{0.726} \\%\hline
        Pima & 0.631 & \textbf{0.638} \\%\hline
        Breastw & 0.957 & \textbf{0.972} \\%\hline
        Arrhythmia & \textbf{0.781} & 0.776 \\%\hline
        Ionosphere & \textbf{0.863} & 0.856 \\\hline
    \end{tabular}
    \captionsetup{width=0.9\linewidth}
    \caption{In bold the higher ROC AUC. The embedding+LDA variant shows better results in 9 out of 12 datasets.}
    \label{table:ldaresults}
\end{table}

\subsection{Embedding + One-Class SVM}  \label{sec:emb+ocsvm}
\begin{adjustwidth}{0.2cm}{}
    \textit{Goal.} Obtain improvements applying One-Class SVM in the embedding.
\end{adjustwidth}
\vspace{5px}

With this experiment we evaluate the performance of OC SVM applied in the embedding space. The following results (Table \ref{table:ocsvm_results}) are obtained by executing 10 runs of iForest on each dataset, and after each run, are computed: a) the standard iForest performance and b) the embedding+OC SVM performance. In b), once the embedding representation is computed, OC SVM performances are measured using 5-fold Cross Validation.

\begin{table}
    \centering
    \renewcommand{\arraystretch}{1.3}
    \begin{tabular}{|ccc|}
        \hline
        \rowcolor{bluePoli!50}
        Dataset & iForest & Emb.+OC SVM \T \B \\\hline \hline
        Http & \textbf{1.000} & 0.999 \\%\hline
        ForestCover & \textbf{0.943} & 0.912 \\%\hline
        Mulcross & 0.896 & \textbf{0.918}\\% & NA \\%\hline
        Smtp & \textbf{0.878} & 0.872\\% & NA \\%\hline
        Shuttle & 0.995 & \textbf{0.997}\\% & NA \\%\hline
        Mammography & 0.752 & \textbf{0.754}\\% & 0.878/0.873 \\%\hline
        Annthyroid & 0.823 & \textbf{0.828}\\% & 0.528/0.551 \\%\hline
        Satellite & 0.703 & \textbf{0.711}\\% & 0.322/0.661 \\%\hline
        Pima & 0.631 & \textbf{0.641}\\% & 0.657/0.669 \\%\hline
        Breastw & 0.957 & \textbf{0.966}\\% & 0.996/995 \\%\hline
        Arrhythmia & 0.780 & \textbf{0.787}\\% & 0.443/0.798 \\%\hline
        Ionosphere & \textbf{0.868} & 0.865\\ \hline% & 0.203/0.842 \\\hline
    \end{tabular}
    \captionsetup{width=0.9\linewidth}
    \caption{In bold the higher ROC AUC. The embedding+OC SVM variant shows better results in 8 out of 12 datasets.}
    \label{table:ocsvm_results}
\end{table}

The results show that in 8 out of 12 datasets the version Embedding+OC SVM has performance better than standard iForest, even if these differences are small (<0.01), except for \textit{Mulcross} for which the difference, in favour of Embedding+OC SVM, is 0.022, and \textit{ForestCover} for which the performance of standard iForest are better for 0.031.
For this experiment there is also an interesting insight on OC SVM parameter tuning. We tuned three parameters defined in the scikit-learn documentation\textsuperscript{\cite{oc_svm_implementation}}:

\vspace{5px}

\begin{itemize}[itemsep=4px]
    \item \textit{Kernel}, function to compute the similarity of two instances in feature space.
    Tested kernels: \textit{linear}, \textit{poly}, \textit{rbf} and \textit{sigmoid}.
    
    \item \textit{Gamma}, how far the influence of a single training example reaches. Two predefined values: \textit{scale} and \textit{auto}. %\textit{scale=$1 / (n\_features \cdot X.variance)$}, with $X$ input data, and \textit{auto=$1 / n\_features$}.
    
    \item $\nu$, upper bound to the fraction of anomalies that can lie outside the boundary of normal region (e.g. if $\nu=0.1$, at most 10\% outside the decision boundary).
\end{itemize}

\vspace{10px}

Comparing the results of \textit{embedding+OC SVM} and \textit{OC SVM on original data} obtained by using different paramters combinations, we note that \textit{embedding+OC SVM} version has much more stability with different parameter values than \textit{OC SVM on original data}. We execute OC SVM by changing the values of the three parameters and we obtain this behaviour in all the datasets.\\
For the \textit{embedding+OC SVM} the obtained ROC AUC differs from a factor of $10^{-4}$ or $10^{-5}$ from one parameters combination to another (except for \textit{rbf kernel} which returns performance that are quite worst in all the datasets), while looking at \textit{OC SVM in original data}, these differences are much bigger and the ROC AUCs change a lot using different \textit{kernels} ($10^{-2}$ or $10^{-3}$).\\
Finally, we can infer that the embedding project input data in such a way that OC SVM perform in a similar way even if it is executed with different parameters, and this behaviour highlight a higher stability.


\section{Conclusions} \label{sec:conclusion}
In this thesis we faced the problem of AD by using a state-of-the-art algorithm: iForest. In particular, the main goal was to change its last operations with the intention of replace them with more complex ones, able to capture new features in the data.\\
The main contributions are:

\vspace{5px}

\begin{itemize}[itemsep=4px]
    \item using iForest as an embedding (Section \ref{sec:embedding}) where we are able to represent data in a totally different way, based on the iForest output;
    
    \item new algorithms (Sections \ref{sec:lda} and \ref{sec:ocsvm}) to perform anomaly detection that reach results better than the starting point iForest.
\end{itemize}

\vspace{10px}

Future works includes: embedding defined using iForest \textit{with correction factor}. A future development can be an in-depth analysis of how to maintain the correction factor when the output of Isolation Forest is obtained and the embedding space is populated by input transformed data. The computation of the maximum value of the correction factor can be useful to determine the upper bound of the embedding dimension, and work with histograms with an higher number of bins probably give more information, even if the increasing dimensionality is a drawback.
Another one is change the embedding formulation. The idea of starting from iForest output to create a new representation is maintained, but the histogram can be computed by using different bins, or giving more importance to the output of specific iTrees, instead of using an aggregative measure as the histograms.
The last one is, given the embedding space, use it to apply other AD techniques that can be applied also in original space.
