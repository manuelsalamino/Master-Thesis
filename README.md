# Master Thesis - Anomaly Detection via Isolation Forest Embedding

You can look at the [Executive_Summary](https://github.com/manuelsalamino/Master-Thesis/blob/main/Executive_Summary.pdf) (a 6 pages paper-like summary) to have a more detailed overview of the thesis.

## Introduction
Anomaly Detection (AD) is a Data Mining process and consists finding unusual patterns or rare observations in a set of data. Usually anomalies represent negative events, in fact anomaly detection is used in many different fields, from medicine to industry.

<p align="center">
  <img width="300" src="https://github.com/manuelsalamino/manuelsalamino/blob/main/Images/anomaly.png">
</p>

We faced the problem by taking as starting point a milestone AD algorithm: _Isolation Forest (iForest)_.

This thesis propose to use the **intermediate output of iForest** to create an **embedding**, hence a new data representation on which known classification or anomaly detection techniques can be applied. Our empirical evaluation shows that our approach performs just as well, and sometimes better, than iForest on the same data. But our most important result is the creation of a new framework to enable other techniques to improve the anomaly detection performance.

## Isolation Forest

iForest is an unsupervised model-based method for anomaly detection. This method represent a breakthrough, before iForest the usual approach to AD problems was: construct a _normal data profile_, then test unseen data instances and identify as anomalies the instances that do not conform to the normal profile. iForest differs from all the previous ones since it is based on the idea of _directly isolates anomalies_, instead of recognized them as far from the normal data profile.

This approach works because anomalies are more susceptible to isolation than normal instances: a normal instance requires much more partitions than an anomaly to be isolated. iForest assigns an anomaly score to each instance based on the number of splits required to isolate them.

The model is based on a trees ensemble, each tree is called _Isolation Tree (iTree)_. In each iTree shortest paths (few splits) identify anomalies, while longest ones (more splits) predict normal instances.

<p align="center">
  <img width="700" src="https://github.com/manuelsalamino/manuelsalamino/blob/main/Images/iforest_label.png">
</p>


## Proposed Solution

We introduce a new **embedding** that gives to _input data x_ a new representation, but first of all introduce some definitions:
 - _depths vector y_: intermediate output of iForest, _y<sub>i</sub>_ is the returned depth of the _i-th_ iTree;
 - _histogram h_: histogram of _depths vector y_, then it is normalized: ||h||<sub>1</sub>=1.

Let's summarize how to obtain the histogram _h_ from input instance _x_:
<img src="https://latex.codecogs.com/svg.image?x&space;\in&space;\mathbb{R}^d&space;\xrightarrow{\hspace{3px}iForest\hspace{3px}}&space;y&space;\in&space;\mathbb{R}^t&space;\xrightarrow{\hspace{3px}histogram\hspace{3px}}&space;h&space;\in&space;\mathbb{Q}^n" title="x \in \mathbb{R}^d \xrightarrow{\hspace{3px}iForest\hspace{3px}} y \in \mathbb{R}^t \xrightarrow{\hspace{3px}histogram\hspace{3px}} h \in \mathbb{Q}^n" />

Using the embedding, and so using this new representation of input data, we expect normal and anomalous instances to yield different histograms, i.e. anomalous instances have high frequencies for bins representing low depths, while normal instances have high frequencies for bins representing high depths.

<p align="center">
  <img width="700" src="https://github.com/manuelsalamino/manuelsalamino/blob/main/Images/histogram.png">
</p>

We use this new embedding to represent data in a totally different way, based on the iForest output, with the goal of perform other anomaly detection techniques in this new space and reach results better than the starting point iForest.

On the [Executive_Summary](https://github.com/manuelsalamino/Master-Thesis/blob/main/Executive_Summary.pdf) (a 6 pages paper-like summary) a more detailed overview of the thesis, with the experiments and the obtained results.
