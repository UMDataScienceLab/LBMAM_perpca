import numpy as np
from numpy import linalg as LA
#import scipy.stats as st
#import statsmodels.api as sm
import os

from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import copy
from scipy.spatial.distance import cdist, euclidean

def subspace_error(U,V):
    pu = U@np.linalg.inv(U.T@U)@U.T    
    pv = V@np.linalg.inv(V.T@V)@V.T
    return np.linalg.norm(pu-pv)
def subspace_angle(U,V):
    qu,ru = np.linalg.qr(U)
    qv,rv = np.linalg.qr(V)

    Z = qu.T@qv
    u,s,vh = np.linalg.svd(Z)
    return 1-(s[1])

def spectral_cluster(V, names):
    ncl = len(V)
    afm = np.zeros((ncl, ncl))
    for i in range(ncl):
        for j in range(i):
            #afm[i, j] = subspace_error(V[i], V[j])
            afm[i,j] = subspace_angle(V[i],V[j])
            afm[j, i] = afm[i, j]

    maxele = np.max(afm)
    afm /= maxele
    afm = afm ** 2
    #print(afm)
    #print(afm[0])
    afm_copy = copy.deepcopy(afm)
    # cluster_plot(afm)
    for i in range(ncl):
        afm[i, i] -= afm[i].sum()
    clustering = SpectralClustering(n_clusters=2,
                                    assign_labels='discretize',
                                    random_state=0, affinity='precomputed').fit(afm)
    print(clustering.labels_)
    cluster_plot(afm_copy, clustering.labels_, names)

def cluster_plot(X, clabels, names):
    from sklearn.datasets import load_digits
    from sklearn.manifold import MDS, TSNE
    
    #embedding = MDS(n_components=2)
    #X_transformed = embedding.fit_transform(X)

    #import sklearn.manifold 
    tsne = TSNE(
            n_components=2, perplexity=40, metric='cosine', early_exaggeration=10.0,
            init='pca', verbose=True, n_iter=400)

    print('Running t-SNE...')
    X_transformed = tsne.fit_transform(X)

    print(X_transformed.shape)
    fig, ax = plt.subplots(1, 1)

    ax.scatter(X_transformed[:,0],X_transformed[:,1],s=0.4)
    for i in range(len(X)):
        ax.annotate(names[i], (X_transformed[i,0], X_transformed[i,1]))
    plt.savefig('user_embeddings.png')

def cluster_plot_simple(dis, clusters, names):
    color = ['lightcoral', 'sienna', 'darkorange', 'greenyellow', 'seagreen',
             'aquamarine', 'cyan', 'steelblue', 'navy', 'blueviolet', 'violet', 'pink']
    N = len(dis)
    dis_sq = dis  # *dis
    G = -0.5 * (np.eye(N) - np.ones((N, N)) / N) @ dis_sq @ (np.eye(N) - np.ones((N, N)) / N)

    evals, evecs = np.linalg.eigh(G)
    # print(evals)
    x = -evecs[:, -1] * np.sqrt(evals[-1])
    y = -evecs[:, -2] * np.sqrt(evals[-2])
    fig, ax = plt.subplots(1, 1)
    for i in range(len(x)):
        ax.scatter(x[i], y[i], color=color[clusters[i]])
    for i in range(len(dis)):
        ax.annotate(names[i], (x[i], y[i]))
    plt.savefig('clietsrelation.png')

