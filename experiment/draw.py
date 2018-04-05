# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib import cm
from numpy.random import randn
import collections
import math


def degree_histogram():
    G = nx.gnp_random_graph(100, 0.02)

    degree_sequence=sorted(nx.degree(G).values(),reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    # dmax=max(degree_sequence)

    plt.loglog(degree_sequence, 'b-', marker='o')
    plt.title("Degree rank plot")
    plt.ylabel("degree")
    plt.xlabel("rank")

    # draw graph in inset
    plt.axes([0.45, 0.45, 0.45, 0.45])
    gcc=sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
    pos=nx.spring_layout(gcc)
    plt.axis('off')
    nx.draw_networkx_nodes(gcc, pos, node_size=20)
    nx.draw_networkx_edges(gcc, pos, alpha=0.4)

    # plt.savefig("degree_histogram.png")
    plt.show()


def imshow_interpolation():
    """
    https://matplotlib.org/gallery/images_contours_and_fields/interpolation_methods
        .html#sphx-glr-gallery-images-contours-and-fields-interpolation-methods-py
    https://matplotlib.org/gallery/subplots_axes_and_figures/subplot_toolbar
        .html#sphx-glr-gallery-subplots-axes-and-figures-subplot-toolbar-py
    """
    methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
               'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
               'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # grid = np.random.rand(50, 50)
    grid = np.random.randint(2, size=(50, 50))

    fig, axes = plt.subplots(3, 6, figsize=(12, 6), subplot_kw={'xticks': [], 'yticks': []})

    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    for ax, interp_method in zip(axes.flat, methods):
        ax.imshow(grid, interpolation=interp_method, cmap='bwr') #'viridis')
        ax.set_title(interp_method)

    plt.show()


def colorbar():
    """
    https://matplotlib.org/gallery/ticks_and_spines/colorbar_tick_labelling_demo
        .html#sphx-glr-gallery-ticks-and-spines-colorbar-tick-labelling-demo-py
    """
    # Make plot with vertical (default) colorbar
    fig, ax = plt.subplots()

    data = np.clip(randn(250, 250), -1, 1)

    cax = ax.imshow(data, interpolation='nearest', cmap=cm.coolwarm)
    ax.set_title('Gaussian noise with vertical colorbar')

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar

    # Make plot with horizontal colorbar
    fig, ax = plt.subplots()

    data = np.clip(randn(250, 250), -1, 1)

    cax = ax.imshow(data, interpolation='nearest', cmap=cm.afmhot)
    ax.set_title('Gaussian noise with horizontal colorbar')

    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], orientation='horizontal')
    cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar

    plt.show()


def degree_distr_logbined():
    """
    log bin plot
    http://stackoverflow.com/questions/16489655/plotting-log-binned-network-degree-distributions
    """
    g = nx.barabasi_albert_graph(1000, 3)
    # degree_h = nx.degree_histogram(g)

    # convert normalized degrees to raw degrees
    raw_data = dict(collections.Counter([k for _, k in g.degree]))
    plt.scatter(raw_data.keys(), raw_data.values(), c='b', marker='x')

    x = [float(i) for i in raw_data.keys()]
    y = raw_data.values()
    max_x = math.log10(max(x))
    max_y = math.log10(max(y))
    max_base = max([max_x, max_y])

    min_x = math.log10(min(filter(lambda z: z > 0, x)))
    bins = np.logspace(min_x, max_base, num=50)

    log_x = (np.histogram(x, bins, weights=y)[0] / np.histogram(x, bins)[0])
    log_y = (np.histogram(x, bins, weights=x)[0] / np.histogram(x, bins)[0])

    plt.scatter(log_x, log_y, c='r', marker='s', s=50)
    # plt.plot(range(len(data)), data, 'bo')
    plt.xscale('log')
    plt.yscale('log')
    # plt.xlim(0, 1000)
    # plt.xlim((1e-1, 1e5))
    # plt.ylim((.9, 1e4))
    # plt.xlim(1, max_x)
    # plt.ylim(1, max_y)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()


def degree_distr():

    def plot(data):
        """ Plot Distribution """
        plt.plot(range(len(data)), data, 'bo')
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel('Freq')
        plt.xlabel('Degree')
        # plt.subplot(131)
        # plt.clf()

        """ Plot CDF """
        # s = float(data.sum())
        # cdf = data.cumsum(0) / s
        # plt.plot(range(len(cdf)), cdf, 'bo')
        # plt.xscale('log')
        # plt.ylim([0, 1])
        # plt.ylabel('CDF')
        # plt.xlabel('Degree')
        # plt.subplot(132)
        # plt.clf()

        """ Plot CCDF """
        # ccdf = 1 - cdf
        # plt.plot(range(len(ccdf)), ccdf, 'bo')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.ylim([0, 1])
        # plt.ylabel('CCDF')
        # plt.xlabel('Degree')
        # plt.subplot(133)
        # plt.clf()
        plt.show()

    """ Load graph """
    G = nx.barabasi_albert_graph(1000, 3)

    """ To sparse adjacency matrix """
    M = nx.to_scipy_sparse_matrix(G)

    indegrees = M.sum(0).A[0]
    outdegrees = M.sum(1).T.A[0]
    indegree_distribution = np.bincount(indegrees)
    outdegree_distribution = np.bincount(outdegrees)
    # print np.bincount([d for _, d in G.degree])
    print indegree_distribution
    # return

    plot(indegree_distribution)
    plot(outdegree_distribution)


# degree_histogram()
# imshow_interpolation()
colorbar()
# degree_distr()
# degree_distr_logbined()
