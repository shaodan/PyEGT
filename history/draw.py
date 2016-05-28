# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib import cm
from numpy.random import randn

def degree_histogram():
    G = nx.gnp_random_graph(100,0.02)

    degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence
    #print "Degree sequence", degree_sequence
    dmax=max(degree_sequence)

    plt.loglog(degree_sequence,'b-',marker='o')
    plt.title("Degree rank plot")
    plt.ylabel("degree")
    plt.xlabel("rank")

    # draw graph in inset
    plt.axes([0.45,0.45,0.45,0.45])
    Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)[0]
    pos=nx.spring_layout(Gcc)
    plt.axis('off')
    nx.draw_networkx_nodes(Gcc,pos,node_size=20)
    nx.draw_networkx_edges(Gcc,pos,alpha=0.4)

    # plt.savefig("degree_histogram.png")
    plt.show()


def unknown():
    # Make plot with vertical (default) colorbar
    fig, ax = plt.subplots()

    data = np.clip(randn(250, 250), -1, 1)

    cax = ax.imshow(data, interpolation='nearest', cmap=cm.coolwarm)
    ax.set_title('Gaussian noise with vertical colorbar')

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['< -1', '0', '> 1'])# vertically oriented colorbar

    # Make plot with horizontal colorbar
    fig, ax = plt.subplots()

    data = np.clip(randn(250, 250), -1, 1)

    cax = ax.imshow(data, interpolation='nearest', cmap=cm.afmhot)
    ax.set_title('Gaussian noise with horizontal colorbar')

    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], orientation='horizontal')
    cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])# horizontal colorbar

    plt.show()


degree_histogram()