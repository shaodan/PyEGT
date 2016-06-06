# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2016.05.05 -*-

from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt


def degree_distribution(graph):
    degree = graph.degree().values()
    distr = Counter(degree)
    plt.scatter(degree, distr)