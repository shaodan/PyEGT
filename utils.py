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



def mutatable(fun, mutate=True):

    def method_without_mutate(self, *argv):
        return fun(self, *argv, False)

    def method_with_mutate(self):
        if np.random.random() <= 0.01:
            return fun(self, *argv, True)
        else:
            return fun(self, *argv, False)

    return method_with_mutate if mutate else method_without_mutate
