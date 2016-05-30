# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2016.03.30 -*-

import networkx as nx
import numpy as np


class Population(nx.Graph):
    """
    Structure of population as a graph
    """
    def __init__(self, graph):
        assert isinstance(graph, nx.Graph)
        super(self.__class__, self).__init__(graph)
        # nx.convert_node_labels_to_integers(self)
        self.size = len(self)
        self.degree = self.degree().values()
        # data of node is stored in dict which is memory inefficient
        # use list instead
        self.fitness = np.empty(self.size, dtype=np.double)
        self.strategy = np.random.randint(2, size=self.size)
        del graph

    def add_edge(self, u, v):
        # u, v must exist and edge[u,v] must not exist
        super(self.__class__, self).add_edge(u, v)
        self.degree[u] += 1
        self.degree[v] += 1

    def remove_edge(self, u, v):
        super(self.__class__, self).remove_edge(u, v)
        self.degree[u] -= 1
        self.degree[v] -= 1

    def random_node(self, ):
        np.random.choice(self.node.keys())

    # def choice_node(self, size=None, replace=True, p=None):
    #     np.random.choice(self.node.keys(), size, replace, p)

    def random_edge(self):
        # choice random pair in graph
        size = len(self.edges())
        birth, death = self.edge[np.random.randint(size)]
        return birth, death
