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
        self.size = len(graph)
        self.degree_list = self.degree().values()
        # data of node is stored in dict which is memory inefficient
        # use list instead
        self.fitness = np.empty(self.size, dtype=np.double)
        # 策略: 0合作， 1背叛
        self.strategy = np.random.randint(2, size=self.size)
        del graph

    def add_edge(self, u, v, attr_dict=None, **attr):
        # u, v must exist and edge[u,v] must not exist
        super(self.__class__, self).add_edge(u, v)
        self.degree_list[u] += 1
        self.degree_list[v] += 1

    def remove_edge(self, u, v):
        super(self.__class__, self).remove_edge(u, v)
        self.degree_list[u] -= 1
        self.degree_list[v] -= 1

    def random_node(self, ):
        np.random.choice(self.node.keys())

    def choice_node(self, size=None, replace=True, p=None):
        np.random.choice(self.node.keys(), size, replace, p)

    def random_edge(self):
        # choice random pair in graph
        size = len(self.edges())
        total = self.size * (self.size-1) / 2
        if total / size > 100:
            birth, death = np.random.randint(size, size=2)
            while birth == death or (not self.has_edge(birth, death)):
                birth, death = np.random.randint(size, size=2)
        else:
            birth, death = self.edge[np.random.randint(size)]
        return birth, death

    def prepare(self):
        self.fitness = np.empty(self.size, dtype=np.double)
        self.strategy = np.random.randint(2, size=self.size)

    def cooperation_rate(self):
        # count_nonzero() is faster than (self.strategy == 0).sum()
        return self.size - np.count_nonzero(self.strategy)

# TEST CODE HERE
if __name__ == '__main__':
    G = nx.random_graphs.watts_strogatz_graph(1000, 4, 0.3)
    P = Population(G)
    print P.degree()
    print P.edges()
    print list(nx.common_neighbors(P, 0, 1))
