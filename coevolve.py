# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class CoEvolveRule(object):
    def __init__(self, order):
        self.order = order

    def rewire(self, graph, s_e, anchor):
        pass


class Rewire(CoEvolveRule):

    def __init__(self, order=4):
        super(self.__class__, self).__init__(order)

    def rewire(self, graph, s_e, anchor):
        size = len(graph)
        if anchor is None:
            pass
        else:
            p = []
            if s_e == 0:    # 随机选择
                p = np.ones(size)
            elif s_e == 1:  # 度优先
                p = np.array(G.degree().values(), dtype=np.float64)
            elif s_e == 2:  # 相似度
                p = np.array([len(list(nx.common_neighbors(G, anchor, x))) for x in G.nodes_iter()], dtype=np.float64)
                # 防止没有足够公共节点的
                p += 1
            elif s_e == 3:
                pass
            elif s_e == 4:
                pass
            p[anchor] = 0
            p /= float(p.sum())
            old = np.random.choice(graph.neighbors(anchor))
            new = np.random.choice(size, replace=False, p=p)
            graph.remove_edge(anchor, old)
            graph.add_edge(anchor, new)
        return old, new

    def rewire_new(self, G, s_e, anchor):
        # rewire only one link
        if s_e == 0:
            p = np.ones(self.N)
        elif s_e == 1:
            p = np.array(G.degree().values(), dtype=np.float64)
        else:
            pass
        p[anchor] = 0
        p /= float(p.sum())
        new_neigh = np.random.choice(self.N, 1, replace=False, p=p)
        k = G.degree(anchor)
        G.remove_edges_from(G.edges(anchor)[np.random.choice(k, 1, False)])
        G.add_edge(anchor, new_neigh)


if __name__ == '__main__':
    G = nx.random_regular_graph(5, 100)
    fitness = np.random.randint(1,3, size=100) * 1.0
