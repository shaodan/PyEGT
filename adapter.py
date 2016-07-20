# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2015.07.11 -*-

import numpy as np
import networkx as nx
from population import Population


class Adapter(object):
    def __init__(self, order, min_degree=2):
        self.order = order
        self.min_degree = min_degree

    # prefer优先策略，anchor重连节点, source?
    def adapt(self, population, prefer, anchor, source=None):
        pass


class LocalAdapter(Adapter):

    def adapt(self, population, prefer, anchor, source=None):
        size = len(population)
        if anchor is None or source is None:
            return None
        alter = population.neighbors_of_neighbors(source)
        new = None
        if prefer == 0:
            new = np.random.choice(alter)
        elif prefer == 1:
            new = np.random.choice(alter)
        else:
            new = 1
        old = 0
        return old, new


class GlobalAdapter(Adapter):

    def adapt(self, population, prefer, anchor, source=None):
        pass


class Preference(Adapter):

    def __init__(self, order=4):
        super(self.__class__, self).__init__(order)

    def adapt(self, population, prefer, anchor, source=None):
        size = len(population)
        if anchor is None:
            return None
        p = []
        if prefer == 0:    # 随机选择
            p = np.ones(size)
        elif prefer == 1:  # 度优先
            p = np.array(population.degree_list, dtype=np.float64)
        elif prefer == 2:  # 相似度
            p = np.array([len(list(nx.common_neighbors(population, anchor, x))) for x in population.nodes_iter()],
                         dtype=np.float64)
            p += 1         # 防止没有足够公共节点的
        elif prefer == 3:
            pass
        elif prefer == 4:
            pass
        p[anchor] = 0
        p /= float(p.sum())
        old = np.random.choice(population.neighbors(anchor))
        new = np.random.choice(size, replace=False, p=p)
        population.remove_edge(anchor, old)
        population.add_edge(anchor, new)
        return old, new

    def adapt_once(self, population, prefer, anchor):
        # rewire only one link
        old = population.random_neighbor(anchor)
        if population.degree_list[old] <= self.min_degree:
            print "=======skip rewire(%d) neigh(%d)'s degree: %d===="%(anchor, old, population.degree(old))
            return
        new_list = population.nodes_exclude_neighbors(anchor)
        if prefer == 1:
            p = np.array([population.degree_list[x] for x in new_list], dtype=np.float64)
        elif prefer == 2:
             p = np.array([len(list(nx.common_neighbors(population, anchor, x))) for x in new_list],
                          dtype=np.float64)
             # p += 1         # 防止没有足够公共节点的
             if p.sum() == 0:
                p = None
        else:
            p = None
        if p is not None:
            p /= float(p.sum())
        new = int(np.random.choice(new_list, p=p))
        population.rewire(anchor, old, new)


if __name__ == '__main__':
    G = nx.random_regular_graph(5, 100)
    P = Population(G)
    P.fitness = np.random.randint(1, 3, size=100) * 1.0
    p = Preference()
    print P.edges(1)
    p.adapt_once(P, 1, 1)
    print P.edges(1)
