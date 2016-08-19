# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2015.07.11 -*-

import numpy as np
import networkx as nx
from population import Population


class Adapter(object):
    def __init__(self, category, mindegree=1):
        self.category = category
        self.min_degree = mindegree

    def bind(self, population):
        population.rbind_adapter(self)
        self.population = population
        self.dynamic = population.dynamic
        return self

    # prefer优先策略，anchor重连节点, source?
    def adapt(self, anchor, source=None):
        raise NotImplementedError("Game.init_play() Should have implemented!")


class LocalAdapter(Adapter):

    def adapt(self, anchor, source=None):
        size = self.population.size
        prefer = self.dynamic[anchor]
        if anchor is None or source is None:
            return None
        alter = self.population.neighbors_of_neighbors(source)
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

    def adapt(self, anchor, source=None):
        pass


class Preference(Adapter):

    def __init__(self, category=4):
        super(self.__class__, self).__init__(category)

    def adapt(self, anchor, source=None):
        population = self.population
        prefer = self.dynamic[anchor]
        if anchor is None:
            return None
        if prefer == 1:  # 度优先
            p = np.array(population.degree_list, dtype=np.float64)
        elif prefer == 2:  # 相似度
            p = np.array([len(list(nx.common_neighbors(population, anchor, x))) for x in population.nodes_iter()],
                         dtype=np.float64)
            p += 1         # 防止没有足够公共节点的
        else:
            p = None
        p[anchor] = 0
        p /= float(p.sum())
        old = np.random.choice(population.neighbors(anchor))
        new = np.random.choice(population.size, replace=False, p=p)
        population.remove_edge(anchor, old)
        population.add_edge(anchor, new)
        return old, new

    def adapt_once(self, anchor):
        # rewire only one link
        population = self.population
        prefer = self.dynamic[anchor]
        old = population.random_neighbor(anchor)
        if population.degree_list[old] <= self.min_degree:
            print "=======skip rewire(%d) neigh(%d)'s degree: %d===="%(anchor, old, population.degree(old))
            return 0, 0
        new_list = population.nodes_exclude_neighbors(anchor)
        if prefer == 1:
            p = np.array([population.degree_list[x] for x in new_list], dtype=np.float64)
        elif prefer == 2:
            p = np.array([len(list(nx.common_neighbors(population, anchor, x))) for x in new_list], dtype=np.float64)
            # p += 1       # 防止没有足够公共节点的
            if p.sum() == 0:
                p = None
        else:
            p = None
        if p is not None:
            p /= float(p.sum())
        new = int(np.random.choice(new_list, p=p))
        population.rewire(anchor, old, new)
        return old, new


if __name__ == '__main__':
    G = nx.random_regular_graph(5, 100)
    P = Population(G)
    P.fitness = np.random.randint(1, 3, size=100) * 1.0
    p = Preference().bind(P)
    print P.edges(1)
    p.adapt_once(1)
    print P.edges(1)
