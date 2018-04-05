# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2015.07.11 -*-

import numpy as np


class Adapter(object):

    def __init__(self, category=3):
        self.category = category
        self.category_words = ['Random', 'Popularity', 'CNN', 'Pop*Sim', 'Similarity']
        self.population = None

    def bind(self, p):
        self.population = p
        return self

    def adapt(self, anchor):
        raise NotImplementedError("Adapter.adapt")


class LocalAdapter(Adapter):
    """ Local Adaptor, only choice in local area(distance < d_max) """
    def adapt(self, anchor, source=None):
        prefer = self.population.dynamic[anchor]
        if anchor is None or source is None:
            return None
        alter = self.population.neighbors_of_neighbors(source)
        if prefer == 0:
            new = np.random.choice(alter)
        elif prefer == 1:
            new = np.random.choice(alter)
        else:
            new = 1
        old = 0
        return old, new


class GlobalAdapter(Adapter):
    """ Global Adapter """
    def adapt(self, anchor, source=None):
        pass


class Preference(Adapter):
    """ Preference Adaptor """
    def adapt(self, anchor):
        prefer = self.population.dynamic[anchor]
        if anchor is None:
            return None
        if prefer == 1:  # 度优先
            p = np.array(self.population.degree_cache, dtype=np.float64)
        elif prefer == 2:  # 相似度
            p = np.array(self.population.number_of_cn(anchor), dtype=np.float64)
            p += 1  # 防止没有足够公共节点的
        else:
            p = None
        p[anchor] = 0
        p /= float(p.sum())
        new = np.random.choice(len(self.population), replace=False, p=p)
        old = self.population.rewire(anchor, new)
        return 0, 0 if old < 0 else old, new

    def adapt_once(self, anchor):
        # rewire only one link
        prefer = self.population.dynamic[anchor]
        new_list = self.population.nodes_nonadjacent(anchor)
        if prefer == 1:
            p = np.array(self.population.degree_cache[new_list], dtype=np.float64)
        elif prefer == 2:
            p = np.array(self.population.number_of_cn(anchor, new_list), dtype=np.float64)
            # p += 1  # 防止没有足够公共节点的
            if p.sum() == 0:
                p = None
        else:
            p = None
        if p is not None:
            p /= float(p.sum())
        new = np.random.choice(new_list, p=p)
        old = self.population.rewire(anchor, new)
        return None if old < 0 else old, new

    def adapt2(self, anchor, ma=None):
        # prefer = self.dynamic[anchor]
        # new = [self.rd, self.pa, self.cnn][prefer](anchor)
        new = self.lb(ma)
        old = self.population.rewire(anchor, new)
        return old, new

    def lb(self, ma):
        # todo learn best
        new = self.population.long_tie[ma]
        return new

    def rd(self, n):
        # random
        new = np.random.randint(len(self.population))
        if self.population.has_edge(n, new):
            new = np.random.randint(len(self.population))
        return new

    def pa(self, n):
        # degree
        # p = np.array(self.population.degree_cache, dtype=np.float64)
        # p /= float(p.sum())
        # return np.random.choice(len(self.population), p=p)
        u, v = self.population.random_edge()
        return np.random.choice((u, v))

    def cnn(self, n):
        # common neighbors
        nn = self.population.random_neighbor(n)
        new = self.population.random_neighbor(nn)
        return new


if __name__ == '__main__':
    import networkx as nx
    import population as pp
    G = nx.random_regular_graph(5, 100)
    DP = pp.DynamicPopulation(G)
    DP.fitness = np.random.randint(1, 3, size=100) * 1.0
    pp = Preference().bind(DP)
    print(DP.edges(1))
    pp.adapt_once(1)
    print(DP.edges(1))
