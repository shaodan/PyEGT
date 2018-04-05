# encoding=utf-8
"""Class for Custom Structured Network.

LatticeWithLongTie
"""
#    Copyright (C) 2016-2020 by
#    Shao Dan <shaodan.cn@gmail.com>
#    BSD license.
import numpy as np
import networkx as nx
from population import Population

__author__ = """\n""".join(['Shao Dan (shaodan.cn@gmail.com)'])


class LatticeWithLongTie(Population):

    def __init__(self, m=None, n=None, copy_from=None):
        if copy_from is not None:
            assert(isinstance(copy_from, LatticeWithLongTie))
            super(self.__class__, self).__init__(copy_from)
            self.fitness = np.empty(len(self), dtype=np.double)
            self.strategy = copy_from.strategy.copy()
            self.rate = 0
            self.dynamics = copy_from.dynamics.copy()
            self.dist = copy_from.dist
            self.long_tie = copy_from.long_tie.copy()
            self.degree_cache = copy_from.degree_cache.copy()
            self.edge_cache = copy_from.edge_cache[:]
            return

        if n is None:
            n = m
        self.m = m
        self.n = n
        graph = nx.grid_2d_graph(m, n, periodic=True)
        graph = nx.convert_node_labels_to_integers(graph)
        super(self.__class__, self).__init__(graph)
        self.long_tie = None
        # todo init after dynamics
        # self.init_longtie()
        self.dynamics = np.empty(len(self), dtype=np.int)
        self.dist = None
        self.category = 0

    def init_dynamics(self, adapter):
        # co-evolution dynamic, see adapter.py
        self.dynamics[:] = np.random.randint(adapter.category, size=len(self))
        # initial distribution
        self.dist = [(self.dynamics == m).sum() for m in xrange(adapter.category)]
        self.category = adapter.category
        # TODO: 优化
        # self.dist = [np.count_nonzero(self.dynamics == m) for m in range(adapter.category)]

    def prefer(self):
        self.dist = [np.count_nonzero(self.dynamics == m) for m in range(self.category)]
        return self.dist

    def copy(self):
        # init longtie 生成的degree_cache错误
        return LatticeWithLongTie(copy_from=self)

    def init_longtie(self):
        size = len(self)
        if self.long_tie is not None:
            for u, v in enumerate(self.long_tie):
                self.remove_edge(u, v)

        long_tie = np.random.randint(size, size=size, dtype=np.int)
        for u, v in enumerate(long_tie):
            # v = long_tie[u]
            while u == v or self.has_edge(u, v):
                v = np.random.randint(size)
            long_tie[u] = v
            self.add_edge(u, v)
            self.degree_cache[u] += 1
            self.degree_cache[v] += 1
        self.long_tie = long_tie
        return

        # random
        self.long_tie = np.random.randint(size, size=size, dtype=np.int)
        for u in np.where(self.dynamics==0)[0]:
            v = long_tie[u]
            while u == v or self.has_edge(u, v):
                v = np.random.randint(size)
            long_tie[u] = v
            self.add_edge(u, v)
            self.degree_cache[u] += 1
            self.degree_cache[v] += 1
        # cnn
        for u in np.where(self.dynamics==2)[0]:
            nn = self.random_neighbor(u)
            v = self.random_neighbor(nn)
            while u == v or self.has_edge(u, v):
                nn = self.random_neighbor(u)
                v = self.random_neighbor(nn)
            long_tie[u] = v
            self.add_edge(u, v)
            self.degree_cache[u] += 1
            self.degree_cache[v] += 1
        # pa
        for u in np.where(self.dynamics==1)[0]:
            v1, v2 = self.random_edge()
            v = np.random.choice((v1, v2))
            while u == v or self.has_edge(u, v):
                nn = self.random_neighbor(u)
                v = self.random_neighbor(nn)
            long_tie[u] = v
            self.add_edge(u, v)
            self.degree_cache[u] += 1
            self.degree_cache[v] += 1

    def rewire(self, u, v):
        if self.has_edge(u, v):
            return -1
        old_v = self.long_tie[u]
        # todo check new llc gain more fitness
        # if self.strategy[v] > self.strategy[old_v]:
        #     return -1
        self.remove_edge(u, old_v)
        self.degree_cache[old_v] -= 1
        self.long_tie[u] = v
        self.add_edge(u, v)
        self.degree_cache[v] += 1
        return old_v

    def is_equal(self, p):
        t_delta = (p.long_tie - self.long_tie).sum()
        d_delta = (p.degree_cache - self.degree_cache).sum()
        dy_delta = (p.dynamics - self.dynamics).sum()
        re = np.random.randint(len(self.edge_cache))
        print(t_delta, d_delta, dy_delta, p.edge_cache[re], self.edge_cache[re])

    def random_edge(self):
        total_edge_size = len(self.edge_cache) + len(self.long_tie)
        edge_index = np.random.choice(total_edge_size)
        if edge_index < len(self.edge_cache):
            return self.edge_cache[edge_index]
        u = edge_index - len(self.edge_cache)
        v = self.long_tie[u]
        # todo 无向图的边，返回节点考虑方向
        # if np.random.randint(2):
        #     return v, u
        return u, v
