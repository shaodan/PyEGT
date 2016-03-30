# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2015.07.11 -*-

import numpy as np
import networkx as nx


class UpdateRule(object):

    # 策略更新过程，必须继承
    # graph网络结构，fitness收益数组
    def update(self, graph, fitness):
        pass


class BirthDeath(UpdateRule):

    def update(self, graph, fitness):
        size = len(graph)
        p = fitness.clip(min=0)
        p = p / p.sum()
        birth = np.random.choice(size, replace=False, p=p)
        neigh = graph.neighbors(birth)
        death = np.random.choice(neigh, replace=False)
        return birth, death


class DeathBirth(UpdateRule):

    def update(self, graph, fitness):
        size = len(graph)
        death = np.random.randint(size)
        neigh = graph.neighbors(death)
        if len(neigh) == 0:
            print "==========no neigh for node:"+death+"=========="
            return death, death
        p = fitness[neigh].clip(min=0)
        if p.sum() == 0:
            p = None
        else:
            p = p / p.sum()
        birth = np.random.choice(neigh, replace=False, p=p)
        return birth, death


class IM(UpdateRule):

    def update(self, graph, fitness):
        pass


class Fermi(UpdateRule):

    def __init__(self, k=0.1):
        self.K = k

    def update(self, graph, fitness):
        size = len(graph)
        # choice random pair in graph
        # edges = graph.edges()
        # size = len(edges)
        # birth, death = edges[np.random.randint(size)]
        birth, death = np.random.randint(size, size=2)
        while birth == death or (not graph.has_edge(birth, death)):
            birth, death = np.random.randint(size, size=2)
        if 1 / (1+np.exp((fitness[death]-fitness[birth])/self.K)) > np.random.random():
            death = birth
        return birth, death


class HeteroFermi(UpdateRule):

    def __init__(self, delta):
        # delta = max(T, R) - min(S,P) > 0
        # for pd delta = T-S
        # for sd delta = T-P
        # for sh delta = R-S
        self.delta = delta

    def update(self, graph, fitness):
        size = len(graph)
        # choice random pair in graph
        birth, death = np.random.randint(size, size=2)
        while birth == death or (not graph.has_edge(birth, death)):
            birth, death = np.random.randint(size, size=2)
        degree_b = graph.degree(birth)
        degree_d = graph.degree(death)
        if (fitness[death]-fitness[birth]) / (self.delta * max(degree_b, degree_d)) > np.random.random():
            death = birth
        return birth, death

if __name__ == '__main__':
    G = nx.random_regular_graph(5, 100)
    f = np.random.randint(1, 3, size=100) * 1.0
    bd = BirthDeath()
    A = bd.update(G, f)
    fermi = Fermi()
    B = fermi.update(G, f)
    print(A)
    print G.has_edge(A[0], A[1])
    print B
    print G.has_edge(B[0], B[1])