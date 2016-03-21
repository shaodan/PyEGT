# -*- coding: utf-8 -*-

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
        p = fitness / fitness.sum()
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
        p = fitness[neigh]
        if p.sum() == 0:
            p = None
        else:
            p = p / p.sum()
        birth = np.random.choice(neigh, replace=False, p=p)
        return birth, death


class IM(UpdateRule):
    def update(self, graph, fitness):
        pass


class MineUpdate(UpdateRule):
    def update(self, graph, fitness):
        birth = 0
        death = 0
        return birth, death

if __name__ == '__main__':
    G = nx.random_regular_graph(5, 100)
    f = np.random.randint(1, 3, size=100) * 1.0
    bd = BirthDeath()
    A = bd.update(G, f)
    print(A)
