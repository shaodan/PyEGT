# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2015.07.11 -*-

import numpy as np
import networkx as nx
from population import Population
import warnings


class Rule(object):

    # 策略更新过程，必须继承
    # graph网络结构，fitness收益数组
    def update(self, population):
        pass


class BirthDeath(Rule):

    def update(self, population):
        p = population.fitness.clip(min=0)       # todo: PGG存在负收益，忽略这些节点
        p = p / p.sum()
        birth = np.random.choice(population.size, replace=False, p=p)
        neigh = population.neighbors(birth)
        death = np.random.choice(neigh, replace=False)
        return birth, death


class DeathBirth(Rule):

    def update(self, population):
        death = np.random.randint(population.size)
        neigh = population.neighbors(death)
        if len(neigh) == 0:
            print "==========no neigh for node:"+death+"=========="
            return death, death
        p = population.fitness[neigh].clip(min=0)
        if p.sum() == 0:
            p = None
        else:
            p = p / p.sum()
        birth = np.random.choice(neigh, replace=False, p=p)
        return birth, death


class IM(Rule):

    def update(self, population):
        pass


class Fermi(Rule):

    def __init__(self, k=0.1):
        self.K = k
        np.seterr(over='warn')

    def update(self, population):
        size = population.size
        birth, death = np.random.randint(size, size=2)
        while birth == death or (not population.has_edge(birth, death)):
            birth, death = np.random.randint(size, size=2)
        if 1 / (1+np.exp((population.fitness[death]-population.fitness[birth])/self.K)) > np.random.random():
            death = birth
        return birth, death


class HeteroFermi(Rule):

    def __init__(self, delta):
        # delta = max(T, R) - min(S,P) > 0
        # for pd delta = T-S
        # for sd delta = T-P
        # for sh delta = R-S
        self.delta = delta

    def update(self, population):
        birth, death = population.random_edge()
        degree = max(population.degrees[birth], population.degrees[death])
        if (population.fitness[death]-population.fitness[birth]) / (self.delta * degree) > np.random.random():
            death = birth
        return birth, death

if __name__ == '__main__':
    G = nx.random_regular_graph(5, 100)
    P = Population(G)
    P.fitness = np.random.randint(1, 3, size=100) * 1.0
    bd = BirthDeath()
    A = bd.update(P)
    fermi = Fermi()
    B = fermi.update(P)
    print(A)
    print G.has_edge(A[0], A[1])
    print B
    print G.has_edge(B[0], B[1])
