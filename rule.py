# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2015.07.11 -*-

import numpy as np
import networkx as nx
from population import Population
import warnings


class Rule(object):

    def __init__(self):
        pass

    def bind(self, population):
        self.population = population
        self.fitness = population.fitness
        return self

    # 策略更新过程，必须继承
    def update(self):
        raise NotImplementedError( "Should have implemented" )


class BirthDeath(Rule):

    def update(self):
        p = self.fitness.clip(min=0)       # todo: PGG存在负收益，忽略这些节点
        p = p / p.sum()
        birth = np.random.choice(self.population.size, replace=False, p=p)
        neigh = self.population.neighbors(birth)
        death = np.random.choice(neigh, replace=False)
        return birth, death


class DeathBirth(Rule):

    def update(self):
        death = np.random.randint(self.population.size)
        neigh = self.population.neighbors(death)
        if len(neigh) == 0:
            print "====no neigh for node:"+str(death)+"===="
            return death, death
        p = self.fitness[neigh].clip(min=0)
        if p.sum() == 0:
            p = None
        else:
            p = p / p.sum()
        birth = np.random.choice(neigh, replace=False, p=p)
        return birth, death


class Imitation(Rule):

    def update(self):
        strategy = self.population.strategy.copy()
        # rand_array = np.random.random(self.population.size)
        for node in self.population.nodes_iter():
            neighbors = self.population.neighbors(node)
            max_fit = self.fitness[node]
            max_ind = node
            for n in neighbors:
                if self.fitness[n] > max_fit:
                    max_ind = n
                    max_fit = self.fitness[n]
            strategy[node] = strategy[n]
        self.population.strategy = strategy


class Fermi(Rule):

    def __init__(self, k=0.1):
        self.K = k
        np.seterr(over='warn')

    def update(self):
        birth, death = self.population.random_edge()
        # fermi转移概率公式
        probability = 1/(1+np.exp((self.fitness[death]-self.fitness[birth])/self.K))
        if np.random.random() > probability:
            # 更新失败
            death = birth
        return birth, death


class HeteroFermi(Rule):

    def __init__(self, delta):
        # delta = max(T, R) - min(S,P) > 0
        # for pd delta = T-S
        # for sd delta = T-P
        # for sh delta = R-S
        self.delta = delta

    def update(self):
        birth, death = self.population.random_edge()
        degree = max(self.population.degree_list[birth], self.population.degree_list[death])
        probability = (self.fitness[birth]-self.fitness[death])/(self.delta*degree)
        if np.random.random() > probability:
            death = birth
        return birth, death

if __name__ == '__main__':
    G = nx.random_regular_graph(5, 100)
    P = Population(G)
    P.fitness = np.random.randint(1, 3, size=100) * 1.0
    bd = BirthDeath().bind(P)
    A = bd.update()
    fermi = Fermi().bind(P)
    B = fermi.update()
    # im = Imitation().bind(P)
    # C = im.update()
    print A
    print G.has_edge(A[0], A[1])
    print B
    print G.has_edge(B[0], B[1])
