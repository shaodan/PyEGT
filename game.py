# -*- coding: utf-8 -*-
# -*- Author: shaodan(shaodan.cn@gmail.com) -*-
# -*-  2015.07.11 -*-

import numpy as np
import networkx as nx
from population import Population


class Game(object):
    """base_class_of_game"""

    def __init__(self):
        pass

    # 博弈过程，必须继承
    def play(self, population, node_list=None, edge_list=None):
        if node_list is None:
            # 初始化fitness，整体进行一次博弈
            pass
        elif node_list:
            # 局部更新
            self.fast_play(population, node_list, edge_list)

    def fast_play(self, population, node_list, edge_list=None):
        # todo : 会有精度差别 10^-16~-15数量级
        if not isinstance(node_list, list):
            # 先转换成list
            node_list = [node_list]
        for node in node_list:
            pass


class PGG(Game):
    """public_goods_game"""

    def __init__(self, r=2, fixed=False):
        super(self.__class__, self).__init__()
        # 获利倍数
        self.r = float(r)
        self.fixed = fixed

    def play(self, population, node_list=None, edge_list=None):
        # 可能会有负的fitness
        if node_list is None:
            population.fitness.fill(0)
            # 第一种每个group投入1
            degrees = np.array(population.degree().values())
            for node in population.nodes_iter():
                degree = degrees[node]
                population.fitness[node] += (population.strategy[node]-1) * (degree+1)
                neighs = population.neighbors(node)
                neighs.append(node)
                # b = self.r * (strategy[neighs]==0).sum() / (degree+1)
                b = self.r * (len(neighs)-np.count_nonzero(population.strategy[neighs])) / (degree+1)
                for neigh in neighs:
                    population.fitness[neigh] += b
            # 第二种每个group投入1/(k+1)
            # degrees = np.array(G.degree().values())
            # inv = (1.0-strategy) / (degrees+1)
            # for node in G.nodes_iter():
            #     fitness[node] += strategy[node] - 1
            #     neighs = G.neighbors(node)
            #     neighs.append(node)
            #     b = self.r * inv[neighs].sum() / (degrees[node]+1)
            #     for neigh in neighs:
            #         fitness[neigh] += b
        else:
            self.fast_play(population, node_list)

    def fast_play(self, population, node_list, edge_list=None):
        if not isinstance(node_list, list):
            node_list = [node_list]
        for node in node_list:
            s = population.strategy[node]
            sign = (1 - 2*s)
            sign_r = sign * self.r
            d = population.degree(node)
            # 更新节点作为中心pgg产生的收益增量
            delta = sign_r/(d+1)
            population.fitness[node] += delta - sign*(d+1)
            for neigh in population.neighbors_iter(node):
                delta_neigh = sign_r/(population.degree(neigh)+1)
                population.fitness[neigh] += delta + delta_neigh
                for neigh_neigh in population.neighbors_iter(neigh):
                    population.fitness[neigh_neigh] += delta_neigh


class PDG(Game):
    """prisoner's_dilemma_game"""

    def __init__(self, r=1, t=1.5, s=0, p=0.1):
        super(self.__class__, self).__init__()
        self.payoff = np.array([[(r, r), (s, t)], [(t, s), (p, p)]], dtype=np.double)
        self.delta = t-s

    def play(self, population, node_list=None, edge_list=None):
        if node_list is None:
            population.fitness.fill(0)
            for edge in population.edges_iter():
                a = edge[0]
                b = edge[1]
                p = self.payoff[population.strategy[a]][population.strategy[b]]
                population.fitness[a] += p[0]
                population.fitness[b] += p[1]
        else:
            self.fast_play(population, node_list)

    def fast_play(self, population, node_list, edge_list=None):
        if not isinstance(node_list, list):
            node_list = [node_list]
        # 只用计算新节点和其邻居节点的收益
        # 如果node_list为空list，则不更新
        for node in node_list:
            f = 0  # 新节点收益从0计算
            s = population.strategy[node]
            s_ = 1 - s
            for neigh in population.neighbors_iter(node):
                p = self.payoff[s][population.strategy[neigh]]
                f += p[0]           # 新节点累加
                new_payoff = p[1]   # 邻居节点计算新的收益
                # 0合作，1背叛
                p = self.payoff[s_][population.strategy[neigh]]
                old_payoff = p[1]   # 邻居节点计算原来的收益
                population.fitness[neigh] += new_payoff - old_payoff
            population.fitness[node] = f


class RPG(Game):
    name = "Rational Player Game"

    def __init__(self, ration):
        super(self.__class__, self).__init__()
        self.ration = ration

    def play(self, population, node_list=None, edge_list=None):
        pass


class IPD(Game):

    def __init__(self):
        super(self.__class__, self).__init__()

# TEST CODE HERE
if __name__ == '__main__':
    # g = PDG()
    g = PGG(2)
    G = nx.random_regular_graph(5, 100)
    P = Population(G)
    g.play(P)
    print P.fitness

    i = np.random.randint(100)
    P.strategy[i] = 1-P.strategy[i]
    fit1 = P.fitness
    fit2 = fit1.copy()

    g.play(P, i)
    P.fitness = fit2
    g.play(P)

    print "=======delta==========="
    print (fit1-fit2)
    print "=======sum==========="
    print (fit1-fit2).sum()

    print (fit1[i] - fit2[i])
    print G.degree(i)
