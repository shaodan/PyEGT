# -*- coding: utf-8 -*-
# -*- Author: shaodan(shaodan.cn@gmail.com) -*-
# -*-  2015.07.11 -*-

import numpy as np
import networkx as nx
from population import Population


class Game(object):
    """base_class_of_game"""

    def __init__(self):
        self.rank = 2

    def bind(self, population):
        self.population = population
        self.fitness = population.fitness
        self.strategy = population.strategy
        return self

    # 博弈过程，必须继承
    def play(self, node_list=None, edge_list=None):
        if node_list is None:
            # 初始化fitness，整体进行一次博弈
            self.init_play()
        else:
            # 局部更新
            if not isinstance(node_list, list):
                node_list = [node_list]
            self.fast_play(node_list, edge_list)

    # 计算所有节点的fitness，必须被继承
    def init_play(self):
        raise NotImplementedError("Game.init_play() Should have implemented!")

    # 计算少数节点引起的变化，必须被继承
    def fast_play(self, node_list, edge_list=None):
        # todo : 会有精度差别 10^-16~-15数量级
        # 考虑一定循环之后整体计算修正
        raise NotImplementedError("Game.fast_play() Should have implemented!")

    def error(self, death):
        self.fast_play(death)
        fit_fast = self.fitness.copy()
        self.init_play()
        fit_init = self.fitness
        return (fit_fast-fit_init).sum()


class PGG(Game):
    """public_goods_game"""

    def __init__(self, r=2, fixed=False):
        super(PGG, self).__init__()
        # 获利倍数
        self.r = float(r)
        self.fixed = fixed

    def init_play(self):
        self.fitness.fill(0)
        degree_array = np.array(self.population.degree_list)
        # 第一种每个group投入1
        for node in self.population.nodes_iter():
            degree = degree_array[node]
            self.fitness[node] += (self.strategy[node]-1) * (degree+1)
            neighs = self.population.neighbors(node)
            neighs.append(node)
            # b = self.r * (self.strategy[neighs]==0).sum() / (degree+1)
            b = self.r * (len(neighs)-np.count_nonzero(self.strategy[neighs])) / (degree+1)
            for neigh in neighs:
                self.fitness[neigh] += b

    # def play(self, node, rewire_edge):
    #     for edge in rewire_edge:
    #         if edge[2] == 0:
    #             pass


    def fast_play(self, node_list, edge_list=None):
        for node in node_list:
            s = self.strategy[node]
            sign = (1 - 2*s)
            sign_r = sign * self.r
            d = self.population.degree_list[node]
            # 更新节点作为中心pgg产生的收益增量
            delta = sign_r/(d+1)
            self.fitness[node] += delta - sign*(d+1)
            for neigh in self.population.neighbors_iter(node):
                delta_neigh = sign_r/(self.population.degree_list[neigh]+1)
                self.fitness[neigh] += delta + delta_neigh
                for neigh_neigh in self.population.neighbors_iter(neigh):
                    self.fitness[neigh_neigh] += delta_neigh


class PGG2(PGG):
    """pgg 2"""

    def init_play(self):
        self.fitness.fill(0)
        degree_array = np.array(self.population.degree_list)
        # 第二种每个group投入1/(k+1)
        inv = (1.0-self.strategy) / (degree_array+1)
        for node in G.nodes_iter():
            self.fitness[node] += self.strategy[node] - 1
            neighs = G.neighbors(node)
            neighs.append(node)
            b = self.r * inv[neighs].sum() / (degree_array[node]+1)
            for neigh in neighs:
                self.fitness[neigh] += b

    def fast_play(self, node_list, edge_list=None):
        for node in node_list:
            s = self.strategy[node]
            d = self.population.degree_list[node]
            sign = (1 - 2*s)
            sign_r = sign * self.r / (d+1)
            # 更新节点作为中心pgg产生的收益增量
            delta = sign_r/(d+1)
            self.fitness[node] += delta - sign
            for neigh in self.population.neighbors_iter(node):
                delta_neigh = sign_r/(self.population.degree_list[neigh]+1)
                self.fitness[neigh] += delta + delta_neigh
                for neigh_neigh in self.population.neighbors_iter(neigh):
                    self.fitness[neigh_neigh] += delta_neigh


class PDG(Game):
    """prisoner's_dilemma_game"""

    def __init__(self, r=1, t=1.5, s=0, p=0.1):
        super(self.__class__, self).__init__()
        self.payoff = np.array([[(r, r), (s, t)], [(t, s), (p, p)]], dtype=np.double)
        self.delta = t-s

    def init_play(self):
        self.fitness.fill(0)
        for a, b in self.population.edges_iter():
            p = self.payoff[self.strategy[a]][self.strategy[b]]
            self.fitness[a] += p[0]
            self.fitness[b] += p[1]

    def fast_play(self, node_list, edge_list=None):
        # 只用计算新节点和其邻居节点的收益
        for edge in edge_list:
            a = edge[0]
            b = edge[1]
            p = self.payoff[self.strategy[a]][self.strategy[b]]
            if edge[2] == 0:
                self.fitness[a] -= p[0]
                self.fitness[b] -= p[0]
            else:
                self.fitness[a] += p[0]
                self.fitness[b] += p[0]
        for node in node_list:
            f = 0  # 新节点收益从0计算
            s = self.strategy[node]
            s_ = 1 - s
            for neigh in self.population.neighbors_iter(node):
                p = self.payoff[s][self.strategy[neigh]]
                f += p[0]           # 新节点累加
                new_payoff = p[1]   # 邻居节点计算新的收益
                # 0合作，1背叛
                p = self.payoff[s_][self.strategy[neigh]]
                old_payoff = p[1]   # 邻居节点计算原来的收益
                self.fitness[neigh] += new_payoff - old_payoff
            self.fitness[node] = f


class RPG(Game):
    """"rational_player_game"""

    def __init__(self, ration):
        super(self.__class__, self).__init__()
        self.ration = ration

    def init_play(self):
        pass

    def fast_play(self, node_list=None, edge_list=None):
        pass


class IPD(Game):

    def __init__(self):
        super(self.__class__, self).__init__()

    def init_play(self):
        pass

    def fast_play(self, node_list=None, edge_list=None):
        pass


# TEST CODE HERE
if __name__ == '__main__':
    G = nx.random_regular_graph(5, 100)
    P = Population(G)
    # g = PDG().bind(P)
    g = PGG2(2).bind(P)
    g.play()
    print P.fitness

    i = np.random.randint(100)
    P.strategy[i] = 1-P.strategy[i]
    fit1 = P.fitness
    fit2 = fit1.copy()

    g.play(i)
    g.fitness = fit2
    g.play()

    print "=======delta======="
    print (fit1-fit2)
    print "=======sum========="
    print (fit1-fit2).sum()

    print (fit1[i] - fit2[i])
    print G.degree(i)
